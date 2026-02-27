import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from groq import AsyncGroq
from src.redactor import Redactor
from loguru import logger
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

# Configuration
INPUT_FILE = Path("data/sample_calls.json")
OUTPUT_FILE = Path("outputs/final_analysis.json")
LOG_FILE = Path("logs/pipeline.log")
MAX_CONCURRENT_TASKS = 1
MODEL_NAME = "llama-3.1-8b-instant"

# Automated Directory Setup
os.makedirs("logs/dlq", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Persistent Logging Configuration
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
logger.add(LOG_FILE, rotation="7 days", format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}")

SYSTEM_PROMPT = """You are a Senior Strategic Analyst. Your task is to perform a high-depth extraction of call transcript data into a specific JSON schema.

Today is 2026-02-27. 

Your task is to extract data from the call transcript into a JSON object.

### ANALYTICAL REQUIREMENTS
1. **Summary:** Use exactly 1-3 sentences. Format: [Call Objective] + [Key Conflict/Event] + [Final Resolution/Status].
2. **Sentiment Velocity:** Do not just give an average. If the customer ends the call more frustrated than they started, mark as "negative" even if they started "positive."
3. **Entity Extraction:** 
   - Convert relative dates (e.g., "next Monday") to YYYY-MM-DD based on Feb 27, 2026.
   - Extract all INR amounts as floats.
4. **Risk Flags:** Be aggressive. If a customer mentions "moving to a competitor" or "lawyer," trigger `customer_churn_risk`.

You MUST strictly follow this exact JSON schema:

{
  "call_id": "string (e.g., 'CALL-0001')",
  "category": "tech_support" | "billing" | "plan_change" | "cancellation" | "privacy" | "other",
  "summary": "string (exactly 1-3 sentences in the specified format)",
  "customer_sentiment": "negative" | "neutral" | "positive" | "mixed",
  "resolution_status": "resolved" | "pending" | "escalated",
  "actions_taken": [
    { "type": "ticket_created" | "technician_scheduled" | "credit_applied" | "plan_activated" | "email_sent" | "sms_sent" | "other", "details": "string" }
  ],
  "followups_required": [
    { "owner": "agent" | "billing_team" | "field_team" | "security_team" | "customer", "due": "YYYY-MM-DD", "details": "string" }
  ],
  "risk_flags": [ "possible_privacy_issue", "customer_churn_risk", "missed_appointment", "billing_dispute", "outage", "none" ],
  "entities": {
    "case_ids": ["string"],
    "plan_prices_inr": [float],
    "time_windows": ["string"],
    "dates": ["YYYY-MM-DD"]
  },
  "confidence": float (between 0.0 and 1.0)
}

Ensure 'call_id' is extracted exactly as it appears in the input (e.g., 'CALL-0001', not '1').
Return ONLY valid JSON.

### IMPORTANT: REHYDRATION PROTOCOL
- You will see bracketed tokens like [REFERENCE_ID_1], [ORG_1], or [PERSON_1] in the transcript.
- These are pseudonymized entities.
- DO NOT attempt to guess, change, or remove these tokens.
- Copy them EXACTLY as they appear into the relevant fields of the JSON output.
"""

class AuditResilientProcessor:
    def __init__(self):
        self.redactor = Redactor()
        self.api_key = os.getenv("OPENAI_API_KEY")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), reraise=True)
    async def process_one(self, client: AsyncGroq, call_data: Dict[str, Any]) -> Dict[str, Any]:
        call_id = call_data.get("call_id")
        transcript = call_data.get("transcript", "")
        
        # 1. Redaction
        masked_text, vault = self.redactor.mask_transcript(transcript)
        
        try:
            # 2. LLM Call
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Call ID: {call_id}\n\nTranscript:\n{masked_text}"}
                ],
                response_format={"type": "json_object"},
                temperature=0
            )
            
            analysis = json.loads(response.choices[0].message.content)
            
            # 3. Rehydrate JSON (Restore original values from Vault)
            rehydrated_analysis = self.redactor.rehydrate(analysis, vault)
            
            return rehydrated_analysis
        except Exception as e:
            logger.error(f"Error processing {call_id}: {str(e)}")
            raise

    def load_state(self) -> List[Dict[str, Any]]:
        """Smart Rerun: Load existing results."""
        if OUTPUT_FILE.exists():
            try:
                with open(OUTPUT_FILE, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Corrupt output file {OUTPUT_FILE}. Starting fresh.")
                return []
        return []

    def save_state(self, results: List[Dict[str, Any]]):
        """Atomic Persistence: Save progress immediately."""
        with open(OUTPUT_FILE, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    async def run(self):
        if not INPUT_FILE.exists():
            logger.error(f"Input file {INPUT_FILE} not found. Terminating.")
            return

        # 1. Initial Load to establish the working queue
        try:
            with open(INPUT_FILE, "r") as f:
                initial_calls = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read source file: {e}")
            return

        results = self.load_state()
        processed_ids = {res["call_id"] for res in results}
        queue = [c for c in initial_calls if c["call_id"] not in processed_ids]
        
        if not queue:
            logger.info("Source file exhausted or cleared. Shutting down.")
            self.finalize_run(len(results), len(initial_calls))
            return

        async with AsyncGroq(api_key=self.api_key) as client:
            for i, target_call in enumerate(queue, start=len(results) + 1):
                # 2. Per-Iterative Loading (Requirement 1)
                try:
                    with open(INPUT_FILE, "r") as f:
                        current_disk_calls = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    current_disk_calls = []

                # 3. Dynamic Stop (Requirement 3)
                if not current_disk_calls:
                    logger.info("Source file exhausted or cleared. Shutting down.")
                    break

                # 4. Existence Check & Memory Refresh (Requirement 2 & 4)
                call_id = target_call["call_id"]
                disk_call_data = next((c for c in current_disk_calls if c["call_id"] == call_id), None)
                
                if not disk_call_data:
                    logger.warning(f"Warning: Call {call_id} removed from source. Skipping.")
                    continue

                logger.info(f"Progress: [{i}/{len(initial_calls)}] - Processing {call_id}...")
                
                try:
                    analysis = await self.process_one(client, disk_call_data)
                    results.append(analysis)
                    
                    # Atomic Save
                    self.save_state(results)
                    logger.success(f"Successfully processed {call_id}. Progress saved.")
                    
                    # TPM Buffer (Deliberate Sleep)
                    if i < len(initial_calls):
                        delay = 12
                        logger.info(f"TPM Buffer: Sleeping {delay}s to avoid 429...")
                        await asyncio.sleep(delay)
                        
                except Exception as e:
                    logger.error(f"FATAL: Permanent failure for {call_id}: {e}")
                    with open("logs/dlq/failed_calls.log", "a") as f:
                        f.write(f"{call_id}: {str(e)}\n")

        # 5. Finalize and Cleanup
        # Recalculate total for final verification based on current disk state
        try:
            with open(INPUT_FILE, "r") as f:
                final_disk_calls = json.load(f)
            final_total = len(final_disk_calls)
        except:
            final_total = 0
            
        self.finalize_run(len(results), final_total)

    def finalize_run(self, total_processed: int, total_expected: int):
        """Perform health check and cleanup backup."""
        logger.info(f"Finished processing. Total results: {total_processed}")

        if total_processed == total_expected and total_expected > 0:
            logger.info("Audit: Performing final health check on output...")
            try:
                with open(OUTPUT_FILE, "r") as f:
                    json.load(f)
                
                bak_file = OUTPUT_FILE.with_suffix(".json.bak")
                if bak_file.exists():
                    try:
                        os.remove(bak_file)
                        logger.success("Audit: Verification passed. Deleting backup file for a clean workspace.")
                    except Exception as e:
                        logger.warning(f"Cleanup Error: Could not delete backup file: {e}")
                else:
                    logger.success("Audit: Verification passed. Workspace is clean.")
            except Exception as e:
                logger.error(f"Health Check Failed: Output integrity compromised. KEEPING backup file. Error: {e}")
        else:
            logger.warning(f"Incomplete/Modified Run: [{total_processed}/{total_expected}] calls processed. KEEPING backup file for safety.")

if __name__ == "__main__":
    processor = AuditResilientProcessor()
    asyncio.run(processor.run())