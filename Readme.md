# 🚀 Call Analyst Pipeline

A resilient, privacy-first asynchronous pipeline designed to process customer call transcripts using **Groq (Llama 3.1)**. This system is engineered to handle large-scale batches on restricted API tiers while maintaining 100% data integrity and PII security.

---

# 🛠️ Core Engineering Features

This pipeline is engineered for high-reliability data processing, specifically optimized for privacy-conscious environments and rate-limited API tiers.

---

### 🔒 1. Zero-Leak PII Security (Vaulting)
The system ensures that sensitive information never touches the cloud, maintaining a strict local-only boundary for identifiable data.
* **Reversible Pseudonymization:** Implements a local "Vault" system that intercepts PII (Names, Phone Numbers, Case IDs) at the edge before transmission.
* **Tokenization:** Sensitive data is replaced with unique, deterministic tokens (e.g., `[REF_001]`). The LLM processes only these abstract placeholders.
* **Rehydration:** Real values are re-inserted into the final JSON output locally. This ensures final reports are actionable while keeping the LLM entirely "blind" to sensitive info.

### 🛡️ 2. Resilient Persistence Engine
Built to survive crashes, network timeouts, and power failures without data corruption.
* **Atomic Checkpointing:** State is preserved on a per-call basis. If a process is interrupted, the engine resumes exactly where it left off, preventing duplicate API costs.
* **Write-Rotate Safety:** Utilizes a temporary file system and `.bak` backup rotation during the write phase to prevent file truncation or loss.
* **Verified Cleanup:** Backup files are only purged after a successful post-run **JSON Health Check** confirms the integrity of the primary output.

### ⚡ 3. High-Concurrency & Throttling
Aggressive performance optimization tuned for Free-Tier or constrained API environments.
* **TPM/RPM Buffering:** Custom orchestration logic utilizing `asyncio.Semaphore` to maintain a steady state just below the **6k TPM** limit, preventing `429 (Too Many Requests)` errors.
* **Hot-Reloading:** The pipeline re-scans source files during every iteration, allowing developers to add, remove, or edit batch items in real-time without stopping the engine.

### 🔍 4. Automated Integrity Auditing
A secondary validation layer to enforce factual consistency and structural accuracy.
* **Hallucination Detection:** A secondary validation script cross-references every AI-generated ID against the original "Source of Truth" transcripts.
* **Deterministic Validation:** Specifically catches edge cases where the model may "invent" data or misidentify currency, dates, or addresses as reference IDs.

---
## 📂 Project Structure

```plaintext
llm-pipeline-creation/
├── data/
│   └── sample_calls.json      # Raw Input Transcripts
├── outputs/
│   ├── final_analysis.json    # Final Processed Data
│   └── final_analysis.json.bak # Atomic Backup
├── src/
│   ├── main.py                # Pipeline Orchestrator
│   ├── redactor.py            # Security & Vaulting Logic
└── logs/                      # Audit Trails
```
### Installation
1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd llm-pipeline-creation
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment:**
    Create a `.env` file in the root directory:
    ```ini
    OPENAI_API_KEY=gsk_your_groq_api_key_here
    ```
### Running the Pipeline
To process the dataset (located in `data/sample_calls.json`):

```bash
    python -m src.main