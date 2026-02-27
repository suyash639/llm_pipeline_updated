"""
PII Redactor — Hybrid NER + regex masking engine with vault-based restore.

Sanitizes transcripts before they reach the LLM, then restores real values
in the final output.  The "Vault" (placeholder ↔ original mapping) is
a LOCAL-ONLY object — never sent to the LLM.

Detection engines (in priority order):
  1. Compiled regex  — credit cards, SSNs, phones, emails, Aadhaar, PAN, IDs
  2. SpaCy NER       — PERSON, ORG, GPE, LOC, DATE, MONEY
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import spacy
from spacy.language import Language


# ---------------------------------------------------------------------------
# PII Pattern definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PIIPattern:
    """A compiled regex pattern paired with a PII category label."""
    category: str
    pattern: re.Pattern


# Patterns are ordered so more-specific ones match first.
# All patterns use word-boundary or look-around anchors where practical.
PII_PATTERNS: List[PIIPattern] = [
    # Credit / debit card numbers (13-19 digits, optional separators)
    PIIPattern(
        category="CREDIT_CARD",
        pattern=re.compile(
            r"\b(?:\d[ -]*?){13,19}\b"
        ),
    ),
    # SSN  (US: 3-2-4 with separators)
    PIIPattern(
        category="SSN",
        pattern=re.compile(
            r"\b\d{3}[-.\s]\d{2}[-.\s]\d{4}\b"
        ),
    ),
    # Indian Aadhaar (12 digits, optional space every 4)
    PIIPattern(
        category="AADHAAR",
        pattern=re.compile(
            r"\b\d{4}\s?\d{4}\s?\d{4}\b"
        ),
    ),
    # Indian PAN  (AAAAA9999A)
    PIIPattern(
        category="PAN",
        pattern=re.compile(
            r"\b[A-Z]{5}\d{4}[A-Z]\b"
        ),
    ),
    # Email addresses
    PIIPattern(
        category="EMAIL",
        pattern=re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        ),
    ),
    # Phone numbers — international / Indian / US (7-15 digits with separators)
    PIIPattern(
        category="PHONE",
        pattern=re.compile(
            r"(?<!\w)"                       # not preceded by word char
            r"(?:\+?\d{1,3}[-.\s]?)?"        # optional country code
            r"(?:\(?\d{2,5}\)?[-.\s]?)?"     # optional area code
            r"\d{3,5}[-.\s]?\d{4}"           # main number
            r"(?!\w)"                        # not followed by word char
        ),
    ),
    # Reference / case / ticket IDs  (PREFIX-DIGITS)
    PIIPattern(
        category="REFERENCE_ID",
        pattern=re.compile(
            r"\b[A-Z]{2,5}-\d{3,10}\b"
        ),
    ),
]


# ---------------------------------------------------------------------------
# Span — a detected PII occurrence in the source text
# ---------------------------------------------------------------------------

@dataclass
class PIISpan:
    """A located PII entity in the source text."""
    start: int
    end: int
    category: str
    text: str


# ---------------------------------------------------------------------------
# Vault — bidirectional placeholder ↔ original mapping
# ---------------------------------------------------------------------------

class Vault:
    """
    Thread-local, session-scoped mapping between real PII values and
    anonymised placeholders.

    Guarantees *consistency*: the same original value always maps to the
    same placeholder within one transcript session.

    >>> v = Vault()
    >>> v.get_placeholder("John Doe", "PERSON")
    '[PERSON_1]'
    >>> v.get_placeholder("John Doe", "PERSON")  # same input → same output
    '[PERSON_1]'
    """

    def __init__(self) -> None:
        # original_text → placeholder
        self._forward: Dict[str, str] = {}
        # placeholder → original_text
        self._reverse: Dict[str, str] = {}
        # category → running counter
        self._counters: Dict[str, int] = defaultdict(int)

    def get_placeholder(self, original: str, category: str) -> str:
        """Return (and cache) a deterministic placeholder for *original*."""
        key = original.strip()
        if key in self._forward:
            return self._forward[key]

        self._counters[category] += 1
        placeholder = f"[{category}_{self._counters[category]}]"
        self._forward[key] = placeholder
        self._reverse[placeholder] = key
        return placeholder

    def to_dict(self) -> Dict[str, str]:
        """Export placeholder → original mapping (for storage / restore)."""
        return dict(self._reverse)

    @classmethod
    def from_dict(cls, mapping: Dict[str, str]) -> "Vault":
        """Reconstruct a Vault from a previously exported dict."""
        vault = cls()
        for placeholder, original in mapping.items():
            vault._reverse[placeholder] = original
            vault._forward[original] = placeholder
            # Rebuild counters
            # Placeholder format: [CATEGORY_N]
            inner = placeholder.strip("[]")
            parts = inner.rsplit("_", 1)
            if len(parts) == 2:
                cat, num = parts[0], int(parts[1])
                vault._counters[cat] = max(vault._counters[cat], num)
        return vault

    def __len__(self) -> int:
        return len(self._forward)

    def __repr__(self) -> str:
        return f"Vault({len(self)} entries)"


# ---------------------------------------------------------------------------
# Redactor — the main public interface
# ---------------------------------------------------------------------------

# SpaCy entity labels we care about → our category names
_NER_LABEL_MAP: Dict[str, str] = {
    "PERSON":   "PERSON",
    "ORG":      "ORG",
    "GPE":      "LOCATION",    # geo-political entity → LOCATION
    "LOC":      "LOCATION",
    "FAC":      "LOCATION",    # facility
    "DATE":     "DATE",
    "MONEY":    "MONEY",
}

# Minimum entity length to avoid single-char false positives
_MIN_ENTITY_LEN = 2


class Redactor:
    """
    Hybrid PII redaction engine.

    Usage
    -----
    >>> redactor = Redactor()
    >>> masked, vault = redactor.mask_transcript(
    ...     "Hi John, your card 4242-4242-4242-4242 was charged."
    ... )
    >>> masked
    'Hi [PERSON_1], your card [CREDIT_CARD_1] was charged.'
    >>> vault
    {'[PERSON_1]': 'John', '[CREDIT_CARD_1]': '4242-4242-4242-4242'}
    >>> redactor.unmask_summary(masked, vault)
    'Hi John, your card 4242-4242-4242-4242 was charged.'
    """

    def __init__(self, spacy_model: str = "en_core_web_sm") -> None:
        """
        Parameters
        ----------
        spacy_model : str
            Any installed SpaCy pipeline.  Use ``en_core_web_trf`` for
            maximum accuracy (requires PyTorch).
        """
        self._nlp: Language = spacy.load(spacy_model)

    # ------------------------------------------------------------------
    # Internal detection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_regex(text: str) -> List[PIISpan]:
        """Run all compiled regex patterns, return non-overlapping spans."""
        spans: List[PIISpan] = []
        occupied: set = set()  # character positions already claimed

        for pii in PII_PATTERNS:
            for match in pii.pattern.finditer(text):
                positions = set(range(match.start(), match.end()))
                if positions & occupied:
                    continue  # skip overlapping hit
                occupied |= positions
                spans.append(PIISpan(
                    start=match.start(),
                    end=match.end(),
                    category=pii.category,
                    text=match.group(),
                ))
        return spans

    def _detect_ner(self, text: str, occupied: set) -> List[PIISpan]:
        """Run SpaCy NER, skip entities that overlap with regex hits."""
        doc = self._nlp(text)
        spans: List[PIISpan] = []
        for ent in doc.ents:
            category = _NER_LABEL_MAP.get(ent.label_)
            if category is None:
                continue
            if len(ent.text.strip()) < _MIN_ENTITY_LEN:
                continue
            positions = set(range(ent.start_char, ent.end_char))
            if positions & occupied:
                continue  # regex already handled this region
            spans.append(PIISpan(
                start=ent.start_char,
                end=ent.end_char,
                category=category,
                text=ent.text,
            ))
        return spans

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mask_transcript(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Detect and mask all PII in *text*.

        Returns
        -------
        masked_text : str
            The anonymised transcript with placeholders like ``[PERSON_1]``.
        vault_dict : dict
            Mapping of ``placeholder → original`` for later restoration.
            **This dict must NEVER be sent to the LLM.**
        """
        vault = Vault()

        # 1. Regex pass (higher priority)
        regex_spans = self._detect_regex(text)
        occupied = set()
        for s in regex_spans:
            occupied |= set(range(s.start, s.end))

        # 2. NER pass (fills gaps)
        ner_spans = self._detect_ner(text, occupied)

        # 3. Merge and sort right-to-left (preserves offsets during replacement)
        all_spans = sorted(
            regex_spans + ner_spans,
            key=lambda s: s.start,
            reverse=True,
        )

        # 4. Replace spans
        masked = text
        for span in all_spans:
            placeholder = vault.get_placeholder(span.text, span.category)
            masked = masked[:span.start] + placeholder + masked[span.end:]

        return masked, vault.to_dict()

    @staticmethod
    def unmask_summary(
        summary_text: str,
        vault_dict: Dict[str, str],
    ) -> str:
        """
        Restore real PII values in the LLM-generated *summary_text*
        using the vault produced during masking.
        """
        result = summary_text
        # Sort by placeholder length descending → avoids substring collisions
        for placeholder, original in sorted(
            vault_dict.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            result = result.replace(placeholder, original)
        return result

    @classmethod
    def rehydrate(cls, data: Any, vault_dict: Dict[str, str]) -> Any:
        """
        Recursively restore real values in a JSON-like structure (dict, list, str).
        """
        if isinstance(data, str):
            return cls.unmask_summary(data, vault_dict)
        elif isinstance(data, list):
            return [cls.rehydrate(item, vault_dict) for item in data]
        elif isinstance(data, dict):
            return {k: cls.rehydrate(v, vault_dict) for k, v in data.items()}
        return data
