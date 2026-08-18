from __future__ import annotations

import re


# ============================================================
# Critical Token Patterns
# ============================================================

NUMERIC_PATTERN = re.compile(r"\b(?:\d+(?:\.\d+)?|\d+st|\d+nd|\d+rd|\d+th|%\d+|\d+%)\b", re.IGNORECASE)
YEAR_PATTERN = re.compile(r"\b(19\d{2}|20\d{2})\b")
NEGATION_WORDS = {"not", "no", "never", "n't", "cannot", "cant", "wont", "dont", "isnt", "arent", "without", "hardly", "scarcely", "barely"}
QUESTION_STARTERS = {"why", "how", "what", "where", "when", "who", "whom", "which", "whose", "can", "could", "should", "would", "is", "are", "do", "does", "did"}


def extract_numbers(text: str) -> set[str]:
    """Extract numeric tokens, percentages, and ordinals."""
    return set(NUMERIC_PATTERN.findall(text.lower()))


def extract_years(text: str) -> set[str]:
    """Extract 4-digit year tokens."""
    return set(YEAR_PATTERN.findall(text))


def extract_capitalized_tokens(raw_text: str) -> set[str]:
    """
    Extract proper nouns / capitalized keywords from raw text (ignoring first word).
    Useful for catching entity substitutions like IITD vs IITG, Delhi vs Bombay.
    """
    words = raw_text.strip().split()
    if len(words) <= 1:
        return set()
    
    # Check words after first word for capitalization
    cap_words = {
        w.strip("?,.!;:\'\"()[]{}")
        for w in words[1:]
        if w and w[0].isupper() and w.lower() not in {"i", "a", "an", "the"}
    }
    return {w.lower() for w in cap_words if len(w) > 1}


def has_negation(tokens: list[str]) -> bool:
    """Check if token list contains negation markers."""
    return any(t.lower() in NEGATION_WORDS or t.lower().endswith("n't") for t in tokens)


def extract_question_starter(tokens: list[str]) -> str | None:
    """Identify the primary question starter word."""
    for t in tokens[:3]:
        clean_t = t.lower().strip("?,.!;:")
        if clean_t in QUESTION_STARTERS:
            return clean_t
    return None


# ============================================================
# Critical Token Diagnostic Evaluation
# ============================================================

def evaluate_critical_tokens(q1_raw: str, q2_raw: str) -> dict[str, bool | float | list[str]]:
    """
    Analyzes critical information-changing tokens between question pair.
    Used for explainability, hard-case diagnostics, and decision verification.
    """
    t1 = q1_raw.split()
    t2 = q2_raw.split()

    # 1. Numeric analysis
    nums1 = extract_numbers(q1_raw)
    nums2 = extract_numbers(q2_raw)
    num_mismatch = bool((nums1 or nums2) and (nums1 != nums2))
    num_overlap = len(nums1 & nums2) / max(len(nums1 | nums2), 1) if (nums1 or nums2) else 1.0

    # 2. Entity / Proper Noun analysis
    caps1 = extract_capitalized_tokens(q1_raw)
    caps2 = extract_capitalized_tokens(q2_raw)
    entity_mismatch = bool((caps1 or caps2) and (caps1 != caps2))

    # 3. Negation analysis
    neg1 = has_negation(t1)
    neg2 = has_negation(t2)
    negation_mismatch = bool(neg1 != neg2)

    # 4. Question intent starter
    q_starter1 = extract_question_starter(t1)
    q_starter2 = extract_question_starter(t2)
    question_type_mismatch = bool(
        q_starter1 and q_starter2 and q_starter1 != q_starter2 and 
        not ({q_starter1, q_starter2} <= {"what", "which"} or {q_starter1, q_starter2} <= {"can", "could", "how"})
    )

    return {
        "numeric_mismatch": num_mismatch,
        "numeric_overlap": float(num_overlap),
        "entity_mismatch": entity_mismatch,
        "negation_mismatch": negation_mismatch,
        "question_type_mismatch": question_type_mismatch,
        "q1_numbers": sorted(list(nums1)),
        "q2_numbers": sorted(list(nums2)),
        "q1_entities": sorted(list(caps1)),
        "q2_entities": sorted(list(caps2)),
    }
