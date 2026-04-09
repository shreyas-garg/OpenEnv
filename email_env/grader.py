"""
Deterministic hybrid grader: grade(action, expected) -> float in [0.0, 1.0]

Response quality combines three signals to defeat keyword-stuffing exploits:
  1. Keyword coverage (40%) — distinct keywords present
  2. Length & coherence sanity (20%) — penalises too-short / rambling / no-punctuation
  3. Structural requirement (40%) — must contain BOTH an acknowledgement
     (apology / acknowledge) AND an action/timeline phrase
"""

import re

CATEGORY_WEIGHT = 0.4
PRIORITY_WEIGHT = 0.3
RESPONSE_WEIGHT = 0.3

EMPTY_RESPONSE_PENALTY = 0.2

ACKNOWLEDGEMENT_TERMS = (
    "sorry", "apolog", "regret", "understand", "acknowledge", "thank"
)
ACTION_TERMS = (
    "will", "shall", "investigat", "refund", "resolv", "fix", "escalat",
    "team", "contact", "process", "issue", "follow up", "update", "check",
    "review", "look into", "assist", "help"
)

MIN_RESPONSE_LEN = 20
MAX_RESPONSE_LEN = 600


def _keyword_coverage(response_lower: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.5
    matched = sum(1 for kw in keywords if kw.lower() in response_lower)
    return matched / len(keywords)


def _length_coherence(response: str) -> float:
    """Sanity score based on length, punctuation and lexical diversity."""
    text = response.strip()
    n = len(text)
    if n == 0:
        return 0.0

    score = 1.0
    # length bounds
    if n < MIN_RESPONSE_LEN:
        score *= 0.4
    elif n > MAX_RESPONSE_LEN:
        score *= 0.6

    # must contain at least one sentence terminator
    if not re.search(r"[.!?]", text):
        score *= 0.5

    # detect keyword stuffing: repeated identical tokens dominate text
    tokens = re.findall(r"[A-Za-z']+", text.lower())
    if tokens:
        unique_ratio = len(set(tokens)) / len(tokens)
        if unique_ratio < 0.5:
            score *= 0.4

    # penalise SHOUTING / no spaces
    letters = [c for c in text if c.isalpha()]
    if letters and sum(1 for c in letters if c.isupper()) / len(letters) > 0.7:
        score *= 0.5

    return max(0.0, min(1.0, score))


def _structural_requirement(response_lower: str) -> float:
    has_ack = any(term in response_lower for term in ACKNOWLEDGEMENT_TERMS)
    has_action = any(term in response_lower for term in ACTION_TERMS)
    if has_ack and has_action:
        return 1.0
    if has_ack or has_action:
        return 0.5
    return 0.0


def grade_response_quality(response: str, keywords: list[str]) -> float:
    """Hybrid response score 0.0–1.0 resistant to keyword stuffing."""
    if not response or not response.strip():
        return 0.0
    response_lower = response.lower()

    coverage = _keyword_coverage(response_lower, keywords)
    coherence = _length_coherence(response)
    structure = _structural_requirement(response_lower)

    quality = 0.40 * coverage + 0.20 * coherence + 0.40 * structure
    return round(max(0.0, min(1.0, quality)), 4)


def grade(action: dict, expected: dict) -> float:
    """
    action: dict with keys category, priority, response
    expected: dict with keys expected_category, expected_priority, response_keywords
    Returns a score between 0.0 and 1.0.
    """
    score = 0.0

    if action.get("category", "").lower() == expected["expected_category"].lower():
        score += CATEGORY_WEIGHT

    if action.get("priority", "").lower() == expected["expected_priority"].lower():
        score += PRIORITY_WEIGHT

    response = action.get("response", "")
    if not response or not response.strip():
        score -= EMPTY_RESPONSE_PENALTY
    else:
        quality = grade_response_quality(response, expected.get("response_keywords", []))
        score += RESPONSE_WEIGHT * quality

    return round(max(0.0, min(1.0, score)), 4)
