"""
Deterministic grader: grade(action, expected) -> float in [0.0, 1.0]
"""

CATEGORY_WEIGHT = 0.4
PRIORITY_WEIGHT = 0.3
RESPONSE_WEIGHT = 0.3

EMPTY_RESPONSE_PENALTY = 0.2


def grade_response_quality(response: str, keywords: list[str]) -> float:
    """Score response 0.0-1.0 based on keyword presence."""
    if not response or not response.strip():
        return 0.0
    response_lower = response.lower()
    matched = sum(1 for kw in keywords if kw.lower() in response_lower)
    return round(matched / len(keywords), 4) if keywords else 0.5


def grade(action: dict, expected: dict) -> float:
    """
    action: dict with keys category, priority, response
    expected: dict with keys expected_category, expected_priority, response_keywords
    Returns a score between 0.0 and 1.0.
    """
    score = 0.0

    # Category score
    if action.get("category", "").lower() == expected["expected_category"].lower():
        score += CATEGORY_WEIGHT

    # Priority score
    if action.get("priority", "").lower() == expected["expected_priority"].lower():
        score += PRIORITY_WEIGHT

    # Response quality
    response = action.get("response", "")
    if not response or not response.strip():
        score -= EMPTY_RESPONSE_PENALTY
    else:
        quality = grade_response_quality(response, expected.get("response_keywords", []))
        score += RESPONSE_WEIGHT * quality

    return round(max(0.0, min(1.0, score)), 4)
