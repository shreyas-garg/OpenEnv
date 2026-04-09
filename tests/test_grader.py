"""Unit tests for the deterministic grader."""

from email_env.grader import grade, grade_response_quality

GOOD_RESPONSE = (
    "We sincerely apologize for the duplicate charge on your account. "
    "Our billing team will investigate immediately and issue a refund "
    "within 3-5 business days. We will resolve this for you."
)

EXPECTED = {
    "expected_category": "billing",
    "expected_priority": "high",
    "response_keywords": ["refund", "apologize", "billing", "resolve", "account"],
}


def test_perfect_action_scores_high():
    score = grade(
        {"category": "billing", "priority": "high", "response": GOOD_RESPONSE},
        EXPECTED,
    )
    assert score >= 0.9


def test_all_wrong_scores_zero():
    score = grade(
        {"category": "general", "priority": "low", "response": ""},
        EXPECTED,
    )
    assert score == 0.0


def test_empty_response_triggers_penalty():
    score = grade(
        {"category": "billing", "priority": "high", "response": ""},
        EXPECTED,
    )
    # 0.4 + 0.3 - 0.2 penalty = 0.5
    assert score == 0.5


def test_correct_category_only():
    score = grade(
        {"category": "billing", "priority": "low", "response": GOOD_RESPONSE},
        EXPECTED,
    )
    assert 0.4 < score < 0.8


def test_correct_priority_only():
    score = grade(
        {"category": "general", "priority": "high", "response": GOOD_RESPONSE},
        EXPECTED,
    )
    assert 0.3 < score < 0.7


def test_keyword_stuffing_is_penalised():
    """Adversarial: just spam keywords. Should NOT score well."""
    spam = "refund refund refund refund refund refund refund refund"
    score = grade_response_quality(spam, EXPECTED["response_keywords"])
    assert score < 0.4, f"keyword stuffing scored {score}"


def test_no_punctuation_is_penalised():
    text = "we apologize and will refund your billing account resolve issue"
    score = grade_response_quality(text, EXPECTED["response_keywords"])
    # has all keywords + structure but no punctuation -> coherence drops
    assert score < 0.95


def test_short_response_is_penalised():
    text = "ok refund."
    score = grade_response_quality(text, EXPECTED["response_keywords"])
    assert score < 0.5


def test_all_caps_is_penalised():
    shout = "WE WILL APOLOGIZE AND REFUND YOUR BILLING ACCOUNT RESOLVE IMMEDIATELY."
    normal = "We will apologize and refund your billing account resolve immediately."
    shout_score = grade_response_quality(shout, EXPECTED["response_keywords"])
    normal_score = grade_response_quality(normal, EXPECTED["response_keywords"])
    assert shout_score < normal_score


def test_good_response_without_all_keywords_still_scores():
    text = (
        "We are very sorry for the trouble. Our team will look into this "
        "and contact you shortly with a resolution."
    )
    score = grade_response_quality(text, EXPECTED["response_keywords"])
    # Has structure (ack + action) but few keywords
    assert score > 0.4


def test_missing_acknowledgement_lowers_score():
    text = "Our team will refund your account and resolve the billing issue."
    score = grade_response_quality(text, EXPECTED["response_keywords"])
    # Has action but no acknowledgement -> structural component halved
    perfect = grade_response_quality(GOOD_RESPONSE, EXPECTED["response_keywords"])
    assert score < perfect


def test_score_is_in_range():
    for cat in ("billing", "technical", "general"):
        for pri in ("low", "medium", "high"):
            for resp in ("", "x", GOOD_RESPONSE, "refund " * 50):
                s = grade(
                    {"category": cat, "priority": pri, "response": resp},
                    EXPECTED,
                )
                assert 0.0 <= s <= 1.0, f"out of range: {s}"


def test_grader_is_deterministic():
    a = {"category": "billing", "priority": "high", "response": GOOD_RESPONSE}
    s1 = grade(a, EXPECTED)
    s2 = grade(a, EXPECTED)
    s3 = grade(a, EXPECTED)
    assert s1 == s2 == s3
