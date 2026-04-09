"""
Task definitions for the Email Triage environment.

Difficulty labels reflect how reliably a small/medium LLM baseline solves them.
Tasks are ordered easy → hard along the difficulty axis.
"""

TASKS = {
    "task_1": {
        "id": "task_1",
        "difficulty": "easy",
        "description": "Simple general inquiry about business hours and live chat",
        "email_text": (
            "Hi there, I was wondering what your customer support hours are. "
            "I tried calling yesterday evening but nobody picked up. "
            "Could you also let me know if you have a live chat option? Thanks!"
        ),
        "sender_type": "customer",
        "expected_category": "general",
        "expected_priority": "low",
        "response_keywords": ["hours", "available", "chat", "support", "help"],
    },
    "task_2": {
        "id": "task_2",
        "difficulty": "easy",
        "description": "Clear billing issue — duplicate charge with obvious classification",
        "email_text": (
            "Hi, I was charged twice for my subscription this month. "
            "My account number is 84729 and both charges appeared on March 15th. "
            "Please issue a refund for the duplicate charge as soon as possible. "
            "Thank you."
        ),
        "sender_type": "customer",
        "expected_category": "billing",
        "expected_priority": "high",
        "response_keywords": ["refund", "apologize", "billing", "resolve", "account"],
    },
    "task_3": {
        "id": "task_3",
        "difficulty": "medium",
        "description": "Multi-issue complaint requiring correct classification and tone",
        "email_text": (
            "I am absolutely furious! For the third time this week your app keeps "
            "crashing every time I try to export my data. On top of that, you charged "
            "me a premium fee for a feature that is completely broken! I have deadlines "
            "to meet and your incompetent support team has not responded in 48 hours. "
            "If this is not fixed TODAY I will be disputing the charge with my bank "
            "and leaving a public review. This is completely unacceptable."
        ),
        "sender_type": "customer",
        "expected_category": "technical",
        "expected_priority": "high",
        "response_keywords": ["sorry", "apologize", "urgent", "escalate", "refund", "fix", "priority"],
    },
    "task_4": {
        "id": "task_4",
        "difficulty": "medium",
        "description": "Internal system alert requiring urgent technical escalation",
        "email_text": (
            "ALERT: Production database replica lag has exceeded 120 seconds "
            "on db-replica-03. Read queries are returning stale data. Multiple "
            "customers have reported seeing outdated order statuses. The primary "
            "node CPU is at 98% and autoscaling has not triggered. Oncall has "
            "been paged but has not acknowledged. This is impacting checkout flow."
        ),
        "sender_type": "system",
        "expected_category": "technical",
        "expected_priority": "high",
        "response_keywords": ["immediately", "escalate", "database", "investigate", "oncall", "production"],
    },
    "task_5": {
        "id": "task_5",
        "difficulty": "hard",
        "description": "Genuinely ambiguous email — looks like billing but root cause is technical",
        "email_text": (
            "Hi support, something weird is going on with my account. Last week I "
            "upgraded from the basic plan to pro using the in-app upgrade button, "
            "and the app showed a confirmation. My credit card was charged the pro "
            "amount the same day. But when I log in, the app still shows me as a "
            "basic user, my pro features are greyed out, and the billing page lists "
            "my plan as 'basic'. I tried logging out and back in, and even reinstalling "
            "the app on a different device — same thing. I am not sure if I should be "
            "asking for a refund or if there is a bug in your upgrade flow that did "
            "not actually flip my account over. Please advise."
        ),
        "sender_type": "customer",
        "expected_category": "technical",
        "expected_priority": "medium",
        "response_keywords": ["investigate", "account", "upgrade", "team", "check"],
    },
}


def get_task(task_id: str) -> dict:
    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id: {task_id}. Valid ids: {list(TASKS.keys())}")
    return TASKS[task_id]


def list_tasks() -> list:
    return [
        {
            "id": t["id"],
            "difficulty": t["difficulty"],
            "description": t["description"],
        }
        for t in TASKS.values()
    ]
