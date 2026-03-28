TASKS = {
    "task_1": {
        "id": "task_1",
        "difficulty": "easy",
        "description": "Clear billing issue with obvious classification",
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
    "task_2": {
        "id": "task_2",
        "difficulty": "medium",
        "description": "Ambiguous email with mixed signals",
        "email_text": (
            "Hello, I logged into my account today and noticed my dashboard looks "
            "different than before. Also, the invoice I received last week seems "
            "higher than expected. Not sure if this is a system update or a billing "
            "error. Can someone look into this?"
        ),
        "sender_type": "customer",
        "expected_category": "technical",
        "expected_priority": "medium",
        "response_keywords": ["investigate", "account", "assist", "team", "check"],
    },
    "task_3": {
        "id": "task_3",
        "difficulty": "hard",
        "description": "Complex complaint with emotional tone and multiple issues",
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
        "difficulty": "easy",
        "description": "Simple general inquiry about business hours",
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
    "task_5": {
        "id": "task_5",
        "difficulty": "hard",
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
