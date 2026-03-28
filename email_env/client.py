"""
Python client for the Email Triage OpenEnv server.
"""

import requests


class EmailTriageClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    def reset(self, task_id: str = "task_1") -> dict:
        r = requests.post(f"{self.base_url}/reset", json={"task_id": task_id})
        r.raise_for_status()
        return r.json()

    def step(self, category: str, priority: str, response: str) -> dict:
        payload = {"category": category, "priority": priority, "response": response}
        r = requests.post(f"{self.base_url}/step", json=payload)
        r.raise_for_status()
        return r.json()

    def state(self) -> dict:
        r = requests.get(f"{self.base_url}/state")
        r.raise_for_status()
        return r.json()

    def tasks(self) -> list:
        r = requests.get(f"{self.base_url}/tasks")
        r.raise_for_status()
        return r.json()

    def grade(self, task_id: str, category: str, priority: str, response: str) -> float:
        payload = {
            "task_id": task_id,
            "category": category,
            "priority": priority,
            "response": response,
        }
        r = requests.post(f"{self.base_url}/grader", json=payload)
        r.raise_for_status()
        return r.json()["score"]
