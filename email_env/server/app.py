from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from email_env.models import Action, StepResult, State, Observation
from email_env.server.environment import EmailTriageEnv
from email_env.tasks import list_tasks, TASKS
from email_env.grader import grade

app = FastAPI(title="Email Triage OpenEnv", version="1.0.0")
env = EmailTriageEnv()


@app.get("/")
def root():
    return {
        "name": "email-triage",
        "version": "1.0.0",
        "description": "Email Triage and Response OpenEnv Environment",
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/grader", "/baseline"],
    }


class ResetRequest(BaseModel):
    task_id: str = "task_1"


class GradeRequest(BaseModel):
    task_id: str
    category: str
    priority: str
    response: str


@app.post("/reset", response_model=Observation)
def reset(req: Optional[ResetRequest] = None):
    task_id = req.task_id if req else "task_1"
    try:
        obs = env.reset(task_id=task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return obs


@app.post("/step", response_model=StepResult)
def step(action: Action):
    try:
        result = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@app.get("/state", response_model=State)
def state():
    return env.state()


@app.get("/tasks")
def tasks():
    return list_tasks()


@app.post("/grader")
def grader(req: GradeRequest):
    if req.task_id not in TASKS:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {req.task_id}")
    action = {"category": req.category, "priority": req.priority, "response": req.response}
    score = grade(action, TASKS[req.task_id])
    return {"task_id": req.task_id, "score": score}


@app.get("/baseline")
def baseline_info():
    return {
        "description": "Run baseline.py locally with OPENAI_API_KEY set to evaluate the agent.",
        "command": "python -m email_env.baseline",
    }
