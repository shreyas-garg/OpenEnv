"""FastAPI server for the DriftEnv. Same OpenEnv contract as round 1."""

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from drift_env.environment import DriftEnv
from drift_env.models import Action, Observation, State, StepResult

app = FastAPI(title="Policy Drift OpenEnv", version="0.1.0")
env = DriftEnv()


@app.get("/")
def root():
    return {
        "name": "policy-drift",
        "version": "0.1.0",
        "description": (
            "Multi-step support-triage environment where policy rules drift "
            "mid-episode via admin emails. Agent must infer current policy "
            "from admin messages and apply it to subsequent customer tickets."
        ),
        "endpoints": ["/reset", "/step", "/state"],
    }


class ResetRequest(BaseModel):
    seed: int = 0
    episode_id: str = "ep_0"


@app.post("/reset", response_model=Observation)
def reset(req: Optional[ResetRequest] = None):
    req = req or ResetRequest()
    return env.reset(seed=req.seed, episode_id=req.episode_id)


@app.post("/step", response_model=StepResult)
def step(action: Action):
    try:
        return env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=State)
def state():
    return env.state()
