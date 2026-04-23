"""FastAPI server for the DriftEnv. Same OpenEnv contract as round 1."""

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from drift_env.environment import DriftEnv
from drift_env.models import Action, Observation, State, StepResult

app = FastAPI(title="LeniencyBench", version="0.1.0")
env = DriftEnv()


@app.get("/")
def root():
    return {
        "name": "LeniencyBench",
        "code_name": "policy-drift",
        "version": "0.1.0",
        "description": (
            "An OpenEnv benchmark that measures and trains out LLM leniency "
            "bias: the tendency to apply old/loose policies even after an "
            "admin message tightens the rule. 20-email customer-support "
            "episodes with 2 policy drifts per episode at fixed positions."
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
