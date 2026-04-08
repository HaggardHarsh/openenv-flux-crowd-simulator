import os
import sys

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure we can import the module correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crowd_env.environment import CrowdManagementEnv
from crowd_env.models import Action

app = FastAPI(title="Crowd Management OpenEnv", version="1.0.0")

# We manage a single global env instance for standard HuggingFace Space deployment
env = CrowdManagementEnv()

class ResetRequest(BaseModel):
    seed: int | None = None
    task: str = "easy"

class StepRequest(BaseModel):
    action_type: str = "no_op"
    source_zone: str = ""
    target_zone: str = ""
    gate_index: int = 0
    gate_open: bool = True

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset_env(req: ResetRequest):
    obs = env.reset(seed=req.seed, options={"task": req.task})
    return {"observation": obs.model_dump()}

@app.post("/step")
def step_env(req: StepRequest):
    try:
        action = Action(
            action_type=req.action_type,
            source_zone=req.source_zone,
            target_zone=req.target_zone,
            gate_index=req.gate_index,
            gate_open=req.gate_open
        )
        result = env.step(action)
        return {
            "observation": result.observation.model_dump(),
            "reward": result.reward,
            "terminated": result.terminated,
            "truncated": result.truncated,
            "info": result.info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
def state_env():
    try:
        s = env.state()
        return s.model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/grade")
def grade_env():
    try:
        g = env.grade()
        return {"score": g.score, "letter_grade": g.letter_grade, "summary": g.summary}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Mount static visualization files at root
viz_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualization")
app.mount("/", StaticFiles(directory=viz_dir, html=True), name="static")
