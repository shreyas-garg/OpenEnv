"""FastAPI server for the DriftEnv. Same OpenEnv contract as round 1."""

from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from drift_env.environment import DriftEnv
from drift_env.models import Action, Observation, State, StepResult

app = FastAPI(title="LeniencyBench", version="0.1.0")
env = DriftEnv()


_INFO = {
    "name": "LeniencyBench",
    "code_name": "policy-drift",
    "version": "0.1.0",
    "description": (
        "An OpenEnv benchmark that measures and trains out LLM leniency "
        "bias: the tendency to apply old/loose policies even after an "
        "admin message tightens the rule. 20-email customer-support "
        "episodes with 2 policy drifts per episode at fixed positions."
    ),
    "endpoints": ["/reset", "/step", "/state", "/docs"],
}


_HTML_PAGE = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LeniencyBench — live OpenEnv server</title>
<style>
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial,
                 sans-serif;
    max-width: 760px; margin: 2.5rem auto; padding: 0 1.25rem;
    background: #0d0f1a; color: #e6e8ee; line-height: 1.55;
  }
  h1 { font-size: 1.75rem; margin: 0 0 .25rem; color: #fff; }
  .tag { display: inline-block; padding: 2px 8px; border-radius: 4px;
         background: #1f8a3b; color: #fff; font-size: .8rem;
         font-weight: 600; vertical-align: middle; margin-left: .5rem; }
  .lede { color: #a8aec0; margin: .25rem 0 1.5rem; }
  .finding { background: #14182a; border-left: 3px solid #f29e2e;
             padding: .85rem 1rem; border-radius: 4px; margin-bottom: 1.5rem; }
  .finding strong { color: #fff; }
  h2 { font-size: 1.05rem; color: #fff; margin: 1.75rem 0 .5rem;
       border-bottom: 1px solid #2a2f44; padding-bottom: .25rem; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: .6rem 1rem;
          margin: .25rem 0 1rem; }
  .grid div { font-family: SFMono-Regular, Menlo, Consolas, monospace;
              font-size: .85rem; }
  .grid code { color: #2a6df4; background: transparent; }
  a { color: #6aa9ff; }
  pre { background: #14182a; padding: .75rem 1rem; border-radius: 4px;
        overflow-x: auto; font-size: .82rem; color: #c8cdda; }
  ul { padding-left: 1.2rem; }
  ul li { margin: .25rem 0; }
  .footer { color: #6e7591; font-size: .82rem; margin-top: 2rem;
            border-top: 1px solid #2a2f44; padding-top: 1rem; }
</style>
</head><body>

<h1>LeniencyBench<span class="tag">live</span></h1>
<p class="lede">An OpenEnv-compliant environment that measures and trains out
the <em>leniency bias</em> in frontier LLMs.</p>

<div class="finding">
  <strong>The finding:</strong> Llama 3.1 8B scores 0% on rules that
  <em>tighten</em> vs 37.5% on rules that <em>loosen</em> — a 37-point
  asymmetric failure from a single mid-context admin message.
  One epoch of SFT on this env's auto-generated labels closes the tightening
  gap from 0% to <strong>91.3%</strong> on Qwen 2.5 3B.
</div>

<h2>Endpoints (this server is responding right now)</h2>
<div class="grid">
  <div><code>GET /</code></div>      <div>this page (HTML) or info JSON (Accept: application/json)</div>
  <div><code>POST /reset</code></div><div>start an episode, returns first Observation</div>
  <div><code>POST /step</code></div> <div>submit one Action, returns reward + next Observation</div>
  <div><code>GET /state</code></div> <div>current episode snapshot</div>
  <div><code>GET /docs</code></div>  <div>interactive Swagger UI — try the API in-browser</div>
</div>

<h2>Try it from a terminal</h2>
<pre>curl -s -X POST https://shreyas-garg-drift-env.hf.space/reset \\
  -H "Content-Type: application/json" -d '{"seed": 42}'</pre>

<h2>Verify the headline result without retraining</h2>
<pre>from huggingface_hub import hf_hub_download
import json
p = hf_hub_download(
    "shreyas-garg/leniencybench-qwen3b-outputs",
    "evals.json", repo_type="model")
print(json.load(open(p))["post_sft"]["drift_acc_by_direction"])
# {'tightening': 0.913, 'loosening': 0.714, ...}</pre>

<h2>Read more</h2>
<ul>
  <li><a href="https://github.com/shreyas-garg/OpenEnv">Source code (GitHub)</a></li>
  <li><a href="https://huggingface.co/shreyas-garg/leniencybench">Source code (HF mirror)</a></li>
  <li><a href="https://huggingface.co/shreyas-garg/leniencybench-qwen3b-outputs">Trained adapter + logs</a></li>
  <li><a href="/docs">Interactive API explorer</a></li>
  <li><a href="https://huggingface.co/spaces/shreyas-garg/drift-env/blob/main/blog.md">Mini blog (≤2 min read)</a></li>
  <li><a href="https://huggingface.co/spaces/shreyas-garg/drift-env/blob/main/README.md">Full README with architecture, reward design, related work</a></li>
</ul>

<p class="footer">Built for the Meta PyTorch × Hugging Face OpenEnv Hackathon, Round 2 · April 2026.</p>

</body></html>
"""


@app.get("/")
def root(request: Request):
    """Content-negotiated landing page.

    - Browsers (Accept: text/html) get a styled landing page.
    - API clients (Accept: application/json or curl default) get the
      JSON metadata expected by validators and OpenEnv tools.
    """
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return HTMLResponse(_HTML_PAGE)
    return JSONResponse(_INFO)


@app.get("/info")
def info():
    """Always-JSON variant of /, useful for scripted clients."""
    return _INFO


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
