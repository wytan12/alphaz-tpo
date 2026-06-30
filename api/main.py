"""
Task 4 — Minimal feedback collection pipeline (FastAPI + JSON-file storage).

Endpoints:
  POST /solve            -> compute & store the base solution
  POST /feedback         -> ingest complaint -> localize -> (clarify | refine) -> log
  POST /explain          -> decision attribution + NL explanation
  POST /counterfactual   -> what-if query (local re-optimize)
  GET  /logs             -> full feedback/refinement log

Run:  uvicorn api.main:app --reload
"""
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import config as C
from solver.solve import solve, solution_json
from explain.attribution import attribute
from explain.counterfactual import counterfactual
from explain.localization import localize
from llm.nl_explainer import explain_attribution, explain_counterfactual
from llm import client as llm_client
from tpo_refine.refine_loop import refine, _critique_to_soft

app = FastAPI(title="TPO Optimization Feedback Pipeline")
LOG = Path(__file__).resolve().parent.parent / "logs" / "feedback_log.jsonl"
LOG.parent.mkdir(exist_ok=True)
SOLUTIONS = {}        # solution_id -> base_sol (in-memory)


def _log(record):
    record["ts"] = datetime.now(timezone.utc).isoformat()
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


class FeedbackIn(BaseModel):
    session_id: str
    solution_id: str
    complaint: str

class ExplainIn(BaseModel):
    target: str                     # "A,Wed,N"

class CounterfactualIn(BaseModel):
    solution_id: str
    changes: dict                   # {"A,Wed,N": 0, "D,Wed,N": 1}


@app.post("/solve")
def post_solve():
    sol = solve()
    sid = f"sol_{uuid.uuid4().hex[:8]}"
    SOLUTIONS[sid] = sol
    out = solution_json(sol)
    out["solution_id"] = sid
    return out


@app.post("/feedback")
def post_feedback(fb: FeedbackIn):
    base = SOLUTIONS.get(fb.solution_id) or solve()
    loc = localize(fb.complaint)

    if loc["action"] == "clarify":
        rec = {"type": "clarify", **fb.dict(), "localization": loc["localization"],
               "question": loc["question"]}
        _log(rec)
        return rec

    critique = loc["localization"]
    refined, traj, converged = refine(critique, base_sol=base)
    rec = {"type": "refine", **fb.dict(), "critique": critique,
           "converged": converged, "trajectory": traj,
           "input_objective": base["objective_value"],
           "refined_objective": refined["objective_value"],
           "feasible": refined["feasible"]}
    _log(rec)
    rec["refined_solution"] = solution_json(refined)
    return rec


@app.post("/explain")
def post_explain(inp: ExplainIn):
    attr = attribute(inp.target)
    attr["explanation"] = explain_attribution(attr)
    return attr


@app.post("/counterfactual")
def post_counterfactual(inp: CounterfactualIn):
    base = SOLUTIONS.get(inp.solution_id) or solve()
    changes = {tuple(k.split(",")): int(v) for k, v in inp.changes.items()}
    cf = counterfactual(base, changes)
    cf["explanation"] = explain_counterfactual(cf)
    return cf


class TraceIn(BaseModel):
    complaint: str
    solution_id: str = ""


@app.post("/trace")
def post_trace(inp: TraceIn):
    """Run the full feedback pipeline and return every stage for step-by-step display."""
    base = SOLUTIONS.get(inp.solution_id) or solve()
    stages = []

    # stage 1 — baseline solve
    stages.append({"id": 1, "title": "Baseline solve (CP-SAT)", "status": "ok",
                   "detail": {"objective_value": base["objective_value"],
                              "num_assignments": len(base["assignments"]),
                              "feasible": base["feasible"]}})

    # stage 2 — feedback localization
    loc = localize(inp.complaint)
    l = loc["localization"]
    if loc["action"] == "clarify":
        stages.append({"id": 2, "title": "Feedback localization", "status": "warn",
                       "detail": {"action": "clarify (low confidence)",
                                  "confidence": l.get("confidence"),
                                  "question": loc["question"]}})
        return {"complaint": inp.complaint, "stopped": "clarify", "stages": stages}
    stages.append({"id": 2, "title": "Feedback localization", "status": "ok",
                   "detail": {"action": "refine", "target_id": l.get("target_id"),
                              "target_type": l.get("target_type"),
                              "modification_type": l.get("modification_type"),
                              "confidence": l.get("confidence")}})

    # stage 3 — structured critique -> soft constraint
    soft = _critique_to_soft(l)
    stages.append({"id": 3, "title": "Structured critique → soft constraint", "status": "ok",
                   "detail": {"critique": l.get("critique"),
                              "soft_constraint": soft}})

    # stage 4 — TPO refinement loop
    refined, traj, converged = refine(l, base_sol=base)
    stages.append({"id": 4, "title": "TPO refinement loop", "status": "ok",
                   "detail": {"iterations": traj, "converged": converged}})

    # stage 5 — final solution + feasibility
    stages.append({"id": 5, "title": "Final solution + feasibility check",
                   "status": "ok" if converged else "warn",
                   "detail": {"feasible": refined["feasible"],
                              "objective_before": base["objective_value"],
                              "objective_after": refined["objective_value"],
                              "converged": converged,
                              "assignments": list(refined["assignments"].keys())}})

    _log({"type": "trace", "complaint": inp.complaint, "converged": converged})
    return {"complaint": inp.complaint, "stopped": None, "stages": stages,
            "baseline_assignments": list(base["assignments"].keys()),
            "final_assignments": list(refined["assignments"].keys())}


@app.get("/logs")
def get_logs():
    if not LOG.exists():
        return []
    return [json.loads(l) for l in LOG.read_text(encoding="utf-8").splitlines()]


@app.get("/status")
def status():
    """Whether explanations come from the LLM or the rule/template fallback."""
    if llm_client.available():
        return {"mode": "llm", "model": C.LLM_MODEL, "base_url": C.LLM_BASE_URL}
    return {"mode": "fallback", "model": None, "base_url": None}


_UI_HTML_PATH = Path(__file__).resolve().parent / "static" / "html" / "index.html"


@app.get("/", response_class=HTMLResponse)
def ui():
    return _UI_HTML_PATH.read_text(encoding="utf-8")
