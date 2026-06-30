"""
Task 2C — Feedback Localization.

Vague complaint -> structured critique {target_type, target_id, modification_type,
critique, confidence}. Hybrid: rule layer first, LLM refines if available.
Low confidence -> clarifying question instead of triggering TPO.
"""
import re
import config as C
from solver.nurse_model import build_model
from llm import client

CONF_THRESHOLD = 0.6
_, _, REGISTRY, _ = build_model()
KNOWN_IDS = list(REGISTRY.keys())


def _rule_localize(complaint):
    t = complaint.lower()
    worker = None
    m = re.search(r"worker\s+([a-h])\b", t)
    if m:
        worker = m.group(1).upper()

    # night-specific complaint takes priority over generic overwork
    if worker and "night" in t:
        return {"target_type": "objective_term", "target_id": f"pref_score_{worker}_N",
                "modification_type": "adjust_objective_weight",
                "critique": f"Lower the number of night shifts assigned to Worker {worker}.",
                "confidence": 0.8}
    if worker and any(k in t for k in ["tired", "too many", "overwork", "exhaust", "burn"]):
        return {"target_type": "worker_constraint", "target_id": f"max_shifts_{worker}",
                "modification_type": "add_soft_constraint",
                "critique": f"Reduce Worker {worker}'s total shifts / avoid demanding back-to-back shifts.",
                "confidence": 0.85}
    if any(k in t for k in ["unfair", "fairness", "balance", "uneven"]):
        return {"target_type": "objective_term", "target_id": "fair_workload_balance",
                "modification_type": "adjust_objective_weight",
                "critique": "Increase the fairness weight to balance workload across workers.",
                "confidence": 0.75}
    if "coverage" in t or "understaff" in t or "not enough" in t:
        return {"target_type": "hard_constraint", "target_id": "coverage_*",
                "modification_type": "modify_input_data",
                "critique": "Adjust coverage requirements for the affected shift(s).",
                "confidence": 0.55}
    return {"target_type": "unknown", "target_id": None,
            "modification_type": None, "critique": complaint, "confidence": 0.3}


def localize(complaint, conf_threshold=CONF_THRESHOLD):
    result = _rule_localize(complaint)

    # LLM refinement (constrained to known IDs) if available
    if client.available():
        prompt = (f"Complaint: {complaint!r}\n"
                  f"Valid target_ids (choose one or a wildcard like coverage_*): {KNOWN_IDS}\n"
                  "Return JSON: {target_type, target_id, modification_type "
                  "(one of add_soft_constraint|adjust_objective_weight|modify_input_data), "
                  "critique, confidence (0-1)}.")
        llm = client.chat_json(prompt)
        if llm and llm.get("target_id"):
            result = llm

    # confidence gate
    if result.get("confidence", 0) < conf_threshold:
        return {"action": "clarify",
                "question": _clarify(complaint, result),
                "localization": result}
    return {"action": "refine", "localization": result}


def _clarify(complaint, result):
    return (f"I'm not fully sure what '{complaint}' refers to. Do you mean a specific "
            f"worker's total workload, their night shifts, or overall fairness? "
            f"Please name the worker/shift.")


if __name__ == "__main__":
    import json
    for c in ["Worker A is too tired", "This schedule is unfair", "huh?"]:
        print(c, "->", json.dumps(localize(c)))
