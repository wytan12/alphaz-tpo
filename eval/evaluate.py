"""
Task 2 evaluation: localization accuracy (by confidence), attribution P/R,
counterfactual consistency vs full re-solve.

Run:  python -m eval.evaluate
"""
import json
from pathlib import Path
from solver.solve import solve
from explain.localization import localize, CONF_THRESHOLD
from explain.counterfactual import counterfactual, full_resolve

CASES = json.loads((Path(__file__).parent / "test_cases.json").read_text())


def eval_localization():
    high_ok = high_n = low_ok = low_n = 0
    for c in CASES:
        if c["gt_type"] not in ("localization", "clarify"):
            continue
        res = localize(c["complaint"])
        loc = res["localization"]
        conf = loc.get("confidence", 0)
        pred = loc.get("target_id")
        gt = c["gt_target"]
        match = (pred == gt) or (gt and gt.endswith("*") and pred and
                                 pred.startswith(gt[:-1])) or (gt is None and res["action"] == "clarify")
        if conf >= CONF_THRESHOLD:
            high_n += 1; high_ok += int(match)
        else:
            low_n += 1; low_ok += int(match)
    return {"high_conf_acc": round(high_ok / max(high_n, 1), 3), "high_n": high_n,
            "low_conf_acc": round(low_ok / max(low_n, 1), 3), "low_n": low_n}


def eval_counterfactual():
    base = solve()
    agree = lat = n = 0
    for c in CASES:
        if c["gt_type"] != "counterfactual":
            continue
        n += 1
        changes = {tuple(k.split(",")): int(v) for k, v in c["changes"].items()}
        local = counterfactual(base, changes)
        full = full_resolve(base, changes)
        feas_match = local["feasible"] == full["feasible"]
        obj_match = (not local["feasible"]) or (full["objective_value"] and
                    abs((local["objective_value"] or 0) - full["objective_value"])
                    <= 0.01 * abs(full["objective_value"]))
        agree += int(feas_match and obj_match)
        lat += local["latency_s"]
    return {"n": n, "agreement_rate": round(agree / max(n, 1), 3),
            "mean_latency_s": round(lat / max(n, 1), 4)}


if __name__ == "__main__":
    print("Localization:", json.dumps(eval_localization()))
    print("Counterfactual:", json.dumps(eval_counterfactual()))
