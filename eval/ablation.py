"""
Task 3 — Ablation: which explanation component drives convergence?

Conditions (from the brief), each run >=3 times; report mean & std of convergence
rate (feasible + complaint-free within 3 TPO iterations):
  A: structured Decision Attribution vs raw complaint text
  B: Counterfactual Probing between iterations vs directly accepting output
  C: structured Localization vs raw NL complaint passed to TPO

Run:  python -m eval.ablation
"""
import json
import statistics
from pathlib import Path
from solver.solve import solve
from explain.localization import localize, _rule_localize
from tpo_refine.refine_loop import refine

CASES = [c for c in json.loads((Path(__file__).parent / "test_cases.json").read_text())
         if c["gt_type"] == "localization"]


def _run_condition(use_localization=True, use_attribution=True, probe=False, seed=0):
    base = solve()
    converged = 0
    for c in CASES:
        if use_localization:
            res = localize(c["complaint"])
            crit = res["localization"]
        else:
            # raw complaint: degraded critique with no structured target
            crit = {"target_id": None, "critique": c["complaint"], "confidence": 0.5}
        if not use_attribution:
            crit = {**crit, "target_id": None}      # strip the implicated IDs
        _, _, ok = refine(crit, base_sol=base, probe=probe)
        converged += int(ok)
    return converged / len(CASES)


def ablate(label, runs=3, **kw):
    rates = [_run_condition(seed=i, **kw) for i in range(runs)]
    return {"condition": label, "mean": round(statistics.mean(rates), 3),
            "std": round(statistics.pstdev(rates), 3), "runs": rates}


if __name__ == "__main__":
    results = [
        ablate("A: with attribution",    use_attribution=True),
        ablate("A: without attribution", use_attribution=False),
        ablate("B: with cf probing",     probe=True),
        ablate("B: without cf probing",  probe=False),
        ablate("C: with localization",   use_localization=True),
        ablate("C: raw complaint",       use_localization=False),
    ]
    for r in results:
        print(json.dumps(r))
