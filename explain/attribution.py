"""
Task 2A — Decision Attribution.

Given an assignment (w,d,s) that is =1 in the optimal solution, return:
  - forced_by_constraints: hard constraints that make removing it infeasible (IIS)
  - preferred_by_objective_terms: objective terms improved by it
  - dual_values / reduced_costs: from the LP relaxation
  - objective_delta_if_removed
"""
from ortools.sat.python import cp_model
import config as C
from solver.nurse_model import build_model
from solver.solve import solve
from solver.lp_duals import lp_duals


def _parse(target):           # "A,Wed,N" -> ("A","Wed","N")
    return tuple(target.split(","))


def attribute(target, base_sol=None):
    w, d, s = _parse(target)
    if base_sol is None:
        base_sol = solve()
    base_obj = base_sol["objective_value"]

    # ---- forcing test: forbid the assignment, find minimal infeasible core ----
    m, x, registry, _ = build_model()
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 42
    m.Add(x[w, d, s] == 0)
    lit_to_cid = {}
    for cid, meta in registry.items():
        m.AddAssumption(meta["lit"])
        lit_to_cid[meta["lit"].Index()] = cid
    status = solver.Solve(m)

    forced_by, objective_delta = [], None
    if status == cp_model.INFEASIBLE:
        for idx in solver.SufficientAssumptionsForInfeasibility():
            if idx in lit_to_cid:
                forced_by.append(lit_to_cid[idx])
    elif status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # feasible without it -> objective *prefers* it; measure how much
        objective_delta = solver.ObjectiveValue() - base_obj

    # ---- objective terms that touch this assignment ----
    preferred_by = []
    if (3 - C.pref(w, s)) < 1:                       # worker prefers this shift
        preferred_by.append(f"pref_score_{w}_{s}")
    preferred_by.append("fair_workload_balance")

    # ---- LP duals / reduced costs ----
    lp = lp_duals()
    cov_id = f"coverage_{d}_{s}"
    return {
        "target_assignment": f"x[{w},{d},{s}] = 1",
        "forced_by_constraints": forced_by,
        "preferred_by_objective_terms": preferred_by,
        "dual_values": {cov_id: lp["dual_values"].get(cov_id)},
        "reduced_costs": {f"x_{w}_{d}_{s}": lp["reduced_costs"].get(f"x_{w}_{d}_{s}")},
        "objective_delta_if_removed": objective_delta,
        "method": "IIS(assumptions) + objective-delta + LP-duals",
    }


if __name__ == "__main__":
    import json
    print(json.dumps(attribute("A,Mon,M"), indent=2, default=str))
