"""
Task 2B — Counterfactual Explanation via fix-and-reoptimize (local neighborhood).

Full re-solve on every query is not acceptable. We freeze all variables outside a
small neighborhood (the involved workers + the affected day and adjacent days),
apply the requested change, and re-optimize only the neighborhood. A full re-solve
is also provided (`full_resolve`) for the Task 2 consistency evaluation.
"""
import time
from ortools.sat.python import cp_model
import config as C
from solver.nurse_model import build_model


def _neighborhood(changes):
    workers = {w for (w, d, s) in changes}
    days = {d for (w, d, s) in changes}
    di = {C.DAYS.index(d) for d in days}
    adj = set()
    for i in di:
        for j in (i - 1, i, i + 1):
            if 0 <= j < len(C.DAYS):
                adj.add(C.DAYS[j])
    return {(w, d, s) for w in workers for d in adj for s in C.SHIFTS}


def _run(base_sol, changes, reopt):
    """reopt = the set of variables allowed to change (the neighborhood). Everything
    NOT in reopt (and not a requested change) is frozen to its base value. reopt=None
    means a full re-solve (nothing frozen)."""
    m, x, registry, _ = build_model()
    for cid, meta in registry.items():
        m.Add(meta["lit"] == 1)
    for key, v in base_sol["assignments"].items():       # warm start
        w, d, s = key.split(",")
        m.AddHint(x[w, d, s], 1)
    for (w, d, s), v in changes.items():
        m.Add(x[w, d, s] == v)
    if reopt is not None:
        base = {tuple(k.split(",")): 1 for k in base_sol["assignments"]}
        for (w, d, s) in x:
            if (w, d, s) in changes or (w, d, s) in reopt:
                continue                                  # the change + neighborhood stay free
            m.Add(x[w, d, s] == base.get((w, d, s), 0))   # everything else frozen to base
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10
    solver.parameters.random_seed = 42
    t0 = time.time()
    st = solver.Solve(m)
    feasible = st in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    new = {f"{w},{d},{s}": 1 for (w, d, s) in x if feasible and solver.Value(x[w, d, s]) == 1}
    return feasible, (solver.ObjectiveValue() if feasible else None), new, time.time() - t0


def counterfactual(base_sol, changes):
    """changes: {(w,d,s): 0/1}. Returns local result + knock-on change set."""
    nbhd = _neighborhood(changes) - set(changes)
    feasible, obj, new, latency = _run(base_sol, changes, reopt=nbhd)
    before = set(base_sol["assignments"])
    after = set(new)
    return {
        "feasible": feasible,
        "objective_value": obj,
        "objective_delta": (obj - base_sol["objective_value"]) if feasible else None,
        "knock_on_changes": sorted((before ^ after) - {f"{w},{d},{s}" for (w, d, s) in changes}),
        "latency_s": round(latency, 4),
        "method": "fix-and-reoptimize (local neighborhood)",
    }


def full_resolve(base_sol, changes):
    """Ground-truth comparison: re-solve everything from scratch (eval only)."""
    feasible, obj, new, latency = _run(base_sol, changes, reopt=None)
    before, after = set(base_sol["assignments"]), set(new)
    return {"feasible": feasible, "objective_value": obj,
            "knock_on_changes": sorted((before ^ after) -
                                       {f"{w},{d},{s}" for (w, d, s) in changes}),
            "latency_s": round(latency, 4)}


if __name__ == "__main__":
    import json
    from solver.solve import solve
    base = solve()
    # Demo: swap Worker A off Wed night, Worker D onto Wed night
    changes = {("A", "Wed", "N"): 0, ("D", "Wed", "N"): 1}
    local = counterfactual(base, changes)
    full  = full_resolve(base, changes)
    print("Local re-opt:", json.dumps(local, indent=2))
    print("Full re-solve:", json.dumps(full, indent=2))
    print(f"Agreement: feasible={local['feasible']==full['feasible']}, "
          f"obj_match={abs((local['objective_value'] or 0)-(full['objective_value'] or 0)) < 0.01}")
