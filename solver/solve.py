"""Solve the instance to optimality and emit a solution dict / JSON."""
import json
from ortools.sat.python import cp_model
import config as C
from solver.nurse_model import build_model


def solve(extra_soft=None, time_limit=10.0, assume_all=False):
    m, x, registry, penalties = build_model(extra_soft)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.random_seed = 42

    if assume_all:
        # required for IIS: enforce every hard constraint and make it assumable
        solver.parameters.num_search_workers = 1
        for cid, meta in registry.items():
            m.AddAssumption(meta["lit"])
    else:
        # normal solve: all hard constraints active
        for cid, meta in registry.items():
            m.Add(meta["lit"] == 1)

    status = solver.Solve(m)
    feasible = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assignments = {}
    if feasible:
        for (w, d, s), var in x.items():
            if solver.Value(var) == 1:
                assignments[f"{w},{d},{s}"] = 1
    return {
        "feasible": feasible,
        "status": solver.StatusName(status),
        "objective_value": solver.ObjectiveValue() if feasible else None,
        "assignments": assignments,            # "W,D,S" -> 1
        "_solver": solver, "_model": m, "_x": x, "_registry": registry,
    }


def solution_json(sol):
    """Strip private solver objects for clean JSON output."""
    return {k: v for k, v in sol.items() if not k.startswith("_")}


if __name__ == "__main__":
    sol = solve()
    print(json.dumps(solution_json(sol), indent=2))
    print(f"\n{len(sol['assignments'])} assignments, "
          f"objective={sol['objective_value']}, status={sol['status']}")
