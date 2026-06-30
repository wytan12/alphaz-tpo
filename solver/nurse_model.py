"""
Build the CP-SAT nurse-scheduling model + a constraint REGISTRY.

The registry is the backbone of Task 2: every hard constraint is created with an
enforcement literal, so we can (a) drop a constraint to test attribution and
(b) extract a minimal infeasible core (IIS) via assumptions.
"""
from ortools.sat.python import cp_model
import config as C


def build_model(extra_soft=None):
    """
    extra_soft: optional list of dicts describing TPO-injected soft constraints, e.g.
        {"id": "soft_reduce_A", "vars": [("A",d,s) ...], "limit": 2, "weight": 50}
    Returns (model, x, registry, penalties) where:
        x[(w,d,s)]     -> BoolVar
        registry[id]   -> {"lit": literal, "desc": str, "vars": [(w,d,s)...]}
        penalties      -> list of (IntVar/LinearExpr, weight) added to objective
    """
    m = cp_model.CpModel()
    x = {(w, d, s): m.NewBoolVar(f"x_{w}_{d}_{s}")
         for w in C.WORKERS for d in C.DAYS for s in C.SHIFTS}
    registry = {}

    def hard(cid, desc, vars_):
        lit = m.NewBoolVar(f"lit_{cid}")
        registry[cid] = {"lit": lit, "desc": desc, "vars": vars_}
        return lit

    # 1) coverage: exactly/at least COVERAGE[s] workers per (day, shift)
    for d in C.DAYS:
        for s in C.SHIFTS:
            cid = f"coverage_{d}_{s}"
            lit = hard(cid, f"Shift {s} on {d} needs >= {C.COVERAGE[s]} workers",
                       [(w, d, s) for w in C.WORKERS])
            m.Add(sum(x[w, d, s] for w in C.WORKERS) >= C.COVERAGE[s]).OnlyEnforceIf(lit)

    # 2) at most one shift per worker per day
    for w in C.WORKERS:
        for d in C.DAYS:
            cid = f"one_per_day_{w}_{d}"
            lit = hard(cid, f"{w} works <=1 shift on {d}",
                       [(w, d, s) for s in C.SHIFTS])
            m.Add(sum(x[w, d, s] for s in C.SHIFTS) <= 1).OnlyEnforceIf(lit)

    # 3) no Night -> next-day Morning (rest rule)
    for w in C.WORKERS:
        for i in range(len(C.DAYS) - 1):
            d0, d1 = C.DAYS[i], C.DAYS[i + 1]
            cid = f"rest_{w}_{d0}"
            lit = hard(cid, f"{w}: no Night on {d0} then Morning on {d1}",
                       [(w, d0, "N"), (w, d1, "M")])
            m.Add(x[w, d0, "N"] + x[w, d1, "M"] <= 1).OnlyEnforceIf(lit)

    # 4) workload bounds
    for w in C.WORKERS:
        load = sum(x[w, d, s] for d in C.DAYS for s in C.SHIFTS)
        cid = f"max_shifts_{w}"
        lit = hard(cid, f"{w} works <= {C.MAX_SHIFTS_PER_WORKER} shifts/week",
                   [(w, d, s) for d in C.DAYS for s in C.SHIFTS])
        m.Add(load <= C.MAX_SHIFTS_PER_WORKER).OnlyEnforceIf(lit)
        cid2 = f"min_shifts_{w}"
        lit2 = hard(cid2, f"{w} works >= {C.MIN_SHIFTS_PER_WORKER} shifts/week",
                    [(w, d, s) for d in C.DAYS for s in C.SHIFTS])
        m.Add(load >= C.MIN_SHIFTS_PER_WORKER).OnlyEnforceIf(lit2)

    # ---- objective (soft) ----
    penalties = []

    # fairness: minimise max-min workload spread
    loads = [sum(x[w, d, s] for d in C.DAYS for s in C.SHIFTS) for w in C.WORKERS]
    hi = m.NewIntVar(0, len(C.DAYS), "load_max")
    lo = m.NewIntVar(0, len(C.DAYS), "load_min")
    m.AddMaxEquality(hi, loads)
    m.AddMinEquality(lo, loads)
    spread = m.NewIntVar(0, len(C.DAYS), "spread")
    m.Add(spread == hi - lo)
    penalties.append((spread, C.W_FAIRNESS))

    # preference dissatisfaction (3 - pref) per assigned shift
    pref_pen = sum((3 - C.pref(w, s)) * x[w, d, s]
                   for w in C.WORKERS for d in C.DAYS for s in C.SHIFTS)
    penalties.append((pref_pen, C.W_PREF))

    # tiredness: consecutive night shifts
    for w in C.WORKERS:
        for i in range(len(C.DAYS) - 1):
            cn = m.NewBoolVar(f"consecN_{w}_{i}")
            m.AddBoolAnd([x[w, C.DAYS[i], "N"], x[w, C.DAYS[i + 1], "N"]]).OnlyEnforceIf(cn)
            m.AddBoolOr([x[w, C.DAYS[i], "N"].Not(),
                         x[w, C.DAYS[i + 1], "N"].Not()]).OnlyEnforceIf(cn.Not())
            penalties.append((cn, C.W_NIGHTS))

    # TPO-injected soft constraints (from refinement loop)
    for sc in (extra_soft or []):
        viol = m.NewIntVar(0, len(C.DAYS) * len(C.SHIFTS), f"viol_{sc['id']}")
        m.Add(viol >= sum(x[w, d, s] for (w, d, s) in sc["vars"]) - sc["limit"])
        m.Add(viol >= 0)
        penalties.append((viol, sc["weight"]))

    m.Minimize(sum(coef * var for var, coef in penalties))
    return m, x, registry, penalties
