"""
LP relaxation (via PuLP) to extract dual values + reduced costs.

CP-SAT does not expose duals; the assessment's attribution JSON asks for
"dual values or reduced costs". We solve the LP relaxation of the same model
purely to populate those fields, mapping duals back to our constraint IDs.
"""
import pulp
import config as C


def lp_duals():
    prob = pulp.LpProblem("nurse_relax", pulp.LpMinimize)
    x = {(w, d, s): pulp.LpVariable(f"x_{w}_{d}_{s}", lowBound=0, upBound=1)
         for w in C.WORKERS for d in C.DAYS for s in C.SHIFTS}

    con = {}
    # coverage (the binding constraints we care about for attribution)
    for d in C.DAYS:
        for s in C.SHIFTS:
            cid = f"coverage_{d}_{s}"
            c = pulp.lpSum(x[w, d, s] for w in C.WORKERS) >= C.COVERAGE[s]
            prob += c, cid
            con[cid] = c
    # one shift per worker per day
    for w in C.WORKERS:
        for d in C.DAYS:
            cid = f"one_per_day_{w}_{d}"
            c = pulp.lpSum(x[w, d, s] for s in C.SHIFTS) <= 1
            prob += c, cid
            con[cid] = c
    # workload bounds
    for w in C.WORKERS:
        load = pulp.lpSum(x[w, d, s] for d in C.DAYS for s in C.SHIFTS)
        prob += (load <= C.MAX_SHIFTS_PER_WORKER), f"max_shifts_{w}"
        prob += (load >= C.MIN_SHIFTS_PER_WORKER), f"min_shifts_{w}"

    # objective: preference dissatisfaction (linear part of the soft objective)
    prob += pulp.lpSum((3 - C.pref(w, s)) * C.W_PREF * x[w, d, s]
                       for w in C.WORKERS for d in C.DAYS for s in C.SHIFTS)

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    duals = {name: c.pi for name, c in prob.constraints.items() if c.pi is not None}
    reduced = {v.name: v.dj for v in prob.variables() if v.dj is not None}
    return {"dual_values": duals, "reduced_costs": reduced,
            "lp_objective": pulp.value(prob.objective)}


if __name__ == "__main__":
    import json
    print(json.dumps(lp_duals(), indent=2))
