"""Render structured attribution/counterfactual JSON into natural language."""
from llm import client


def explain_attribution(attr):
    if client.available():
        out = client.chat(
            f"Explain this optimization attribution to a non-technical scheduler in 2-3 "
            f"sentences. Be faithful to the data; do not invent constraints.\n{attr}")
        if out:
            return out
    forced = ", ".join(attr["forced_by_constraints"]) or "no hard constraint (objective-driven)"
    return (f"{attr['target_assignment']} is required by: {forced}. "
            f"It is also preferred by objective terms "
            f"({', '.join(attr['preferred_by_objective_terms'])}).")


def explain_counterfactual(cf):
    if client.available():
        out = client.chat(f"Summarize this what-if result in 2 sentences:\n{cf}")
        if out:
            return out
    if not cf["feasible"]:
        return "That change makes the schedule infeasible — it cannot be satisfied."
    return (f"Feasible. Objective changes by {cf['objective_delta']}; "
            f"{len(cf['knock_on_changes'])} other assignments must change.")
