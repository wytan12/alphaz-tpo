# TPO + Optimization Feedback System 

A feedback-driven optimization refinement system: a solver produces a schedule, the user
complains in natural language, the system **explains / attributes / localizes** the complaint
into a structured critique, and a **TPO-style loop** refines the solution at inference time. 


## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you don't have it, then:

```bash
uv sync        # creates .venv and installs all dependencies
```

**Optional LLM**  without it, rule-based fallback runs everywhere. To enable it, create a `.env` in the project root:

```env
LLM_API_KEY=sk-...
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=llama-3.1-8b-instant
```

`config.py` reads these automatically via `python-dotenv`. Works with DeepSeek, OpenAI, Qwen, or any OpenAI-compatible endpoint.

## Running

```bash
python -m solver.solve            # Task 2: solve to optimality, print schedule
python -m explain.attribution     # Task 2A: which constraints forced this assignment
python -m explain.counterfactual  # Task 2B: what-if swap, local re-opt vs full re-solve
python -m explain.localization    # Task 2C: map a complaint to a target_id
python -m eval.evaluate           # Task 2 metrics
python -m eval.ablation           # Task 3: ablation across 3 conditions
uvicorn api.main:app --reload     # Task 4: REST API
```

## Task map

| Task | File |
|------|-------|
| 1 TPO repro | `task1_tpo/` — see its own README |
| 2A attribution | `explain/attribution.py` |
| 2B counterfactual | `explain/counterfactual.py` |
| 2C localization | `explain/localization.py` |
| 2 eval | `eval/evaluate.py`, `eval/test_cases.json` |
| 3 ablation | `eval/ablation.py` |
| 4 pipeline | `api/main.py` |
| 5 design | see **Task 5** section below |

## Test cases (`eval/test_cases.json`)

10 cases covering all three components. Each has a complaint, the ground-truth target, and (for counterfactuals) the requested change.

| # | Complaint | Ground-truth target | What it tests |
|---|-----------|---------------------|---------------|
| 1 | "Worker A is too tired" | `max_shifts_A` | localization → workload constraint |
| 2 | "Why is Worker A on so many night shifts?" | `pref_score_A_N` | localization → preference objective |
| 3 | "This schedule is unfair to some workers" | `fair_workload_balance` | localization → fairness objective |
| 4 | "Worker C has way too many shifts this week" | `max_shifts_C` | localization → workload (different worker) |
| 5 | "Can Worker D replace Worker A on Wed night?" | swap A↔D | counterfactual (swap) |
| 6 | "Worker B is overworked and exhausted" | `max_shifts_B` | localization → workload (synonyms) |
| 7 | "There aren't enough people on the morning shift" | `coverage_*` | localization → hard constraint |
| 8 | "Too many night shifts for Worker C" | `pref_score_C_N` | localization → preference (night) |
| 9 | "What if Worker E takes an extra Tue evening?" | add `E,Tue,E` | counterfactual (add) |
| 10 | "huh this looks weird" | (none) | clarify gate |

Cases 1–4, 6–8, 10 test localization; 5 and 9 test counterfactual consistency; 10 tests that the confidence gate fires instead of guessing. Attribution doesn't have its own ground-truth cases yet — `expected_attribution` currently mirrors the localization target.

## Task 5 — Production Self-Evolution

The TPO loop runs at inference time with frozen weights. To make improvements persist, I'd close the loop like this:

**Closing the loop.** Each `/feedback` call logs the full trajectory: complaint, structured critique, before/after solution, feasibility, converged. Converged trajectories become training data, accepted refinements as DPO preference pairs (refined = chosen, original = rejected), high confidence localizations as SFT examples. I'd run nightly ETL into a versioned dataset and do weekly DPO refreshes, never training directly on raw runtime signals.

**Filtering feedback.** Not every complaint is worth training on. I'd keep only refinements that are feasible and improve the targeted objective term. The attribution output is useful here too. If the user's complaint contradicts what actually caused the assignment, it's probably a misunderstanding, not a valid training signal. Low confidence complaints already go to clarify rather than refine, so they never enter the pipeline. This is partly inspired by Meta's RLUF work (arXiv:2505.14946) where optimizing a single `P[Love]` proxy led to reward hacking, where the policy started spamming "Bye!" while the live metric kept rising. A multi-objective signal (feasibility + objective quality + user acceptance) is harder to game.

**Multi-tenant isolation.** Each tenant gets their own constraint registry, feedback logs and a private LoRA adapter trained only on their data. Shared improvements only come from aggregated, anonymised signals (no raw cross-tenant data). For regulated industries (healthcare, finance), raw logs and inference stay inside the tenant's region or VPC. I'm using a hosted API here because there's no GPU available but in production that would need to be a within region deployment.

**Deployment.** Shadow traffic first, then canary, then full rollout behind a feature flag with the previous version pinned. I'd track feasibility rate, user acceptance rate and complaint resolution rate (converged within 3 iterations). If any of those drop more than 2% on the canary and automatic rollback before trying again.

---

## Current status

- Solver → OPTIMAL, objective 39, 32 assignments
- Attribution: forcing constraints + LP duals working
- Localization: high confidence cases refine, low confidence cases ask for clarification
- Counterfactual: local re-opt matches full re-solve on all test cases (~0.02s)
- Ablation runs 3× per condition (discriminating only when LLM key is set — see limitations)

## Known limitations

**Ablation results are flat without an LLM key.** In the degraded conditions (`target_id = None`), no soft constraint is injected so the solver returns the same schedule every time. `_complaint_free` then falls back to checking `sol["feasible"]`, which is always True, so every condition converges at 1.0. The LLM judge path (`_llm_complaint_free`) is wired in and will correctly return "not resolved" — it just needs an API key set.

**Task 1 win rates are 100% due to a different AlpacaEval version.** The cache was built from `alpaca_eval.json` (AlpacaEval v1, reference = `text_davinci_003`). The paper benchmarks against `gpt4_turbo` (AlpacaEval-2). Swapping to `alpaca_eval_gpt4_turbo_fn.json` and clearing the cache would give more meaningful win rates. The RM trajectory scores do rise across iterations even now, which confirms the loop is working mechanically.

**Localization misses case 7.** "There aren't enough people on the morning shift" doesn't hit the `"not enough"` keyword so it routes to clarify instead of `coverage_*`. A quick fix is adding `"aren't enough"` to the rule or switching to a regex.

**Counterfactual neighborhood.** The neighborhood is workers + the affected day ± 1 day. It agrees with full re-solve on the two current test cases but hasn't been tested on changes that affect multiple workers across multiple days.
