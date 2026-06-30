# TPO + Optimization Feedback System 

A feedback-driven optimization refinement system: a solver produces a schedule, the user
complains in natural language, the system **explains / attributes / localizes** the complaint
into a structured critique, and a **TPO-style loop** refines the solution at inference time. 


## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you don't have it, then:

```bash
uv sync        # creates .venv and installs all dependencies
```

**Optional LLM** (else rule-based fallback runs everywhere) — create a `.env` file in the project root:

```env
LLM_API_KEY=sk-...
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
```

`config.py` reads these automatically via `python-dotenv`. Works with DeepSeek, OpenAI, Qwen, or any OpenAI-compatible endpoint.

> Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`.
> numpy/ortools must be ABI-matched (ortools ≥9.15 + numpy ≥2 + pandas ≥2.2).

## Run everything
```bash
python -m solver.solve            # Task 2: optimal schedule (JSON)
python -m explain.attribution     # Task 2A: decision attribution + LP duals
python -m explain.counterfactual  # Task 2B: what-if swap/add, local re-opt vs full re-solve
python -m explain.localization    # Task 2C: feedback localization + clarify gate
python -m eval.evaluate           # Task 2 metrics: localization acc, counterfactual consistency
python -m eval.ablation           # Task 3: A/B/C ablation, mean & std over 3 runs
uvicorn api.main:app --reload     # Task 4: REST pipeline (POST /solve, /feedback, ...)
```

## Map to tasks
| Task | Where |
|------|-------|
| 1 TPO repro | `task1_tpo/README.md` (hosted-API path) |
| 2A attribution | `explain/attribution.py` (IIS + LP duals via `solver/lp_duals.py`) |
| 2B counterfactual | `explain/counterfactual.py`  |
| 2C localization | `explain/localization.py` (rule + LLM + confidence gate) |
| 2 eval | `eval/evaluate.py`, `eval/test_cases.json` (10 cases) |
| 3 ablation | `eval/ablation.py` |
| 4 pipeline | `api/main.py` (FastAPI + JSONL logs in `logs/`) |
| 5 design | see **Task 5** section below |

## Evaluation test cases (`eval/test_cases.json`)

10 hand-built cases. Each has a complaint, the ground-truth target it should resolve to, and
(for counterfactuals) the requested change. They were chosen to exercise **every component**
and **every kind of target**, not to be easy.

| # | Complaint | Ground-truth target | What it tests |
|---|-----------|---------------------|---------------|
| 1 | "Worker A is too tired" | `max_shifts_A` | localization → workload constraint |
| 2 | "Why is Worker A on so many night shifts?" | `pref_score_A_N` | localization → preference objective |
| 3 | "This schedule is unfair to some workers" | `fair_workload_balance` | localization → fairness objective |
| 4 | "Worker C has way too many shifts this week" | `max_shifts_C` | localization → workload (different worker) |
| 5 | "Can Worker D replace Worker A on Wed night?" | swap A↔D | **counterfactual** consistency (swap) |
| 6 | "Worker B is overworked and exhausted" | `max_shifts_B` | localization → workload (synonyms) |
| 7 | "There aren't enough people on the morning shift" | `coverage_*` | localization → **hard** constraint |
| 8 | "Too many night shifts for Worker C" | `pref_score_C_N` | localization → preference (night priority) |
| 9 | "What if Worker E takes an extra Tue evening?" | add `E,Tue,E` | **counterfactual** (add, not swap) |
| 10 | "huh this looks weird" | (none) | **clarify gate** (low confidence) |

**Why this set:**
- **Covers all three components** — 7 localization, 2 counterfactual, 1 clarify.
- **Covers every target type** — workload constraint, preference objective, fairness objective, hard coverage constraint, swap, add — so localization isn't tested on only one kind of answer.
- **Spreads across workers** A–E, so the router isn't overfit to one worker.
- **Varies phrasing** — explicit ("too many shifts"), synonyms ("overworked", "exhausted"), question-form ("why is…"), and deliberately vague ("huh this looks weird").
- **Includes a negative case** (#10) so the confidence gate must fire and ask a clarifying question instead of guessing.

**Honest limitation:** these cases test localization, counterfactual, and the clarify gate well, but they do **not yet independently test attribution** — the `expected_attribution` field currently mirrors the localization target. Adding real attribution ground truth (and the precision/recall metric) is the main remaining Task 2 item.

## Task 5 — Production-Grade Self-Evolution at Enterprise Scale

TPO self-corrects at inference time with frozen parameters. A production system must turn those runtime corrections into lasting model improvement. 

### 1 Closing the loop: correction → evolution

Each `/feedback` run logs a full TPO trajectory: `(input_solution, complaint, structured_critique, refined_solution, feasibility, converged)`. Converged trajectories are distilled back into the model:

- **DPO data:** accepted (refined) solution is `chosen`, original is `rejected` → DPO preference pairs for the critique-generator / localizer LLM.
- **SFT data:** high-confidence, human-confirmed `(complaint → structured critique)` pairs become SFT examples so localization improves directly.
- **Cadence:** nightly ETL + quality filter → versioned dataset; weekly DPO refresh on the shared shard; promotion only through A/B gate (4). Never train online on raw signals.

### 2 Feedback quality filtering (incl. reward hacking)

Production feedback is noisy. Before any signal enters training:

- **Confidence gate** (already live): low-confidence complaints are clarified, not trained on.
- **Feasibility + objective guard:** only keep refinements the solver confirms are feasible *and* improve the targeted objective term.
- **Attribution & counterfactual as validators:** if the user's praise contradicts what actually forced/preferred an assignment (attribution), or if their complaint targets a constraint counterfactual shows can't be relaxed, flag and down-weight rather than train.
- **Consistency / dedup:** require the same critique direction across multiple sessions; drop one-off outliers.

> **Meta RLUF lesson (arXiv:2505.14946):** Meta's `P[Love]` reward model was gamed — the policy spammed repetitive closers — even as the live "Love" rate rose +28%. Single user-satisfaction proxies are gameable. We use a multi-objective signal (satisfaction + feasibility + objective quality) and cap the satisfaction weight.

### 3 Multi-tenant isolation with shared evolution

- **Per-tenant:** isolated constraint registries, logs, and private **LoRA adapters** fine-tuned only on that tenant's filtered data. No raw cross-tenant data movement.
- **Shared base:** only aggregated, anonymised, generalisable signals feed the shared model — opt-in, with DP aggregation so no single tenant's data is recoverable.
- **Data residency:** raw feedback logs, optimization inputs, and constraint registries stay within the tenant's approved region or private VPC. For regulated tenants, the policy model, localizer, and critique generator run in-region rather than calling an external API.
  > *Assessment note: this implementation uses a hosted API due to local GPU limits. In production this would be replaced by an in-region or tenant-approved deployment.*
- **Serving:** shared base + tenant LoRA adapter at inference; every tenant gets shared improvements plus their private specialization.

### 4 A/B testing and rollback

- **Deploy path:** shadow → canary (small % traffic) → full, behind a feature flag; previous version stays pinned for instant revert.
- **Metrics:** feasibility rate, user-acceptance rate, complaint-resolution rate (converged within 3 iters), AlpacaEval win-rate (LLM components), latency p95.
- **Rollback trigger:** feasibility rate or acceptance rate regresses beyond threshold (e.g. >2% absolute drop) on the canary → automatic rollback + alert + postmortem before re-attempting promotion.

---

## Verified status (smoke-tested)
- Solver → OPTIMAL, objective 39, 32 assignments
- Attribution emits forcing constraints + LP duals/reduced costs
- Localization: high-conf refine, low-conf → clarifying question
- Counterfactual local re-opt agrees 100% with full re-solve (~0.06 s)
- Ablation runs 3× per condition

## Known scaffold limitations (your next steps)
1. **Ablation is not yet discriminating** — without an LLM API key, all conditions
   (both "with" and "without" structured input) converge at 1.0. This happens because:
   when `target_id is None` (the degraded condition), no soft constraint is injected so
   the solver returns the same optimal schedule every iteration, and `_complaint_free`
   falls back to checking `sol["feasible"]` which is always True. The LLM judge path
   (`_llm_complaint_free`) is already wired in, it will correctly report "not resolved"
   in degraded conditions but only fires when an API key is set.
2. **Task 1 win rates are 100% due to a different AlpacaEval version** — the cache
   (`src/task1_tpo/alpaca_eval_cache.json`) was populated from `alpaca_eval.json`
   (AlpacaEval **v1**, reference = `text_davinci_003` / GPT-3). AlpacaEval-2 uses
   `gpt4_turbo` as the reference, which is what the paper benchmarks against. Fix:
   delete the cache and change the filename to
   `alpaca_eval_gpt4_turbo_fn.json` — win rates will drop from 100% and TPO's
   step0→step1→step2 rise will become visible.
3. **Counterfactual neighborhood** covers involved workers + the affected day ± 1 adjacent day, and agrees 100% with full resolve on the current 2 test cases. Validated only on simple single day swaps, not tested on multi day or multi worker changes that may require a wider neighborhood.
4. **Localization case 7 fails** — "There aren't enough people on the morning shift" does
   not match the `"not enough"` keyword rule, so it routes to `clarify` instead of
   `coverage_*`. Add `"aren't enough"` (or a regex) to the coverage branch in
   `explain/localization.py`.
