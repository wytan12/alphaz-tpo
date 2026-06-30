# Task 1 — TPO Reproduction (Hosted-API Path)

## Approach

This is a **from-scratch reproduction** of the core TPO idea (Li et al., 2025,
arXiv 2501.12895) using plain API calls — no textgrad, no TPO library, no vLLM.
The goal is to demonstrate understanding of the TPO loop by implementing it directly:

```
generate N candidates
      ↓
score each with a reward model
      ↓
pick best (chosen) + worst (rejected)
      ↓
textual gradient: LLM explains why chosen beats rejected
      ↓
refine: regenerate conditioned on best answer + critique
      ↓
repeat for D iterations → report win-rate at step 0, 1, 2
```

No model weights are updated at any point, alignment happens purely through
the iterative text feedback loop at inference time.

## Infrastructure (this run)

The paper uses vLLM on an A100/H100 GPU. This reproduction substitutes a
**Groq hosted API** throughout — no GPU required.

| Role | Paper | This run |
|---|---|---|
| Policy model | Llama-3.1-8B-SFT (vLLM) | `llama-3.1-8b-instant` (Groq) |
| Reward model | FsfairX-LLaMA3-RM-v0.1 (dedicated RM) | `llama-3.3-70b-versatile` as LLM-as-judge (Groq) |
| Win-rate judge | GPT-4-turbo (AlpacaEval-2) | `llama-3.3-70b-versatile` (Groq) |
| Reference answer | GPT-4-turbo baseline | `text_davinci_003` (AlpacaEval v1 cache) |
| Infra | vLLM, GPU | Groq REST API |

## Run config

```
n_prompts    = 20   (subset of AlpacaEval)
n_candidates = 5    (N candidates per iteration)
d_iterations = 2    (D TPO iterations)
```

Paper uses: 100 prompts, N=20, D=5 on a 70B model.

## Results

| Step | Win-rate (this run) | Paper reference |
|---|---|---|
| Step 0 — baseline | 100.0% | ~16.8% |
| Step 1 — after 1 TPO iter | 100.0% | — |
| Step 2 — after 2 TPO iters | 100.0% | ~55.7% |

Progress: **13 / 20 prompts** completed (run was interrupted; results saved incrementally).

## Why win-rate is 100% — and what that means

The win-rate is saturated at 100% for two compounding reasons:

**1. AlpacaEval version mismatch**
The cached reference answers (`alpaca_eval_cache.json`) come from `alpaca_eval.json`
on HuggingFace, which is **AlpacaEval v1** — its reference model is `text_davinci_003`
(GPT-3, 2022). AlpacaEval-2 (what the paper uses) has `gpt4_turbo` as the reference.
A modern 8B instruction-tuned model beats `text_davinci_003` on virtually every prompt,
so win-rate is 100% before TPO even runs.


**What the RM scores do show**
The `trajectory_scores` in `results.json` show the reward model score does rise across
iterations for most prompts (e.g., 78 → 90 on the Larry Page prompt). This confirms
the TPO loop is functioning mechanically — generating candidates, scoring, contrasting,
and refining — even though the binary win-rate metric is saturated.

## Delta explanation vs the paper

| Factor | Effect on win-rate |
|---|---|
| Reference = `text_davinci_003` instead of `gpt4_turbo` | Ceiling hit at step 0 → flat 100% |
| Policy already instruction-tuned (no room to improve) | No step0→step2 rise visible |
| N=5 vs N=20 candidates | Fewer candidates → noisier best/worst selection |
| D=2 vs D=5 iterations | Fewer refinement steps |
| LLM-as-judge vs dedicated RM | Judge scores may be less calibrated |
| 20 prompts vs 100 | Smaller sample, higher variance |

To reproduce the paper's rising curve, the correct setup is:
- Use AlpacaEval-2 as the reference 
- Use a dedicated reward model (FsfairX-LLaMA3-RM-v0.1)

## How to run

Ensure `.env` is set with Groq credentials (see root README), then:

```bash
# Full evaluation (20 prompts, resumes from existing results.json)
python task1_tpo/evaluate.py --n 20

# Quick smoke test (3 prompts)
python task1_tpo/evaluate.py --n 3

# Raw TPO loop only (2 prompts)
python task1_tpo/run_tpo.py
```

Results are saved incrementally to `task1_tpo/results.json` after every prompt,
so a partial run can always be resumed.
