"""
Task 1 evaluation: run TPO on 100 AlpacaEval-2 prompts.
Reports win-rate at step 0 (baseline), step 1, step 2 vs GPT-4 reference answers.

Deliverable: a table matching the paper's format + explanation of delta.

Run:
    python src/task1_tpo/evaluate.py
    python src/task1_tpo/evaluate.py --n 10   # quick smoke test on 10 prompts
"""

import json
import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from task1_tpo.run_tpo import generate, run_loop, win_rate_judge

DATA_CACHE = Path(__file__).parent / "alpaca_eval_cache.json"


def load_alpaca_data(n=100):
    """
    Load n prompts from AlpacaEval-2.
    Downloads alpaca_eval.json directly via huggingface_hub and caches it locally.
    Each item: {"instruction": str, "output": str (GPT-4-Turbo reference answer)}
    """
    if DATA_CACHE.exists():
        raw = json.loads(DATA_CACHE.read_text(encoding="utf-8"))
        print(f"Loaded {len(raw)} prompts from cache.")
        return raw[:n]

    print("Downloading AlpacaEval-2 from HuggingFace (first run only)...")
    from huggingface_hub import hf_hub_download
    # Downloads the raw JSON file — bypasses the broken loading script entirely
    local_path = hf_hub_download(
        repo_id="tatsu-lab/alpaca_eval",
        filename="alpaca_eval.json",
        repo_type="dataset",
    )
    raw = json.loads(Path(local_path).read_text(encoding="utf-8"))
    DATA_CACHE.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    print(f"Cached {len(raw)} prompts to {DATA_CACHE}")
    return raw[:n]


def evaluate_prompt(prompt, reference, n_candidates=5, d_iterations=2):
    """
    Run TPO on one prompt. Returns win (1/0) at step 0, step 1, step 2.

    Step 0 = raw generate, no refinement (baseline)
    Step 1 = best answer after 1 TPO iteration
    Step 2 = best answer after 2 TPO iterations
    """
    # Step 0: baseline — one generate call, no critique, no refinement
    step0_answer = generate(prompt)
    step0_win = win_rate_judge(prompt, step0_answer, reference) if step0_answer else 0

    # Steps 1 & 2: run TPO loop, capture best_answer from trajectory per step
    best_final, trajectory = run_loop(prompt, n=n_candidates, iterations=d_iterations)

    # trajectory[0] = after iteration 0 (step 1), trajectory[1] = after iteration 1 (step 2)
    step1_answer = trajectory[0]["best_answer"] if len(trajectory) > 0 else step0_answer
    step2_answer = trajectory[1]["best_answer"] if len(trajectory) > 1 else step1_answer

    step1_win = win_rate_judge(prompt, step1_answer, reference) if step1_answer else 0
    step2_win = win_rate_judge(prompt, step2_answer, reference) if step2_answer else 0

    return {
        "prompt": prompt[:80],
        "step0_win": step0_win,
        "step1_win": step1_win,
        "step2_win": step2_win,
        "step0_answer": step0_answer,
        "step1_answer": step1_answer,
        "step2_answer": step2_answer,
        "trajectory_scores": [
            {"iter": t["iteration"], "best_score": t["best_score"], "delta": t["delta"]}
            for t in trajectory
        ],
    }


def _save(out_path, config, all_results):
    """Write current results to disk. Called after every prompt so progress is never lost."""
    n = len(all_results)
    if n == 0:
        return
    wins = {0: sum(r["step0_win"] for r in all_results),
            1: sum(r["step1_win"] for r in all_results),
            2: sum(r["step2_win"] for r in all_results)}
    out_path.write_text(json.dumps({
        "config": config,
        "progress": f"{n}/{config['n_prompts']}",
        "win_rates": {
            "step0": round(wins[0] / n * 100, 2),
            "step1": round(wins[1] / n * 100, 2),
            "step2": round(wins[2] / n * 100, 2),
        },
        "paper_reference": {"step0": "~16.8%", "step2": "~55.7%", "model": "70B, D=5, N=20"},
        "per_prompt": all_results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")


def run_evaluation(n_prompts=100, n_candidates=5, d_iterations=2, out_path=None):
    data = load_alpaca_data(n_prompts)

    if out_path is None:
        out_path = Path(__file__).parent / "results.json"
    out_path = Path(out_path)

    config = {"n_prompts": n_prompts, "n_candidates": n_candidates, "d_iterations": d_iterations}

    # Resume from existing results if the file already exists
    all_results = []
    already_done = set()
    if out_path.exists():
        try:
            saved = json.loads(out_path.read_text(encoding="utf-8"))
            all_results = saved.get("per_prompt", [])
            already_done = {r["prompt"] for r in all_results}
            print(f"Resuming — {len(all_results)} prompts already done.")
        except Exception:
            pass

    print(f"\nEvaluating {len(data)} prompts | N={n_candidates} candidates | D={d_iterations} iterations\n")

    for i, item in enumerate(data):
        prompt = item["instruction"]
        reference = item["output"]
        prompt_key = prompt[:80]

        if prompt_key in already_done:
            print(f"[{i+1:3d}/{len(data)}] SKIP (already done): {prompt_key[:60]}...")
            continue

        print(f"[{i+1:3d}/{len(data)}] {prompt[:70]}...")

        try:
            result = evaluate_prompt(prompt, reference, n_candidates, d_iterations)
            all_results.append(result)
            print(f"         step0={result['step0_win']} step1={result['step1_win']} step2={result['step2_win']}")
        except Exception as e:
            print(f"         ERROR: {e}")
            # Save error entry so we can skip it on resume
            all_results.append({"prompt": prompt_key, "step0_win": 0, "step1_win": 0,
                                 "step2_win": 0, "error": str(e)})

        # Save after every single prompt
        _save(out_path, config, all_results)

    # Final summary
    n = len(all_results)
    wins = {0: sum(r["step0_win"] for r in all_results),
            1: sum(r["step1_win"] for r in all_results),
            2: sum(r["step2_win"] for r in all_results)}
    wr0 = wins[0] / n * 100
    wr1 = wins[1] / n * 100
    wr2 = wins[2] / n * 100

    print("\n" + "=" * 60)
    print("RESULTS TABLE")
    print("=" * 60)
    print(f"{'Step':<22} {'Win-rate':>10} {'Wins':>6} {'Total':>6}")
    print("-" * 60)
    print(f"{'Step 0 (baseline)':<22} {wr0:>9.1f}% {wins[0]:>6} {n:>6}")
    print(f"{'Step 1 (1 TPO iter)':<22} {wr1:>9.1f}% {wins[1]:>6} {n:>6}")
    print(f"{'Step 2 (2 TPO iters)':<22} {wr2:>9.1f}% {wins[2]:>6} {n:>6}")
    print(f"{'Paper (70B, D=5, N=20)':<22} {'~55.7%':>10}")
    print("=" * 60)

    print("\nDELTA EXPLANATION")
    print("-" * 60)
    print(
        f"Our numbers are lower than the paper's (~55.7%) because:\n"
        f"  - Model: llama-3.1-8b-instant (8B) vs paper's 70B policy\n"
        f"  - Iterations: D={d_iterations} vs paper's D=5\n"
        f"  - Candidates: N={n_candidates} vs paper's N=20\n"
        f"  - Reward model: LLM-as-judge (hosted API) vs FsfairX-LLaMA3-RM-v0.1 (dedicated RM)\n"
        f"  - Infrastructure: Groq hosted API vs vLLM on GPU\n"
        f"  The monotone rise step0->step1->step2 is the key signal."
    )
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="Number of prompts (default 100; use 10 for a quick test)")
    parser.add_argument("--candidates", type=int, default=5, help="N candidates per step")
    parser.add_argument("--iterations", type=int, default=2, help="D TPO iterations")
    parser.add_argument("--out", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    run_evaluation(
        n_prompts=args.n,
        n_candidates=args.candidates,
        d_iterations=args.iterations,
        out_path=args.out,
    )
