"""
TPO high level idea
policy model: generate N samples/answers
reward model: score the generated N samples
textual critique/ gradient: evaluate the answer vs the worst
optimizer/refinement: improve the answer with the critique

refer to https://github.com/Simplified-Reasoning/TPO
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm import client

JUDGE_MODEL = "llama-3.3-70b-versatile"  # None = use client default (llama-3.1-8b-instant via Groq)
MAX_CHARS = 8000    # character-based truncation (no tokenizer needed)


def generate(prompt, base=None, gradient=None, temperature=0.9):
    """Generate one candidate answer. Pass base+gradient on iteration 2+ for refinement."""
    sys_msg = "You are a helpful assistant. Answer the user's instruction well."
    if base and gradient:
        user = (f"{prompt}\nCurrent best answer:{base}\n"
                f"Feedback on how to improve it:{gradient}\n"
                "Write an improved, distinct answer that applies the feedback.")
    else:
        user = prompt
    return client.chat(user, system=sys_msg, temperature=temperature)


def reward(prompt, answer):
    """Score one answer 0-100 using LLM-as-judge. Returns float."""
    out = client.chat_json(
        f"Instruction:{prompt}\nResponse:{answer}\n"
        "Rate the response 0-100. Use the FULL range: "
        "a mediocre answer should score 40-55, a good answer 65-78, excellent 85+. "
        "Most answers should NOT score above 80."
        'Return ONLY JSON: {"score": <integer>}',
        model=JUDGE_MODEL)
    try:
        return float(out["score"])
    except Exception:
        return 50.0


def get_contrastive_samples(scores, qa_pairs):
    """
    Pick best and worst (prompt, answer) pairs by reward score.
    Returns {"best": answer, "worst": answer}, delta (score gap).
    """
    def truncate_text(text, max_chars=MAX_CHARS):
        """prevent the context windows being too long"""
        return text[:max_chars]

    best_index = int(np.argmax(scores))
    worst_index = int(np.argmin(scores))
    delta = max(scores) - min(scores)

    best_answer = truncate_text(qa_pairs[best_index][1])
    worst_answer = truncate_text(qa_pairs[worst_index][1])
    return {"best": best_answer, "worst": worst_answer}, delta


def textual_gradient(prompt, best, worst):
    """
    The textual gradient: compare chosen vs rejected and explain how to improve.
    Mirrors EVALUATION_SYS_TEMPLATE from the original TPO repo.
    Returns a critique string.
    """
    sys_msg = (
        "You are evaluating a chosen response by comparing it with a rejected response "
        "to a user query. Analyze the strengths and weaknesses of each response step by step, "
        "and explain why one is chosen and the other rejected. Be concise."
    )
    user = (
        f"User Query:{prompt}\n"
        f"Rejected Response:{worst}\n"
        f"Chosen Response:{best}\n"
        "What makes the chosen response better? "
        "What specific improvements would make the chosen response even stronger?"
    )
    return client.chat(user, system=sys_msg, temperature=0.3)


def win_rate_judge(prompt, candidate, reference):
    """
    Head-to-head judge: does candidate beat reference?
    Returns 1 (win) or 0 (loss).
    Mirrors AlpacaEval-2 judge: compare your answer vs GPT-4 reference answer.
    """
    out = client.chat_json(
        f"Instruction:\n{prompt}\n\n"
        f"Response A:\n{candidate}\n\n"
        f"Response B:\n{reference}\n\n"
        "Which response better follows the instruction? "
        "Consider helpfulness, accuracy, completeness, and clarity. "
        'Return ONLY JSON: {"winner": "A" or "B", "reason": "<one sentence>"}',
        system="You are an impartial judge evaluating two responses to an instruction.",
        temperature=0.0,
    )
    try:
        return 1 if out["winner"] == "A" else 0
    except Exception:
        return 0


def run_loop(prompt, n=5, iterations=3):
    """
    Main TPO loop.
    - Generates N candidates per iteration
    - Scores all candidates (accumulated across iterations, like the original repo)
    - Picks best/worst via get_contrastive_samples
    - Computes textual gradient (critique)
    - Uses best+critique to seed next iteration's generation
    Returns (final_best_answer, trajectory)
    """
    # Accumulated cache of all (prompt, answer) pairs and their scores across all iterations
    all_qa_pairs = []
    all_scores = []
    trajectory = []

    current_best = None
    current_gradient = None

    for it in range(iterations):
        # --- Step 1: Generate N candidates ---
        candidates = [generate(prompt, base=current_best, gradient=current_gradient)
                      for _ in range(n)]

        # --- Step 2: Score each candidate ---
        new_scores = [reward(prompt, c) for c in candidates]
        new_qa_pairs = [(prompt, c) for c in candidates]

        # --- Step 3: Accumulate across iterations (key TPO insight) ---
        all_qa_pairs.extend(new_qa_pairs)
        all_scores.extend(new_scores)

        # --- Step 4: Pick best and worst from full accumulated history ---
        contrastive, delta = get_contrastive_samples(all_scores, all_qa_pairs)
        current_best = contrastive["best"]
        current_worst = contrastive["worst"]

        # --- Step 5: Compute textual gradient (critique) ---
        current_gradient = textual_gradient(prompt, current_best, current_worst)

        trajectory.append({
            "iteration": it,
            "n_candidates": len(candidates),
            "scores": new_scores,
            "best_score": max(all_scores),
            "worst_score": min(all_scores),
            "delta": delta,
            "critique": current_gradient,
            "best_answer": current_best,   # needed for per-step win-rate evaluation
        })

        print(f"[iter {it}] best={max(all_scores):.1f} worst={min(all_scores):.1f} delta={delta:.1f}")

        # Early stop: if gap is tiny, refinement won't help much
        # if delta < 2.0:
        #     print("  delta too small, stopping early.")
        #     break

    return current_best, trajectory


def main():
    import json
    from pathlib import Path

    cache = Path(__file__).parent / "alpaca_eval_cache.json"
    prompts = [item["instruction"] for item in json.loads(cache.read_text(encoding="utf-8"))]

    for prompt in prompts[:2]:  # run on first 2 prompts as a smoke test
        print(f"PROMPT: {prompt}")
        best, traj = run_loop(prompt, n=5, iterations=2)
        print(f"\nFINAL BEST ANSWER:\n{best}")
        print(f"\nTRAJECTORY SUMMARY:")
        for step in traj:
            print(f"  iter={step['iteration']} best={step['best_score']:.1f} delta={step['delta']:.1f}")


if __name__ == "__main__":
    main()
