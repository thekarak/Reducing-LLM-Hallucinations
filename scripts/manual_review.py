"""Generate a manual review sheet for human verification of automatic metrics.

Samples N rows (stratified by category) from results/results.csv and writes
results/manual_review.csv with blank reviewer columns.

Workflow:
    1. python scripts/manual_review.py            # creates the review sheet
    2. Open results/manual_review.csv and fill in:
       - reviewer_hallucinated: 1 if the answer contains facts not supported by
         the context/ground truth, else 0
       - reviewer_faithfulness_0_5: your own 0-5 faithfulness judgement
       - reviewer_notes: anything worth noting (optional)
    3. Commit the filled CSV so reviewers can see the human-verified subset.
"""
import argparse
import pandas as pd
from pathlib import Path

from src.config import RESULTS_FILE, MANUAL_REVIEW_FILE


def build_review_sheet(n_per_category: int = 8, out_path: Path = MANUAL_REVIEW_FILE):
    if not RESULTS_FILE.exists():
        raise SystemExit("results/results.csv not found. Run `python run_experiments.py` first.")

    df = pd.read_csv(RESULTS_FILE)
    sampled = (
        df.groupby("category", group_keys=False)
        .apply(lambda g: g.sample(n=min(n_per_category, len(g)), random_state=42))
        .reset_index(drop=True)
    )

    rows = []
    for _, r in sampled.iterrows():
        rows.append({
            "id": r["id"], "category": r["category"],
            "question": r["question"], "ground_truth": r["ground_truth"],
            "system": "baseline", "answer": r["baseline_answer"],
            "auto_hallucinated": r["baseline_hallucinated"],
            "reviewer_hallucinated": "", "reviewer_faithfulness_0_5": "", "reviewer_notes": "",
        })
        rows.append({
            "id": r["id"], "category": r["category"],
            "question": r["question"], "ground_truth": r["ground_truth"],
            "system": "rag_k3", "answer": r["rag_k3_answer"],
            "auto_hallucinated": r["rag_k3_hallucinated"],
            "reviewer_hallucinated": "", "reviewer_faithfulness_0_5": "", "reviewer_notes": "",
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Wrote {len(out_df)} review rows ({len(sampled)} questions x 2 systems) to {out_path}")
    print("Fill in reviewer_hallucinated, reviewer_faithfulness_0_5, reviewer_notes, then commit the file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a manual metric-verification sheet.")
    parser.add_argument("--n-per-category", type=int, default=8,
                        help="Questions sampled per category (each yields a baseline + RAG row).")
    args = parser.parse_args()
    build_review_sheet(args.n_per_category)
