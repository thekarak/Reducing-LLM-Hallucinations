import os
import csv
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.config import (
    DOCUMENTS_DIR, QUESTIONS_FILE, RESULTS_FILE, PLOTS_DIR,
    VECTOR_STORE_DIR, LLM_PROVIDER, LLM_MODEL
)
from src.data_loader import load_documents, load_questions, RecursiveCharacterTextSplitter
from src.vector_store import SimpleVectorStore, EmbeddingEngine
from src.llm_client import LLMClient
from src.rag_pipeline import BaselinePipeline, RAGPipeline
from src.evaluator import Evaluator
from src.visualization import (
    plot_hallucination_comparison,
    plot_category_breakdown,
    plot_ablation_comparison
)

def run_benchmark():
    print("=" * 70)
    print("  RAG HALLUCINATION EVALUATION BENCHMARK")
    print(f"  Provider: {LLM_PROVIDER.upper()} | Model: {LLM_MODEL}")
    print("=" * 70)

    # 1. Ingest and Chunk Documents
    print("\n[Step 1/5] Ingesting and Chunking Knowledge Corpus...")
    raw_docs = load_documents(DOCUMENTS_DIR)
    print(f"Loaded {len(raw_docs)} documents from {DOCUMENTS_DIR}")
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60)
    chunked_docs = splitter.split_documents(raw_docs)
    print(f"Created {len(chunked_docs)} semantic chunks.")

    # 2. Build Vector Store
    print("\n[Step 2/5] Indexing Chunks in Vector Store...")
    embedding_engine = EmbeddingEngine()
    vector_store = SimpleVectorStore(embedding_engine=embedding_engine)
    vector_store.add_documents(chunked_docs)
    vector_store.save(VECTOR_STORE_DIR)
    print(f"Vector store indexed and saved to {VECTOR_STORE_DIR}")

    # 3. Initialize Pipelines & Evaluator
    llm = LLMClient()
    baseline_pipe = BaselinePipeline(llm_client=llm)
    rag_k3_strict = RAGPipeline(vector_store=vector_store, llm_client=llm, top_k=3, strict_grounding=True)
    rag_k5_strict = RAGPipeline(vector_store=vector_store, llm_client=llm, top_k=5, strict_grounding=True)
    rag_k3_loose = RAGPipeline(vector_store=vector_store, llm_client=llm, top_k=3, strict_grounding=False)
    evaluator = Evaluator(llm_client=llm)

    # 4. Load Evaluation Dataset
    questions = load_questions(QUESTIONS_FILE)
    print(f"\n[Step 3/5] Running Experiments on {len(questions)} Benchmark Questions...")

    results_data = []
    
    for row in tqdm(questions, desc="Evaluating Questions"):
        q_id = row["id"]
        category = row["category"]
        question = row["question"]
        ground_truth = row["ground_truth"]
        in_corpus = row["in_corpus"].strip().lower() == "true"
        source_doc = row["source_doc"]

        # Run 1: Baseline
        res_baseline = baseline_pipe.query(question)
        eval_base = evaluator.evaluate_sample(
            question=question,
            answer=res_baseline["answer"],
            ground_truth=ground_truth,
            category=category,
            in_corpus=in_corpus,
            context=""
        )

        # Run 2: RAG Top-3 (Strict)
        res_rag_k3 = rag_k3_strict.query(question)
        eval_rag_k3 = evaluator.evaluate_sample(
            question=question,
            answer=res_rag_k3["answer"],
            ground_truth=ground_truth,
            category=category,
            in_corpus=in_corpus,
            context=res_rag_k3["retrieved_context"]
        )

        # Run 3: RAG Top-5 (Strict)
        res_rag_k5 = rag_k5_strict.query(question)
        eval_rag_k5 = evaluator.evaluate_sample(
            question=question,
            answer=res_rag_k5["answer"],
            ground_truth=ground_truth,
            category=category,
            in_corpus=in_corpus,
            context=res_rag_k5["retrieved_context"]
        )

        # Run 4: RAG Top-3 (Loose Grounding)
        res_rag_loose = rag_k3_loose.query(question)
        eval_rag_loose = evaluator.evaluate_sample(
            question=question,
            answer=res_rag_loose["answer"],
            ground_truth=ground_truth,
            category=category,
            in_corpus=in_corpus,
            context=res_rag_loose["retrieved_context"]
        )

        record = {
            "id": q_id,
            "category": category,
            "question": question,
            "ground_truth": ground_truth,
            "in_corpus": in_corpus,
            "source_doc": source_doc,
            
            # Baseline Data
            "baseline_answer": res_baseline["answer"],
            "baseline_correctness": eval_base["factual_correctness"],
            "baseline_correct_score": eval_base["correctness_score"],
            "baseline_faithfulness": eval_base["faithfulness"],
            "baseline_hallucinated": eval_base["hallucinated"],
            "baseline_f1": eval_base["f1_score"],
            "baseline_latency_ms": res_baseline["latency_ms"],

            # RAG Top-3 Strict Data
            "rag_k3_answer": res_rag_k3["answer"],
            "rag_k3_correctness": eval_rag_k3["factual_correctness"],
            "rag_k3_correct_score": eval_rag_k3["correctness_score"],
            "rag_k3_faithfulness": eval_rag_k3["faithfulness"],
            "rag_k3_hallucinated": eval_rag_k3["hallucinated"],
            "rag_k3_f1": eval_rag_k3["f1_score"],
            "rag_k3_latency_ms": res_rag_k3["latency_ms"],
            "rag_k3_retrieved_context": res_rag_k3["retrieved_context"],

            # RAG Top-5 Strict Data
            "rag_k5_answer": res_rag_k5["answer"],
            "rag_k5_correctness": eval_rag_k5["factual_correctness"],
            "rag_k5_correct_score": eval_rag_k5["correctness_score"],
            "rag_k5_faithfulness": eval_rag_k5["faithfulness"],
            "rag_k5_hallucinated": eval_rag_k5["hallucinated"],
            "rag_k5_f1": eval_rag_k5["f1_score"],

            # RAG Top-3 Loose Data
            "rag_loose_answer": res_rag_loose["answer"],
            "rag_loose_correctness": eval_rag_loose["factual_correctness"],
            "rag_loose_correct_score": eval_rag_loose["correctness_score"],
            "rag_loose_faithfulness": eval_rag_loose["faithfulness"],
            "rag_loose_hallucinated": eval_rag_loose["hallucinated"],
            "rag_loose_f1": eval_rag_loose["f1_score"]
        }
        results_data.append(record)

    # 5. Save Results CSV
    print(f"\n[Step 4/5] Saving Detailed Results to {RESULTS_FILE}...")
    df_results = pd.DataFrame(results_data)
    df_results.to_csv(RESULTS_FILE, index=False, encoding="utf-8")

    # 6. Aggregate Metric Computations
    print("\n[Step 5/5] Computing Benchmark Summary & Generating Plots...")
    
    # Compute system-level summary
    base_rows = [{"hallucinated": r["baseline_hallucinated"], "factual_correctness": r["baseline_correctness"], "faithfulness": r["baseline_faithfulness"], "f1_score": r["baseline_f1"], "helpfulness": 5 if r["baseline_correct_score"]==1 else (3 if r["baseline_correct_score"]==0.5 else 1)} for r in results_data]
    rag3_rows = [{"hallucinated": r["rag_k3_hallucinated"], "factual_correctness": r["rag_k3_correctness"], "faithfulness": r["rag_k3_faithfulness"], "f1_score": r["rag_k3_f1"], "helpfulness": 5 if r["rag_k3_correct_score"]==1 else (3 if r["rag_k3_correct_score"]==0.5 else 1)} for r in results_data]
    rag5_rows = [{"hallucinated": r["rag_k5_hallucinated"], "factual_correctness": r["rag_k5_correctness"], "faithfulness": r["rag_k5_faithfulness"], "f1_score": r["rag_k5_f1"], "helpfulness": 5 if r["rag_k5_correct_score"]==1 else (3 if r["rag_k5_correct_score"]==0.5 else 1)} for r in results_data]
    ragl_rows = [{"hallucinated": r["rag_loose_hallucinated"], "factual_correctness": r["rag_loose_correctness"], "faithfulness": r["rag_loose_faithfulness"], "f1_score": r["rag_loose_f1"], "helpfulness": 5 if r["rag_loose_correct_score"]==1 else (3 if r["rag_loose_correct_score"]==0.5 else 1)} for r in results_data]

    summary_metrics = {
        "Baseline (No RAG)": evaluator.compute_aggregate_metrics(base_rows),
        "RAG (Top-3 Strict)": evaluator.compute_aggregate_metrics(rag3_rows),
        "RAG (Top-5 Strict)": evaluator.compute_aggregate_metrics(rag5_rows),
        "RAG (Top-3 Loose)": evaluator.compute_aggregate_metrics(ragl_rows)
    }

    # Generate Figures
    plot_hallucination_comparison(summary_metrics)
    plot_category_breakdown(df_results)
    plot_ablation_comparison(summary_metrics)

    # Print Summary Table
    print("\n" + "=" * 78)
    print(f"{'Setting':<25} | {'Hallucination Rate':<20} | {'Faithfulness':<14} | {'Accuracy':<10}")
    print("-" * 78)
    for name, m in summary_metrics.items():
        print(f"{name:<25} | {m['hallucination_rate_pct']:>6.1f}%{' '*12} | {m['avg_faithfulness_pct']:>6.1f}%{' '*6} | {m['accuracy_score_pct']:>6.1f}%")
    print("=" * 78)

    # Hallucination Reduction calculation
    base_hr = summary_metrics["Baseline (No RAG)"]["hallucination_rate_pct"]
    rag_hr = summary_metrics["RAG (Top-3 Strict)"]["hallucination_rate_pct"]
    reduction = ((base_hr - rag_hr) / base_hr) * 100.0 if base_hr > 0 else 0
    print(f"\n>> KEY FINDING: Strict RAG reduced factual hallucination rate by {reduction:.1f}% relative to Baseline LLM!")
    print(f">> Full results saved to: {RESULTS_FILE}")
    print(f">> Plots generated in: {PLOTS_DIR}\n")

if __name__ == "__main__":
    run_benchmark()
