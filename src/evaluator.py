import re
import string
from typing import Dict, Any, List, Optional
from src.llm_client import LLMClient

def normalize_text(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score between prediction and ground truth."""
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(ground_truth).split()
    
    if not pred_tokens or not truth_tokens:
        return 1.0 if pred_tokens == truth_tokens else 0.0
    
    common = set(pred_tokens) & set(truth_tokens)
    if not common:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    return round(f1, 4)

def check_refusal(text: str) -> bool:
    """Check if the response constitutes a refusal or declaration of insufficient info."""
    refusal_patterns = [
        r"do not have enough information",
        r"not present in the provided context",
        r"not mentioned in the provided text",
        r"cannot be answered based on the provided",
        r"i don't know",
        r"information is not provided",
        r"no information"
    ]
    lower = text.lower()
    return any(re.search(pat, lower) for pat in refusal_patterns)


class Evaluator:
    """Evaluates Baseline and RAG outputs for Faithfulness, Hallucination, and Correctness."""
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    def evaluate_sample(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        category: str,
        in_corpus: bool,
        context: str = ""
    ) -> Dict[str, Any]:
        """
        Evaluate a single answer against ground truth, context, and category.
        """
        answer_clean = answer.strip()
        is_refusal = check_refusal(answer_clean)
        
        # 1. Out-of-corpus handling
        if not in_corpus or category == "Out_Of_Corpus":
            if is_refusal:
                # Correct refusal -> No hallucination, High Faithfulness, Correct
                return {
                    "factual_correctness": "Correct",
                    "correctness_score": 1.0,
                    "faithfulness": 1.0,
                    "hallucinated": 0,
                    "helpfulness": 5,
                    "f1_score": 1.0,
                    "is_refusal": True,
                    "eval_note": "Accurately refused unanswerable query."
                }
            else:
                # Confabulated answer on out-of-corpus question -> Hallucination!
                return {
                    "factual_correctness": "Incorrect",
                    "correctness_score": 0.0,
                    "faithfulness": 0.0,
                    "hallucinated": 1,
                    "helpfulness": 1,
                    "f1_score": compute_f1(answer_clean, ground_truth),
                    "is_refusal": False,
                    "eval_note": "Hallucinated fictitious facts on out-of-corpus query."
                }

        # 2. Adversarial & Misconceptions
        if category == "Adversarial_Misconception":
            f1 = compute_f1(answer_clean, ground_truth)
            # Check if it properly debunked the premise
            gt_lower = ground_truth.lower()
            ans_lower = answer_clean.lower()
            
            # If ground truth starts with 'No', did answer start with 'No' or reject premise?
            debunked = ("no" in ans_lower[:10]) if gt_lower.startswith("no") else True
            
            if debunked and f1 > 0.25:
                return {
                    "factual_correctness": "Correct",
                    "correctness_score": 1.0,
                    "faithfulness": 1.0,
                    "hallucinated": 0,
                    "helpfulness": 5,
                    "f1_score": f1,
                    "is_refusal": is_refusal,
                    "eval_note": "Correctly debunked adversarial premise using facts."
                }
            elif debunked:
                return {
                    "factual_correctness": "Partially Correct",
                    "correctness_score": 0.5,
                    "faithfulness": 0.8,
                    "hallucinated": 0,
                    "helpfulness": 3,
                    "f1_score": f1,
                    "is_refusal": is_refusal,
                    "eval_note": "Partially aligned with factual ground truth."
                }
            else:
                return {
                    "factual_correctness": "Incorrect",
                    "correctness_score": 0.0,
                    "faithfulness": 0.0,
                    "hallucinated": 1,
                    "helpfulness": 1,
                    "f1_score": f1,
                    "is_refusal": is_refusal,
                    "eval_note": "Fell into adversarial trap / hallucinated affirmation."
                }

        # 3. Direct Fact and Multi-Hop
        f1 = compute_f1(answer_clean, ground_truth)
        
        # Check faithfulness against retrieved context if available
        if context:
            # Check if key tokens in answer exist in context
            ans_tokens = normalize_text(answer_clean).split()
            ctx_normalized = normalize_text(context)
            matching_tokens = [t for t in ans_tokens if t in ctx_normalized]
            context_overlap = len(matching_tokens) / max(len(ans_tokens), 1)
            faithfulness_score = round(min(1.0, context_overlap * 1.1), 2)
        else:
            # Baseline (no context) - faithfulness is evaluated vs factual correctness
            faithfulness_score = 1.0 if f1 >= 0.6 else (0.5 if f1 >= 0.3 else 0.0)

        if f1 >= 0.55:
            correctness = "Correct"
            c_score = 1.0
            hallucinated = 0
            helpfulness = 5
        elif f1 >= 0.25:
            correctness = "Partially Correct"
            c_score = 0.5
            hallucinated = 1 if faithfulness_score < 0.5 else 0
            helpfulness = 3
        else:
            correctness = "Incorrect"
            c_score = 0.0
            hallucinated = 1
            helpfulness = 1

        return {
            "factual_correctness": correctness,
            "correctness_score": c_score,
            "faithfulness": faithfulness_score,
            "hallucinated": hallucinated,
            "helpfulness": helpfulness,
            "f1_score": f1,
            "is_refusal": is_refusal,
            "eval_note": f"Evaluated with F1: {f1:.2f}, Faithfulness: {faithfulness_score:.2f}."
        }

    def compute_aggregate_metrics(self, evaluated_rows: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute summary benchmark metrics across a set of evaluated rows."""
        total = len(evaluated_rows)
        if total == 0:
            return {}

        hallucination_count = sum(r.get("hallucinated", 0) for r in evaluated_rows)
        correct_count = sum(1 for r in evaluated_rows if r.get("factual_correctness") == "Correct")
        partial_count = sum(1 for r in evaluated_rows if r.get("factual_correctness") == "Partially Correct")
        avg_faithfulness = sum(r.get("faithfulness", 0.0) for r in evaluated_rows) / total
        avg_f1 = sum(r.get("f1_score", 0.0) for r in evaluated_rows) / total
        avg_helpfulness = sum(r.get("helpfulness", 1) for r in evaluated_rows) / total

        return {
            "total_questions": total,
            "hallucination_rate_pct": round((hallucination_count / total) * 100.0, 2),
            "correct_rate_pct": round((correct_count / total) * 100.0, 2),
            "partial_rate_pct": round((partial_count / total) * 100.0, 2),
            "accuracy_score_pct": round(((correct_count + 0.5 * partial_count) / total) * 100.0, 2),
            "avg_faithfulness_pct": round(avg_faithfulness * 100.0, 2),
            "avg_f1_score": round(avg_f1, 4),
            "avg_helpfulness_1to5": round(avg_helpfulness, 2)
        }
