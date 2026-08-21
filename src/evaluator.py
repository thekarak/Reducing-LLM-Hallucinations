import re
import json
import string
from typing import Dict, Any, List, Optional
from src.llm_client import LLMClient

# ---------------------------------------------------------------------------
# Text normalisation & token-level metrics
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Heuristic context-support scoring (offline fallback for the LLM judge)
# ---------------------------------------------------------------------------

_CLAUSE_STOPWORDS = {
    "the", "and", "was", "were", "with", "from", "that", "this", "have",
    "has", "had", "its", "their", "which", "while", "when", "during",
    "after", "before", "between", "into", "onto", "about", "also", "than",
    "then", "over", "under", "both", "whereas", "however", "based", "only",
    "using", "uses", "use", "answer", "question", "context", "provided"
}

def _salient_tokens(text: str) -> set:
    raw = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9.\-]*", text.lower())
    tokens = set()
    for tok in raw:
        clean = tok.strip(".-")
        if not clean:
            continue
        if any(ch.isdigit() for ch in clean):
            tokens.add(clean)
        elif len(clean) >= 4 and clean not in _CLAUSE_STOPWORDS:
            tokens.add(clean)
    return tokens

def clause_support_score(answer: str, context: str) -> float:
    """Fraction of answer clauses whose salient tokens are supported by the context.

    More forgiving than raw token overlap: each sentence/clause is checked
    independently, so one unsupported clause does not zero out an otherwise
    grounded answer.
    """
    if not context or not answer.strip():
        return 0.0
    ctx_tokens = _salient_tokens(context)
    clauses = [c for c in re.split(r"(?<=[.!?])\s+|;\s+|\s—\s", answer.strip()) if c.strip()]
    if not clauses:
        return 0.0
    supported = 0
    for clause in clauses:
        needed = _salient_tokens(clause)
        if not needed:
            supported += 1  # purely structural clause (e.g. "No.")
            continue
        if len(needed & ctx_tokens) / len(needed) >= 0.5:
            supported += 1
    return round(supported / len(clauses), 2)

# ---------------------------------------------------------------------------
# LLM-as-Judge prompt
# ---------------------------------------------------------------------------

FAITHFULNESS_JUDGE_PROMPT = """You are a strict, impartial judge evaluating a Retrieval-Augmented Generation (RAG) system.

[Retrieved Context]
{context}

[Question]
{question}

[Answer to Evaluate]
{answer}

Your task:
1. faithfulness (integer 0-5): How much of the answer is DIRECTLY supported by the retrieved context?
   - 5 = every claim is fully supported by the context
   - 3 = mostly supported, minor unsupported details
   - 1 = largely unsupported or contradicts the context
   - 0 = completely fabricated relative to the context
2. hallucinated (true/false): Does the answer assert ANY factual claim that is NOT supported by (or contradicts) the context? A missing answer or an admitted lack of information is NOT a hallucination.

Respond with ONLY a JSON object, no other text:
{{"faithfulness": <0-5>, "hallucinated": <true|false>, "reason": "<one short sentence>"}}"""


def _parse_judge_json(raw: str) -> Optional[Dict[str, Any]]:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        faithfulness = float(data["faithfulness"])
        hallucinated = bool(data.get("hallucinated", False))
        if not 0.0 <= faithfulness <= 5.0:
            return None
        return {"faithfulness": faithfulness / 5.0, "hallucinated": hallucinated}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None

# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """Evaluates Baseline and RAG outputs for Faithfulness, Hallucination, and Correctness.

    Faithfulness measurement strategy:
    - With a live LLM provider, an LLM-as-Judge prompt scores whether each answer
      is supported by the retrieved context (primary method).
    - Offline (`local_mock`) runs fall back to a deterministic clause-support
      heuristic so results remain reproducible without API keys.
    """
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()
        self.judge_available = self.llm.provider not in ("local_mock",)

    # ------------------------------------------------------------------ judge
    def judge_faithfulness(self, question: str, answer: str, context: str) -> Optional[Dict[str, Any]]:
        """Ask the LLM judge to score answer-vs-context faithfulness. Returns None on failure."""
        if not self.judge_available:
            return None
        prompt = FAITHFULNESS_JUDGE_PROMPT.format(context=context, question=question, answer=answer)
        try:
            raw = self.llm.generate(prompt=prompt, temperature=0.0)
            return _parse_judge_json(raw)
        except Exception:
            return None

    # --------------------------------------------------------------- evaluate
    def evaluate_sample(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        category: str,
        in_corpus: bool,
        context: str = ""
    ) -> Dict[str, Any]:
        """Evaluate a single answer against ground truth, context, and category."""
        answer_clean = answer.strip()
        is_refusal = check_refusal(answer_clean)

        # 1. Out-of-corpus handling
        if not in_corpus or category == "Out_Of_Corpus":
            if is_refusal:
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

            # Refusing to engage with a false-premise question is safe behaviour:
            # it is a miss on helpfulness but NOT a hallucination.
            if is_refusal:
                return {
                    "factual_correctness": "Incorrect",
                    "correctness_score": 0.0,
                    "faithfulness": 1.0,
                    "hallucinated": 0,
                    "helpfulness": 2,
                    "f1_score": f1,
                    "is_refusal": True,
                    "eval_note": "Abstained on adversarial premise; counted as a miss, not a hallucination."
                }

            gt_lower = ground_truth.lower()
            ans_lower = answer_clean.lower()
            debunked = ("no" in ans_lower[:10]) if gt_lower.startswith("no") else True

            # Judge cross-check when a live LLM is available.
            judge_verdict = None
            if context:
                judge_verdict = self.judge_faithfulness(question, answer_clean, context)

            if debunked and f1 > 0.25:
                hallucinated = 0
                if judge_verdict is not None and judge_verdict["hallucinated"]:
                    hallucinated = 1
                return {
                    "factual_correctness": "Correct",
                    "correctness_score": 1.0,
                    "faithfulness": judge_verdict["faithfulness"] if judge_verdict else 1.0,
                    "hallucinated": hallucinated,
                    "helpfulness": 5,
                    "f1_score": f1,
                    "is_refusal": is_refusal,
                    "eval_note": "Correctly debunked adversarial premise using facts."
                }
            elif debunked:
                return {
                    "factual_correctness": "Partially Correct",
                    "correctness_score": 0.5,
                    "faithfulness": judge_verdict["faithfulness"] if judge_verdict else 0.8,
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
                    "faithfulness": judge_verdict["faithfulness"] if judge_verdict else 0.0,
                    "hallucinated": 1,
                    "helpfulness": 1,
                    "f1_score": f1,
                    "is_refusal": is_refusal,
                    "eval_note": "Fell into adversarial trap / hallucinated affirmation."
                }

        # 3. Direct Fact and Multi-Hop
        f1 = compute_f1(answer_clean, ground_truth)

        # An explicit refusal on an answerable question is a retrieval/answering
        # MISS, not a hallucination: no fabricated content was produced.
        if is_refusal:
            return {
                "factual_correctness": "Incorrect",
                "correctness_score": 0.0,
                "faithfulness": 1.0,
                "hallucinated": 0,
                "helpfulness": 1,
                "f1_score": f1,
                "is_refusal": True,
                "eval_note": "Abstained instead of answering (no fabricated content); counted as a miss, not a hallucination."
            }

        if context:
            # Primary: LLM-as-Judge. Fallback: deterministic clause-support heuristic.
            judge_verdict = self.judge_faithfulness(question, answer_clean, context)
            if judge_verdict is not None:
                faithfulness_score = round(judge_verdict["faithfulness"], 2)
                judge_hallucinated = judge_verdict["hallucinated"]
            else:
                faithfulness_score = clause_support_score(answer_clean, context)
                judge_hallucinated = None
        else:
            # Baseline (no context): reference-support proxy, documented in README.
            faithfulness_score = 1.0 if f1 >= 0.6 else (0.5 if f1 >= 0.3 else 0.0)
            judge_hallucinated = None

        if f1 >= 0.55:
            correctness = "Correct"
            c_score = 1.0
            helpfulness = 5
        elif f1 >= 0.25:
            correctness = "Partially Correct"
            c_score = 0.5
            helpfulness = 3
        else:
            correctness = "Incorrect"
            c_score = 0.0
            helpfulness = 1

        if judge_hallucinated is not None:
            hallucinated = 1 if judge_hallucinated else 0
        else:
            hallucinated = 1 if (c_score == 0.0 or (c_score == 0.5 and faithfulness_score < 0.5)) else 0

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
