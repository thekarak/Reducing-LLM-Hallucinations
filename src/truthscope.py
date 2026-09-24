import re
from typing import Any, Dict, List, Optional, Sequence

from src.data_loader import Document
from src.evaluator import check_refusal

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from",
    "had", "has", "have", "he", "her", "his", "in", "is", "it", "its", "of",
    "on", "or", "she", "that", "the", "their", "there", "they", "this", "to",
    "was", "were", "which", "with", "would", "could", "should", "about", "into"
}

STATUS_ORDER = {"supported": 0, "uncertain": 1, "unsupported": 2, "unverified": 3, "abstained": 4}


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())


def _salient_tokens(text: str) -> set:
    return {
        token for token in _tokens(text)
        if token not in STOPWORDS and (len(token) >= 4 or any(char.isdigit() for char in token))
    }


def _has_negation(text: str) -> bool:
    return bool(re.search(r"\b(?:no|not|never|nor|cannot|without|n't)\b", text.lower()))


def split_claims(answer: str) -> List[str]:
    normalized = re.sub(r"\s+", " ", answer.strip())
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+|(?:^|\s)[-*]\s+", normalized)
    claims = []
    for part in parts:
        cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", part).strip()
        if cleaned:
            claims.append(cleaned)
    return claims


def _evidence_sentences(documents: Sequence[Document]) -> List[Dict[str, Any]]:
    evidence = []
    for doc_index, document in enumerate(documents):
        source = document.metadata.get("source", f"document {doc_index + 1}")
        sentences = re.split(r"(?<=[.!?])\s+|\n+", document.page_content.strip())
        for sentence_index, sentence in enumerate(sentences):
            cleaned = sentence.strip()
            if cleaned:
                evidence.append({
                    "text": cleaned,
                    "source": source,
                    "document_index": doc_index,
                    "sentence_index": sentence_index,
                })
    return evidence


def _claim_support(claim: str, evidence: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    claim_tokens = _salient_tokens(claim)
    if not claim_tokens:
        return {
            "status": "uncertain",
            "support": 0.0,
            "explanation": "This claim has no specific words to verify against the evidence.",
            "evidence": None,
        }

    best = None
    for sentence in evidence:
        evidence_tokens = _salient_tokens(sentence["text"])
        overlap = len(claim_tokens & evidence_tokens)
        support = overlap / len(claim_tokens)
        candidate = {"support": support, "evidence": sentence}
        if best is None or support > best["support"] or (
            support == best["support"] and len(sentence["text"]) < len(best["evidence"]["text"])
        ):
            best = candidate

    support = round(best["support"] if best else 0.0, 2)
    negation_conflict = bool(
        best and _has_negation(claim) != _has_negation(best["evidence"]["text"])
    )
    if negation_conflict:
        support = 0.0
    if support >= 0.7:
        status = "supported"
        explanation = "Most key terms in this claim appear in the cited evidence."
    elif support >= 0.4:
        status = "uncertain"
        explanation = "Some key terms match, but the evidence does not verify the complete claim."
    else:
        status = "unsupported"
        explanation = (
            "The closest evidence has conflicting negation or wording."
            if negation_conflict
            else "The retrieved evidence does not directly support this claim."
        )
    return {
        "status": status,
        "support": support,
        "explanation": explanation,
        "evidence": best["evidence"] if best else None,
    }


def analyze_claims(
    answer: str,
    documents: Optional[Sequence[Document]] = None,
    retrieval_scores: Optional[Sequence[float]] = None
) -> Dict[str, Any]:
    evidence_documents = list(documents or [])
    evidence_sentences = _evidence_sentences(evidence_documents)
    claims = split_claims(answer)
    refusal = check_refusal(answer)

    analyzed = []
    for index, claim in enumerate(claims, 1):
        result = _claim_support(claim, evidence_sentences)
        if refusal:
            result.update({
                "status": "abstained",
                "support": 1.0,
                "explanation": "The system declined to make a factual claim without evidence.",
                "evidence": None,
            })
        elif not evidence_documents:
            result.update({
                "status": "unverified",
                "support": 0.0,
                "explanation": "No retrieved evidence was available to verify this claim.",
                "evidence": None,
            })
        document_index = result["evidence"]["document_index"] if result["evidence"] else None
        analyzed.append({
            "id": index,
            "claim": claim,
            "status": result["status"],
            "support": result["support"],
            "explanation": result["explanation"],
            "evidence": result["evidence"],
            "retrieval_score": (
                round(retrieval_scores[document_index], 3)
                if document_index is not None and document_index < len(retrieval_scores)
                else None
            ),
        })

    counts = {status: sum(claim["status"] == status for claim in analyzed) for status in STATUS_ORDER}
    if not analyzed:
        overall_status = "unverified"
    else:
        overall_status = max(
            (claim["status"] for claim in analyzed),
            key=lambda status: STATUS_ORDER[status]
        )
    return {
        "claims": analyzed,
        "summary": {
            "overall_status": overall_status,
            "verified_claims": counts["supported"],
            "uncertain_claims": counts["uncertain"] + counts["unverified"],
            "unsupported_claims": counts["unsupported"],
            "abstained": bool(refusal),
            "citation_coverage_pct": round(
                100 * sum(claim["support"] >= 0.4 for claim in analyzed) / len(analyzed), 1
            ) if analyzed else 0.0,
        },
    }


def assess_risk(
    question: str,
    documents: Optional[Sequence[Document]] = None,
    retrieval_scores: Optional[Sequence[float]] = None,
    category: str = "Custom"
) -> Dict[str, Any]:
    retrieved_count = len(documents or [])
    scores = list(retrieval_scores or [])
    top_score = scores[0] if scores else 0.0
    reasons = []
    score = 10

    normalized_category = category.lower()
    if normalized_category == "out_of_corpus":
        score += 65
        reasons.append("The benchmark marks this as outside the available knowledge corpus.")
    elif normalized_category == "adversarial_misconception":
        score += 45
        reasons.append("The question may contain a false premise.")
    elif normalized_category == "multi_hop":
        score += 18
        reasons.append("The answer may need evidence from multiple sources.")

    risky_patterns = {
        r"\b(fictional|fake|imaginary|never happened|alternate reality)\b": 30,
        r"\b(prove|proved|definitively|confirmed discovery)\b": 20,
        r"\b(first ever|the only|always|never|all)\b": 15,
        r"\b(exactly|specifically|precisely|how many)\b": 8,
    }
    for pattern, points in risky_patterns.items():
        if re.search(pattern, question.lower()):
            score += points
            reasons.append("The wording asks for a precise or assumption-sensitive fact.")
            break

    if retrieved_count == 0:
        score += 55
        reasons.append("No knowledge chunk passed the relevance threshold.")
    elif retrieved_count == 1:
        score += 15
        reasons.append("Only one evidence chunk was found, so corroboration is limited.")
    if scores and top_score < 0.45:
        score += 15
        reasons.append("The strongest retrieval score indicates weak semantic confidence.")

    score = min(score, 100)
    if score >= 65:
        level = "High"
    elif score >= 35:
        level = "Medium"
    else:
        level = "Low"
    return {
        "level": level,
        "score": score,
        "retrieved_chunks": retrieved_count,
        "top_score": round(top_score, 3),
        "reasons": reasons or ["The query has a conventional structure and retrieved supporting context."],
    }


def build_learning_coach(
    analysis: Dict[str, Any],
    risk: Dict[str, Any],
    category: str = "Custom"
) -> Dict[str, Any]:
    summary = analysis["summary"]
    if category.lower() == "adversarial_misconception":
        lesson = "Treat the premise as something to verify, not something to assume. Compare each part of the question with the cited evidence before agreeing."
        practice = "Rewrite this question as a neutral claim, then find one piece of evidence that supports or rejects it."
    elif summary["abstained"]:
        lesson = "Abstention is a successful safety behavior when the corpus cannot answer the question."
        practice = "Identify one keyword in the question and test whether the indexed documents contain that concept."
    elif summary["unsupported_claims"]:
        lesson = "The answer contains a knowledge gap: some wording is not grounded in the retrieved evidence."
        practice = "Open the cited sentence and highlight the exact words that justify or contradict the answer."
    else:
        lesson = "The answer is reasonably grounded, but confidence should come from evidence rather than fluency."
        practice = "Find a second source for the main claim and compare the dates or measurements."

    if risk["level"] == "High":
        lesson += " Pause before trusting this response and verify its premises first."
    return {
        "lesson": lesson,
        "practice": practice,
        "next_step": "Run a more specific question that names the mission, instrument, date, or measurement you need.",
    }
