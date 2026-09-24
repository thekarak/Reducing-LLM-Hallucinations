import json

import pandas as pd
import streamlit as st

from src.config import (
    DOCUMENTS_DIR,
    QUESTIONS_FILE,
    RESULTS_FILE,
    SUMMARY_FILE,
    PLOTS_DIR,
    VECTOR_STORE_DIR,
    LLM_PROVIDER,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SIMILARITY_THRESHOLD,
    RETRIEVAL_MODE,
)
from src.data_loader import RecursiveCharacterTextSplitter, load_documents, load_questions
from src.llm_client import LLMClient
from src.rag_pipeline import BaselinePipeline, RAGPipeline
from src.truthscope import analyze_claims, assess_risk, build_learning_coach
from src.vector_store import EmbeddingEngine, SimpleVectorStore

st.set_page_config(page_title="TruthScope", page_icon=None, layout="wide")

PROVIDER_MAP = {
    "Local Simulator": "local_mock",
    "Groq": "groq",
    "OpenCode Zen": "opencode_zen",
    "Google Gemini": "gemini",
    "OpenAI": "openai",
}

STATUS_ICONS = {
    "supported": "Supported",
    "uncertain": "Uncertain",
    "unsupported": "Unsupported",
    "unverified": "Unverified",
    "abstained": "Abstained",
}


@st.cache_resource
def get_system_components():
    raw_docs = load_documents(DOCUMENTS_DIR)
    questions = load_questions(QUESTIONS_FILE)
    embedding_engine = EmbeddingEngine()
    vector_store = None
    if (VECTOR_STORE_DIR / "vectors.npy").exists():
        try:
            vector_store = SimpleVectorStore.load(
                VECTOR_STORE_DIR,
                embedding_engine=embedding_engine,
                similarity_threshold=SIMILARITY_THRESHOLD,
                retrieval_mode=RETRIEVAL_MODE,
            )
        except RuntimeError as error:
            st.warning(f"Rebuilding the saved index: {error}")
    if vector_store is None or not vector_store.documents:
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        vector_store = SimpleVectorStore(
            embedding_engine=embedding_engine,
            similarity_threshold=SIMILARITY_THRESHOLD,
            retrieval_mode=RETRIEVAL_MODE,
        )
        vector_store.add_documents(splitter.split_documents(raw_docs))
        vector_store.save(VECTOR_STORE_DIR)
    return raw_docs, questions, vector_store


@st.cache_data
def load_summary():
    if SUMMARY_FILE.exists():
        with open(SUMMARY_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    return None


def render_claim_analysis(analysis):
    summary = analysis["summary"]
    metric_columns = st.columns(4)
    metric_columns[0].metric("Overall", STATUS_ICONS[summary["overall_status"]])
    metric_columns[1].metric("Verified claims", summary["verified_claims"])
    metric_columns[2].metric("Unsupported", summary["unsupported_claims"])
    metric_columns[3].metric("Citation coverage", f"{summary['citation_coverage_pct']:.0f}%")

    for claim in analysis["claims"]:
        with st.container(border=True):
            title_column, score_column = st.columns([5, 1])
            title_column.markdown(f"**Claim {claim['id']}: {claim['claim']}**")
            score_column.metric("Evidence match", f"{claim['support'] * 100:.0f}%")
            st.write(f"Verdict: {STATUS_ICONS[claim['status']]}")
            st.caption(claim["explanation"])
            evidence = claim["evidence"]
            if evidence:
                st.markdown(
                    f"**Evidence — {evidence['source']}:** {evidence['text']}"
                )


def render_evidence(documents, scores):
    if not documents:
        st.warning("No evidence cleared the relevance threshold. TruthScope should abstain or ask for a more specific question.")
        return
    for index, document in enumerate(documents, 1):
        score = scores[index - 1] if index - 1 < len(scores) else 0.0
        with st.container(border=True):
            source = document.metadata.get("source", "Unknown source")
            chunk_id = document.metadata.get("chunk_id", f"chunk-{index}")
            st.markdown(f"**Evidence {index}: {source}**")
            st.caption(f"{chunk_id} | retrieval score {score:.3f}")
            st.write(document.page_content)


raw_docs, questions, vector_store = get_system_components()
summary = load_summary()

st.sidebar.title("TruthScope controls")
provider_labels = list(PROVIDER_MAP)
configured_provider = {
    "groq": "Groq",
    "opencode_zen": "OpenCode Zen",
    "local_mock": "Local Simulator",
    "gemini": "Google Gemini",
    "openai": "OpenAI",
}.get(LLM_PROVIDER, "Local Simulator")
provider_label = st.sidebar.selectbox(
    "Generator",
    provider_labels,
    index=provider_labels.index(configured_provider),
)
key_settings = {
    "Groq": ("GROQ_API_KEY", "Groq API key"),
    "OpenCode Zen": ("OPENCODE_ZEN_API_KEY", "OpenCode Zen API key"),
    "Google Gemini": ("GEMINI_API_KEY", "Gemini API key"),
    "OpenAI": ("OPENAI_API_KEY", "OpenAI API key"),
}
api_key = ""
if provider_label in key_settings:
    _, label = key_settings[provider_label]
    api_key = st.sidebar.text_input(label, type="password")

top_k = st.sidebar.slider("Evidence chunks", 1, 8, 3)
strict_grounding = st.sidebar.toggle(
    "Strict grounding",
    value=True,
    help="Require the generator to use only retrieved evidence and abstain when it is insufficient.",
)
st.sidebar.caption(f"Index: {RETRIEVAL_MODE} retrieval, threshold {SIMILARITY_THRESHOLD}")

st.title("TruthScope")
st.write("An evidence-first learning tool that shows why an AI answer should be trusted, questioned, or rejected.")

metric_columns = st.columns(4)
metric_columns[0].metric("Knowledge files", len(raw_docs))
metric_columns[1].metric("Benchmark questions", len(questions))
metric_columns[2].metric("Retrieval mode", RETRIEVAL_MODE)
metric_columns[3].metric("Default evidence", f"Top {top_k}")
if summary and summary.get("mock_simulation"):
    st.info("The saved benchmark uses the deterministic local simulator. Treat it as a reproducible software demo, not a real LLM evaluation.")

analyze_tab, benchmark_tab, learn_tab = st.tabs(["Analyze an answer", "Benchmark explorer", "How to verify"])

with analyze_tab:
    preset_questions = ["Custom question"] + [
        f"[{question['id']}] {question['question']}" for question in questions
    ]
    selected_preset = st.selectbox("Benchmark preset", preset_questions)
    if selected_preset == "Custom question":
        default_question = "What is the primary power source for the Curiosity Mars rover?"
        category = "Custom"
        ground_truth = ""
    else:
        question_id = selected_preset.split("]")[0].replace("[", "")
        selected_question = next(question for question in questions if question["id"] == question_id)
        default_question = selected_question["question"]
        category = selected_question["category"]
        ground_truth = selected_question["ground_truth"]

    question = st.text_area("Question", value=default_question, height=90)
    if ground_truth:
        with st.expander("Reference answer for this preset"):
            st.write(ground_truth)

    if st.button("Analyze answer", type="primary"):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            llm = LLMClient(provider=PROVIDER_MAP[provider_label], api_key=api_key or None)
            baseline = BaselinePipeline(llm_client=llm).query(question.strip())
            grounded = RAGPipeline(
                vector_store=vector_store,
                llm_client=llm,
                top_k=top_k,
                strict_grounding=strict_grounding,
            ).query(question.strip(), k=top_k, strict=strict_grounding)
            grounded_analysis = analyze_claims(
                grounded["answer"],
                grounded["retrieved_docs"],
                grounded["retrieval_scores"],
            )
            baseline_analysis = analyze_claims(baseline["answer"])
            risk = assess_risk(
                question,
                grounded["retrieved_docs"],
                grounded["retrieval_scores"],
                category,
            )
            coach = build_learning_coach(grounded_analysis, risk, category)
            grounded_tab, comparison_tab, evidence_tab, coaching_tab = st.tabs([
                "Grounded answer",
                "System comparison",
                "Retrieved evidence",
                "Learning coach",
            ])

            with grounded_tab:
                risk_columns = st.columns(3)
                risk_columns[0].metric("Hallucination risk", f"{risk['level']} ({risk['score']}/100)")
                risk_columns[1].metric("Evidence chunks", risk["retrieved_chunks"])
                risk_columns[2].metric("Best retrieval score", f"{risk['top_score']:.3f}")
                st.subheader("Answer")
                st.write(grounded["answer"])
                with st.expander("Why this risk level?", expanded=risk["level"] == "High"):
                    for reason in risk["reasons"]:
                        st.write(f"- {reason}")
                st.subheader("Claim inspector")
                render_claim_analysis(grounded_analysis)

            with comparison_tab:
                left, right = st.columns(2)
                with left:
                    st.subheader("Baseline without evidence")
                    st.write(baseline["answer"])
                    st.caption(f"Latency {baseline['latency_ms']} ms")
                    render_claim_analysis(baseline_analysis)
                with right:
                    st.subheader("TruthScope grounded answer")
                    st.write(grounded["answer"])
                    st.caption(f"Latency {grounded['latency_ms']} ms")
                    render_claim_analysis(grounded_analysis)

            with evidence_tab:
                render_evidence(grounded["retrieved_docs"], grounded["retrieval_scores"])

            with coaching_tab:
                st.subheader("What just happened")
                st.write(coach["lesson"])
                st.subheader("Try this next")
                st.write(coach["practice"])
                st.info(coach["next_step"])
                if category == "Adversarial_Misconception":
                    st.warning("Misconception detected: check whether the question assumes something false before accepting its yes/no framing.")

with benchmark_tab:
    st.subheader("Saved experiment results")
    if not summary:
        st.info("Run `python run_experiments.py` to generate a benchmark summary.")
    else:
        summary_rows = []
        for system, metrics in summary["systems"].items():
            summary_rows.append({
                "System": system,
                "Hallucination rate": f"{metrics['hallucination_rate_pct']:.1f}%",
                "Faithfulness": f"{metrics['avg_faithfulness_pct']:.1f}%",
                "Accuracy": f"{metrics['accuracy_score_pct']:.1f}%",
                "Token F1": f"{metrics['avg_f1_score']:.2f}",
            })
        st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)
        plot_columns = st.columns(2)
        for column, filename in zip(plot_columns, [
            "hallucination_reduction.png",
            "faithfulness_by_category.png",
        ]):
            plot_path = PLOTS_DIR / filename
            if plot_path.exists():
                column.image(str(plot_path), width="stretch")
    if RESULTS_FILE.exists():
        results = pd.read_csv(RESULTS_FILE)
        categories = st.multiselect("Filter categories", results["category"].unique().tolist(), default=[])
        visible = results if not categories else results[results["category"].isin(categories)]
        columns = [column for column in [
            "id", "category", "question", "baseline_correctness", "rag_k3_correctness",
            "rag_k3_claim_status", "rag_k3_citation_coverage_pct",
        ] if column in visible.columns]
        st.dataframe(visible[columns], width="stretch", hide_index=True)

with learn_tab:
    st.subheader("A practical verification loop")
    steps = [
        "Classify the risk: precise dates, firsts, onlys, and false-premise wording deserve more scrutiny.",
        "Read the answer one claim at a time instead of accepting it as a single block of text.",
        "Open the cited evidence and check whether it supports the complete claim, not just matching words.",
        "Look for corroboration when a decision depends on one source or one retrieval chunk.",
        "Treat abstention as a valid result when the available corpus does not contain the answer.",
    ]
    for index, step in enumerate(steps, 1):
        st.markdown(f"{index}. {step}")
    st.subheader("What the labels mean")
    st.markdown("**Supported:** at least 70% of the claim's key terms are found in the best evidence sentence.")
    st.markdown("**Uncertain:** evidence is related but does not directly verify the complete claim.")
    st.markdown("**Unsupported:** retrieved evidence does not directly support the claim.")
    st.markdown("**Unverified:** no evidence was supplied, as in the baseline response.")
    st.caption("These labels are deterministic retrieval checks, not proof that a claim is true outside the indexed corpus.")
