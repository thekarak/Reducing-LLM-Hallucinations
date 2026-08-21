import os
import json
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

from src.config import (
    DOCUMENTS_DIR, QUESTIONS_FILE, RESULTS_FILE, SUMMARY_FILE, PLOTS_DIR,
    VECTOR_STORE_DIR, LLM_PROVIDER, CHUNK_SIZE, CHUNK_OVERLAP,
    SIMILARITY_THRESHOLD, RETRIEVAL_MODE
)
from src.data_loader import load_documents, load_questions, RecursiveCharacterTextSplitter
from src.vector_store import SimpleVectorStore, EmbeddingEngine
from src.llm_client import LLMClient
from src.rag_pipeline import BaselinePipeline, RAGPipeline
from src.evaluator import Evaluator

st.set_page_config(
    page_title="RAG Hallucination Evaluation Suite",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean, professional research theme
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #3498db;
        margin-bottom: 12px;
    }
    .badge-hallucinated {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85em;
    }
    .badge-grounded {
        background-color: #dcfce7;
        color: #15803d;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85em;
    }
    .badge-partial {
        background-color: #fef3c7;
        color: #b45309;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85em;
    }
    .chunk-box {
        background-color: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        padding: 12px;
        margin-bottom: 8px;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_system_components():
    """Load or initialize vector store, raw docs, and question set."""
    raw_docs = load_documents(DOCUMENTS_DIR)
    questions = load_questions(QUESTIONS_FILE)

    # Initialize Vector Store
    embedding_engine = EmbeddingEngine()
    if (VECTOR_STORE_DIR / "vectors.npy").exists():
        vector_store = SimpleVectorStore.load(
            VECTOR_STORE_DIR, embedding_engine=embedding_engine,
            similarity_threshold=SIMILARITY_THRESHOLD, retrieval_mode=RETRIEVAL_MODE
        )
    else:
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_documents(raw_docs)
        vector_store = SimpleVectorStore(
            embedding_engine=embedding_engine,
            similarity_threshold=SIMILARITY_THRESHOLD,
            retrieval_mode=RETRIEVAL_MODE
        )
        vector_store.add_documents(chunks)
        vector_store.save(VECTOR_STORE_DIR)

    return raw_docs, questions, vector_store


@st.cache_data
def load_summary():
    """Load aggregate benchmark metrics computed by run_experiments.py."""
    if SUMMARY_FILE.exists():
        with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    if RESULTS_FILE.exists():
        df = pd.read_csv(RESULTS_FILE)

        def agg(prefix):
            n = len(df)
            return {
                "hallucination_rate_pct": round(df[f"{prefix}_hallucinated"].mean() * 100, 2),
                "avg_faithfulness_pct": round(df[f"{prefix}_faithfulness"].mean() * 100, 2),
                "accuracy_score_pct": round(
                    ((df[f"{prefix}_correct_score"]).mean()) * 100, 2
                ),
                "avg_f1_score": round(df[f"{prefix}_f1"].mean(), 4),
            }

        return {
            "total_questions": len(df),
            "systems": {
                "Baseline (No RAG)": agg("baseline"),
                "RAG (Top-3 Strict)": agg("rag_k3"),
                "RAG (Top-5 Strict)": agg("rag_k5"),
                "RAG (Top-3 Loose)": agg("rag_loose"),
            },
        }
    return None

raw_docs, questions_list, vector_store = get_system_components()
summary = load_summary()

# Sidebar Configuration
st.sidebar.title("⚙️ Experiment Controls")
provider_options = ["Groq Cloud", "OpenCode Zen", "Local Mock (Deterministic)", "Google Gemini", "OpenAI"]
default_provider = {
    "groq": "Groq Cloud",
    "opencode_zen": "OpenCode Zen",
    "local_mock": "Local Mock (Deterministic)",
    "gemini": "Google Gemini",
    "openai": "OpenAI",
}.get(LLM_PROVIDER, "Groq Cloud")
provider = st.sidebar.selectbox("LLM Provider", provider_options, index=provider_options.index(default_provider))

provider_map = {
    "OpenCode Zen": "opencode_zen",
    "Local Mock (Deterministic)": "local_mock",
    "Groq Cloud": "groq",
    "Google Gemini": "gemini",
    "OpenAI": "openai"
}

# Provider Specific Configurations
api_key_input = None
if provider == "OpenCode Zen":
    api_key_input = st.sidebar.text_input(
        "OpenCode Zen API Key",
        value=os.getenv("OPENCODE_ZEN_API_KEY", ""),
        type="password",
        help="Enter your OpenCode Zen API key"
    )
    base_url_input = st.sidebar.text_input(
        "OpenCode Zen Base URL",
        value=os.getenv("OPENCODE_ZEN_BASE_URL", "https://api.opencodezen.com/v1"),
        help="API Base URL for OpenCode Zen endpoint"
    )
    if api_key_input:
        os.environ["OPENCODE_ZEN_API_KEY"] = api_key_input
    if base_url_input:
        os.environ["OPENCODE_ZEN_BASE_URL"] = base_url_input
elif provider == "Groq Cloud":
    api_key_input = st.sidebar.text_input("Groq API Key", value=os.getenv("GROQ_API_KEY", ""), type="password", help="Enter your free Groq API key")
    if api_key_input:
        os.environ["GROQ_API_KEY"] = api_key_input
elif provider == "Google Gemini":
    api_key_input = st.sidebar.text_input("Gemini API Key", value=os.getenv("GEMINI_API_KEY", ""), type="password")
    if api_key_input:
        os.environ["GEMINI_API_KEY"] = api_key_input
elif provider == "OpenAI":
    api_key_input = st.sidebar.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password")
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input

top_k_select = st.sidebar.slider("Top-K Retrieved Chunks", min_value=1, max_value=8, value=3)
strict_mode = st.sidebar.checkbox("Strict Refusal Grounding Prompt", value=True, help="Instructs model to answer ONLY from context or declare 'I don't know'")

# Header
st.title("🔬 Evaluating RAG's Impact on Reducing LLM Hallucinations")
st.markdown(
    "An empirical research testbench measuring factual accuracy, context faithfulness, "
    "and hallucination rates across **Baseline LLM**, **RAG (Top-3/5)**, and prompt grounding variants."
)

# Top KPI Summary Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Knowledge Documents", value=f"{len(raw_docs)} files", delta="Indexed")
with col2:
    st.metric(label="Evaluation Questions", value=f"{len(questions_list)} queries", delta="4 Categories")
if summary:
    systems = summary["systems"]
    base_hr = systems["Baseline (No RAG)"]["hallucination_rate_pct"]
    rag_hr = systems["RAG (Top-3 Strict)"]["hallucination_rate_pct"]
    base_faith = systems["Baseline (No RAG)"]["avg_faithfulness_pct"]
    rag_faith = systems["RAG (Top-3 Strict)"]["avg_faithfulness_pct"]
    with col3:
        st.metric(label="Baseline Hallucination Rate", value=f"{base_hr:.1f}%",
                  delta=f"{rag_hr - base_hr:+.1f} pts with RAG", delta_color="inverse")
    with col4:
        st.metric(label="RAG (Top-3) Faithfulness", value=f"{rag_faith:.1f}%",
                  delta=f"{rag_faith - base_faith:+.1f} pts", delta_color="normal")
else:
    with col3:
        st.metric(label="Baseline Hallucination Rate", value="Run benchmark")
    with col4:
        st.metric(label="RAG (Top-3) Faithfulness", value="Run benchmark")

# Tabs
tab1, tab2, tab3 = st.tabs(["🧪 Interactive Test Bench", "📊 Benchmark Dashboard & Plots", "🔎 Qualitative Failure Analysis"])

# TAB 1: Interactive Test Bench
with tab1:
    st.subheader("Side-by-Side Playground: Baseline LLM vs Grounded RAG")
    st.markdown("Test individual benchmark questions or enter your own custom query to observe hallucination reduction.")

    preset_questions = ["-- Select a benchmark sample question --"] + [f"[{q['id']}] ({q['category']}) {q['question']}" for q in questions_list]
    selected_preset = st.selectbox("Choose a Sample Question from Benchmark:", preset_questions)

    if selected_preset != "-- Select a benchmark sample question --":
        q_id = selected_preset.split("]")[0].replace("[", "")
        selected_q_obj = next((q for q in questions_list if q["id"] == q_id), None)
        default_prompt = selected_q_obj["question"] if selected_q_obj else ""
        expected_gt = selected_q_obj["ground_truth"] if selected_q_obj else ""
        cat_name = selected_q_obj["category"] if selected_q_obj else ""
        is_in_corpus = selected_q_obj["in_corpus"].strip().lower() == "true" if selected_q_obj else True
    else:
        default_prompt = "What is the primary power source for the Curiosity Mars rover?"
        expected_gt = "A Multi-Mission Radioisotope Thermoelectric Generator (MMRTG) using plutonium-238 dioxide."
        cat_name = "Direct_Fact"
        is_in_corpus = True

    user_query = st.text_area("Input Query / Question:", value=default_prompt, height=70)

    if st.button("🚀 Run Dual System Evaluation", type="primary"):
        llm = LLMClient(provider=provider_map[provider])
        baseline_pipe = BaselinePipeline(llm_client=llm)
        rag_pipe = RAGPipeline(vector_store=vector_store, llm_client=llm, top_k=top_k_select, strict_grounding=strict_mode)
        evaluator = Evaluator(llm_client=llm)

        with st.spinner("Generating and evaluating responses..."):
            # Query Baseline
            res_baseline = baseline_pipe.query(user_query)
            eval_base = evaluator.evaluate_sample(
                question=user_query,
                answer=res_baseline["answer"],
                ground_truth=expected_gt,
                category=cat_name,
                in_corpus=is_in_corpus,
                context=""
            )

            # Query RAG
            res_rag = rag_pipe.query(user_query, k=top_k_select, strict=strict_mode)
            eval_rag = evaluator.evaluate_sample(
                question=user_query,
                answer=res_rag["answer"],
                ground_truth=expected_gt,
                category=cat_name,
                in_corpus=is_in_corpus,
                context=res_rag["retrieved_context"]
            )

        # Ground Truth Box
        st.info(f"**Ground Truth Reference:** {expected_gt}")

        col_left, col_right = st.columns(2)

        # Left Column: Baseline LLM
        with col_left:
            st.markdown("### 🔴 System A: Baseline LLM (No RAG)")
            st.markdown(f"**Generated Response:**\n\n> {res_baseline['answer']}")
            
            # Badge
            if eval_base["hallucinated"]:
                st.markdown('<span class="badge-hallucinated">⚠️ Hallucination Detected</span>', unsafe_allow_html=True)
            elif eval_base["factual_correctness"] == "Correct":
                st.markdown('<span class="badge-grounded">✅ Factually Accurate</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-partial">⚠️ Partially Correct</span>', unsafe_allow_html=True)

            st.write(f"- **Factual Correctness:** {eval_base['factual_correctness']}")
            st.write(f"- **Faithfulness Score:** {eval_base['faithfulness'] * 100:.1f}%")
            st.write(f"- **Token F1 Match:** {eval_base['f1_score']:.2f}")
            st.write(f"- **Inference Latency:** {res_baseline['latency_ms']} ms")

        # Right Column: RAG Pipeline
        with col_right:
            st.markdown(f"### 🟢 System B: Grounded RAG (Top-{top_k_select})")
            st.markdown(f"**Generated Response:**\n\n> {res_rag['answer']}")
            
            # Badge
            if eval_rag["hallucinated"]:
                st.markdown('<span class="badge-hallucinated">⚠️ Hallucination Detected</span>', unsafe_allow_html=True)
            elif eval_rag["factual_correctness"] == "Correct":
                st.markdown('<span class="badge-grounded">✅ Grounded & Factually Accurate</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-partial">⚠️ Partially Correct</span>', unsafe_allow_html=True)

            st.write(f"- **Factual Correctness:** {eval_rag['factual_correctness']}")
            st.write(f"- **Faithfulness Score:** {eval_rag['faithfulness'] * 100:.1f}%")
            st.write(f"- **Token F1 Match:** {eval_rag['f1_score']:.2f}")
            st.write(f"- **Inference Latency:** {res_rag['latency_ms']} ms")

        # Retrieved Context Accordion
        with st.expander(f"🔍 Inspect {len(res_rag['retrieved_docs'])} Retrieved Knowledge Chunks", expanded=True):
            if not res_rag["retrieved_docs"]:
                st.warning("No chunks passed the similarity threshold — a strict RAG system should refuse to answer here.")
            for idx, doc in enumerate(res_rag["retrieved_docs"], 1):
                src = doc.metadata.get("source", "doc")
                chunk_id = doc.metadata.get("chunk_id", f"c{idx}")
                score = res_rag["retrieval_scores"][idx - 1] if idx - 1 < len(res_rag["retrieval_scores"]) else None
                score_badge = f"cosine ≈ {score:.2f}" if score is not None else ""
                st.markdown(f"""
                <div class="chunk-box">
                    <strong>Chunk [{idx}] — Source: <code>{src}</code> (ID: {chunk_id}) {score_badge}</strong><br>
                    {doc.page_content}
                </div>
                """, unsafe_allow_html=True)

# TAB 2: Benchmark Dashboard & Plots
with tab2:
    st.subheader("📊 Quantitative Benchmark Results & Statistical Visualizations")

    if RESULTS_FILE.exists():
        df_results = pd.read_csv(RESULTS_FILE)
        st.markdown(f"Displaying evaluation results across **{len(df_results)} curated questions**.")

        # Display Comparative Figures
        st.markdown("#### Publication-Quality Comparative Plots")
        fig_col1, fig_col2 = st.columns(2)
        
        plot1_path = PLOTS_DIR / "hallucination_reduction.png"
        plot2_path = PLOTS_DIR / "faithfulness_by_category.png"
        plot3_path = PLOTS_DIR / "top_k_ablation.png"

        if plot1_path.exists():
            fig_col1.image(str(plot1_path), caption="Figure 1: Hallucination Rate, Faithfulness, and Accuracy Comparison")
        if plot2_path.exists():
            fig_col2.image(str(plot2_path), caption="Figure 2: Breakdown by Query Category (Fact, Multi-Hop, Out-of-Corpus, Adversarial)")
        if plot3_path.exists():
            st.image(str(plot3_path), caption="Figure 3: Ablation Study: Top-K Context Depth vs Grounding Prompt Strictness", use_container_width=True)

        # Summary Table (computed from results, never hardcoded)
        st.markdown("#### Benchmark Aggregate Comparison Table")
        if summary:
            rows = []
            for name, m in summary["systems"].items():
                rows.append({
                    "System Setting": name,
                    "Hallucination Rate": f"{m['hallucination_rate_pct']:.1f}%",
                    "Faithfulness Score": f"{m['avg_faithfulness_pct']:.1f}%",
                    "Factual Accuracy": f"{m['accuracy_score_pct']:.1f}%",
                    "Token F1": f"{m['avg_f1_score']:.2f}",
                })
            st.table(pd.DataFrame(rows))
            st.caption(
                f"Computed from {summary.get('total_questions', len(df_results))} questions · "
                f"provider: `{summary.get('provider', 'n/a')}` · chunk size {summary.get('chunk_size', 'n/a')} / "
                f"overlap {summary.get('chunk_overlap', 'n/a')} · retrieval: {summary.get('retrieval_mode', 'n/a')} "
                f"(threshold {summary.get('similarity_threshold', 'n/a')})"
            )
        else:
            st.info("Run `python run_experiments.py` to compute aggregate metrics.")

        # Filterable Data Table
        st.markdown("#### Explore Raw Question-by-Question Evaluation Records")
        cat_filter = st.multiselect("Filter by Category", options=df_results["category"].unique().tolist(), default=df_results["category"].unique().tolist())
        show_halluc_only = st.checkbox("Show Only Questions where Baseline Hallucinated")

        filtered_df = df_results[df_results["category"].isin(cat_filter)]
        if show_halluc_only:
            filtered_df = filtered_df[filtered_df["baseline_hallucinated"] == 1]

        cols_to_show = ["id", "category", "question", "ground_truth", "baseline_answer", "baseline_correctness", "rag_k3_answer", "rag_k3_correctness"]
        st.dataframe(filtered_df[cols_to_show], use_container_width=True)
    else:
        st.warning("Results CSV not found yet. Run `python run_experiments.py` to generate the complete dataset.")

# TAB 3: Qualitative Failure Analysis
with tab3:
    st.subheader("🔎 Qualitative Analysis: When & Why Does RAG Still Hallucinate?")
    st.markdown("""
    RAG substantially reduces hallucinations on this benchmark, but it is not immune. Residual failures fall into four mechanisms:
    """)

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.markdown("""
        #### 1. Retrieval Miss / Semantic Drift
        * **Mechanism**: The retriever returns semantically adjacent but non-pertinent documents (e.g. retrieving Apollo 11 text when asked about Apollo 13 liquid nitrogen).
        * **Result**: If the model has weak instruction following, it falls back to parametric training priors and hallucinates.
        * **Mitigation**: Implement re-ranking (e.g., cross-encoders or Cohere rerank) and strict negative rejection thresholds.

        #### 2. Extrapolation Beyond Context (Loose Prompts)
        * **Mechanism**: When prompt instructions do not strictly enforce *'answer only from context'*, the model blends facts from context with ungrounded speculation.
        * **Result**: Hallucinations rise noticeably in the loose-grounding ablation (see the comparison table above for the exact gap).
        * **Mitigation**: Explicit negative constraint prompting: *"If not present in context, state: I do not have enough information."*
        """)

    with col_f2:
        st.markdown("""
        #### 3. Adversarial Premise Affirmation (Sycophancy)
        * **Mechanism**: The user's query embeds a false factual premise (e.g., *"Did DART capture Dimorphos and bring it to Earth orbit?"*).
        * **Result**: The LLM instinctively affirms the query structure unless the retrieved context contains direct counter-evidence.
        * **Mitigation**: Chain-of-thought verification step or self-contradiction checking before generation.

        #### 4. Multi-Document Entity Confusion
        * **Mechanism**: In multi-hop synthesis queries requiring facts from multiple distinct chunks (e.g. comparing power sources of Curiosity vs Juno), the model may swap entity attributes.
        * **Result**: Partial hallucination (e.g., attributing Juno's solar panels to Curiosity).
        * **Mitigation**: Structured attribute extraction or hierarchical retrieval schemas.
        """)

    st.markdown("---")
    st.markdown("### 🏆 4 Exemplary Case Studies")
    
    st.markdown("""
    | Case | Question | Baseline (Hallucinated) | RAG (Grounded Fix) | Ground Truth |
    |---|---|---|---|---|
    | **Case 1: Unanswerable Trap** | *What was the battery capacity of the fictional Apollo 18 Mars module?* | Invented *"64 kWh silver-zinc secondary pack"* | *"I do not have enough information in the provided context to answer this question."* | Out of corpus / Fictional |
    | **Case 2: Misconception Trap** | *Did the 1976 Viking experiments definitively prove life on Mars?* | *"Yes, Viking confirmed life on Mars via the Labeled Release test."* | *"No, the GCMS found no organic compounds; scientists attributed it to non-biological oxidants."* | Unproven / Non-biological |
    | **Case 3: Numeric Metric** | *What total mass of asteroid sample did OSIRIS-REx return?* | *"Approximately 2.5 kilograms."* | *"121.6 grams (4.29 oz)."* | 121.6 grams |
    | **Case 4: Space Telescope Specs** | *What is the diameter and material of JWST's primary mirror?* | *"8.0-meter polished aluminum array."* | *"6.5 meters in diameter, consisting of 18 hexagonal beryllium segments coated with gold."* | 6.5m beryllium/gold |
    """)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip**: Set `LLM_PROVIDER=opencode_zen` or `LLM_PROVIDER=groq` in `.env` to test live API models.")
