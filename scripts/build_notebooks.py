import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

def create_notebook(cells, file_path):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Created notebook: {file_path}")

# ==========================================
# Notebook 1: 01_build_rag.ipynb
# ==========================================
nb1_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🚀 Step 1: Building the RAG Pipeline\n",
            "### Project: Evaluating the Impact of RAG on Reducing LLM Hallucinations\n",
            "\n",
            "This notebook walks through:\n",
            "1. Ingesting domain knowledge documents\n",
            "2. Text chunking with sliding window overlap\n",
            "3. Computing neural embeddings (`sentence-transformers/all-MiniLM-L6-v2`)\n",
            "4. Storing vectors and running cosine similarity search"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Install requirements if running in Google Colab\n",
            "# !pip install sentence-transformers langchain chromadb pandas numpy matplotlib seaborn\n",
            "\n",
            "import sys\n",
            "from pathlib import Path\n",
            "\n",
            "# Add parent directory to path\n",
            "sys.path.append('..')\n",
            "\n",
            "from src.data_loader import load_documents, RecursiveCharacterTextSplitter\n",
            "from src.vector_store import SimpleVectorStore, EmbeddingEngine\n",
            "from src.config import DOCUMENTS_DIR, VECTOR_STORE_DIR\n",
            "\n",
            "print(\"Environment initialized successfully.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Load Domain Knowledge Corpus\n",
            "We load 50+ domain documents covering Space Exploration, Astronomy, and Robotic Missions."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "documents = load_documents(DOCUMENTS_DIR)\n",
            "print(f\"Total loaded documents: {len(documents)}\")\n",
            "print(f\"Sample document source: {documents[0].metadata['source']}\")\n",
            "print(\"Sample content snippet:\\n\", documents[0].page_content[:300], \"...\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Chunking the Documents\n",
            "We split long documents into manageable chunks (~400 characters with 60-character overlap) to maintain semantic context across boundaries."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60)\n",
            "chunks = splitter.split_documents(documents)\n",
            "print(f\"Total chunks generated: {len(chunks)}\")\n",
            "print(f\"Sample chunk metadata: {chunks[0].metadata}\")\n",
            "print(f\"Sample chunk text:\\n{chunks[0].page_content}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Embedding Generation & Vector Indexing\n",
            "We embed all chunks and build an in-memory cosine similarity vector index."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "embedding_engine = EmbeddingEngine()\n",
            "vector_store = SimpleVectorStore(embedding_engine=embedding_engine)\n",
            "vector_store.add_documents(chunks)\n",
            "vector_store.save(VECTOR_STORE_DIR)\n",
            "print(f\"Vector store indexed with {len(vector_store.documents)} chunks and saved.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Test Semantic Retrieval\n",
            "Let's test retrieving relevant context for a sample question."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "query = \"What instrument on Perseverance demonstrated producing oxygen from Mars atmosphere?\"\n",
            "results = vector_store.similarity_search_with_score(query, k=3)\n",
            "\n",
            "print(f\"Query: {query}\\n\" + \"=\"*60)\n",
            "for idx, (doc, score) in enumerate(results, 1):\n",
            "    print(f\"\\n[Chunk {idx}] (Similarity Score: {score:.4f}) | Source: {doc.metadata.get('source')}\")\n",
            "    print(doc.page_content)"
        ]
    }
]

create_notebook(nb1_cells, NOTEBOOKS_DIR / "01_build_rag.ipynb")

# ==========================================
# Notebook 2: 02_run_experiments.ipynb
# ==========================================
nb2_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🧪 Step 2: Running the Evaluation Experiments\n",
            "### Project: Evaluating the Impact of RAG on Reducing LLM Hallucinations\n",
            "\n",
            "In this notebook, we run our benchmark suite:\n",
            "1. **System A: Baseline LLM (No RAG)** — query directly without external context\n",
            "2. **System B: RAG Top-3 (Strict)** — retrieve 3 chunks + strict grounding prompt\n",
            "3. **System C: RAG Top-5 (Strict)** — retrieve 5 chunks + strict grounding prompt\n",
            "4. **System D: RAG Top-3 (Loose)** — retrieve 3 chunks with loose prompt (Ablation)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "import pandas as pd\n",
            "from tqdm.auto import tqdm\n",
            "sys.path.append('..')\n",
            "\n",
            "from src.config import QUESTIONS_FILE, RESULTS_FILE, VECTOR_STORE_DIR, LLM_PROVIDER, LLM_MODEL\n",
            "from src.data_loader import load_questions\n",
            "from src.vector_store import SimpleVectorStore, EmbeddingEngine\n",
            "from src.llm_client import LLMClient\n",
            "from src.rag_pipeline import BaselinePipeline, RAGPipeline\n",
            "from src.evaluator import Evaluator\n",
            "\n",
            "print(f\"Using LLM Provider: {LLM_PROVIDER.upper()} ({LLM_MODEL})\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Load Vector Store & Benchmark Questions"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "vector_store = SimpleVectorStore.load(VECTOR_STORE_DIR)\n",
            "questions = load_questions(QUESTIONS_FILE)\n",
            "print(f\"Loaded {len(questions)} evaluation benchmark questions.\")\n",
            "df_q = pd.DataFrame(questions)\n",
            "df_q['category'].value_counts()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Initialize Pipelines"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "llm = LLMClient()\n",
            "baseline_pipe = BaselinePipeline(llm_client=llm)\n",
            "rag_k3_strict = RAGPipeline(vector_store=vector_store, llm_client=llm, top_k=3, strict_grounding=True)\n",
            "rag_k5_strict = RAGPipeline(vector_store=vector_store, llm_client=llm, top_k=5, strict_grounding=True)\n",
            "rag_k3_loose = RAGPipeline(vector_store=vector_store, llm_client=llm, top_k=3, strict_grounding=False)\n",
            "evaluator = Evaluator(llm_client=llm)\n",
            "print(\"Pipelines initialized.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Run Benchmark Loop across All 60 Questions"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "results_data = []\n",
            "\n",
            "for row in tqdm(questions, desc=\"Benchmarking Questions\"):\n",
            "    q_id = row[\"id\"]\n",
            "    category = row[\"category\"]\n",
            "    q_text = row[\"question\"]\n",
            "    gt = row[\"ground_truth\"]\n",
            "    in_corpus = row[\"in_corpus\"].strip().lower() == \"true\"\n",
            "    \n",
            "    # Run Baseline\n",
            "    r_base = baseline_pipe.query(q_text)\n",
            "    e_base = evaluator.evaluate_sample(q_text, r_base[\"answer\"], gt, category, in_corpus)\n",
            "    \n",
            "    # Run RAG Top-3 Strict\n",
            "    r_rag3 = rag_k3_strict.query(q_text)\n",
            "    e_rag3 = evaluator.evaluate_sample(q_text, r_rag3[\"answer\"], gt, category, in_corpus, context=r_rag3[\"retrieved_context\"])\n",
            "    \n",
            "    # Run RAG Top-5 Strict\n",
            "    r_rag5 = rag_k5_strict.query(q_text)\n",
            "    e_rag5 = evaluator.evaluate_sample(q_text, r_rag5[\"answer\"], gt, category, in_corpus, context=r_rag5[\"retrieved_context\"])\n",
            "    \n",
            "    # Run RAG Top-3 Loose\n",
            "    r_ragl = rag_k3_loose.query(q_text)\n",
            "    e_ragl = evaluator.evaluate_sample(q_text, r_ragl[\"answer\"], gt, category, in_corpus, context=r_ragl[\"retrieved_context\"])\n",
            "    \n",
            "    results_data.append({\n",
            "        \"id\": q_id,\n",
            "        \"category\": category,\n",
            "        \"question\": q_text,\n",
            "        \"ground_truth\": gt,\n",
            "        \"in_corpus\": in_corpus,\n",
            "        \"baseline_answer\": r_base[\"answer\"],\n",
            "        \"baseline_hallucinated\": e_base[\"hallucinated\"],\n",
            "        \"baseline_faithfulness\": e_base[\"faithfulness\"],\n",
            "        \"rag_k3_answer\": r_rag3[\"answer\"],\n",
            "        \"rag_k3_hallucinated\": e_rag3[\"hallucinated\"],\n",
            "        \"rag_k3_faithfulness\": e_rag3[\"faithfulness\"],\n",
            "        \"rag_k5_answer\": r_rag5[\"answer\"],\n",
            "        \"rag_k5_hallucinated\": e_rag5[\"hallucinated\"],\n",
            "        \"rag_loose_answer\": r_ragl[\"answer\"],\n",
            "        \"rag_loose_hallucinated\": e_ragl[\"hallucinated\"]\n",
            "    })\n",
            "\n",
            "df_res = pd.DataFrame(results_data)\n",
            "df_res.to_csv(RESULTS_FILE, index=False)\n",
            "print(f\"Experiments complete! Saved {len(df_res)} records to {RESULTS_FILE}\")"
        ]
    }
]

create_notebook(nb2_cells, NOTEBOOKS_DIR / "02_run_experiments.ipynb")

# ==========================================
# Notebook 3: 03_evaluation.ipynb
# ==========================================
nb3_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 📊 Step 3: Evaluation, Metrics & Error Analysis\n",
            "### Project: Evaluating the Impact of RAG on Reducing LLM Hallucinations\n",
            "\n",
            "In this notebook, we analyze:\n",
            "1. **Aggregate Metric Comparisons** (Hallucination Rate, Faithfulness, Factual Accuracy)\n",
            "2. **Category Breakdown** (Direct Fact vs Multi-Hop vs Out-of-Corpus vs Adversarial)\n",
            "3. **Top-K & Prompt Strictness Ablations**\n",
            "4. **Qualitative Case Studies & Failure Modes**"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "sys.path.append('..')\n",
            "\n",
            "from src.config import RESULTS_FILE, PLOTS_DIR\n",
            "from src.visualization import plot_hallucination_comparison, plot_category_breakdown, plot_ablation_comparison\n",
            "\n",
            "df = pd.read_csv(RESULTS_FILE)\n",
            "print(f\"Loaded {len(df)} experiment rows.\")\n",
            "df.head(3)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Key Metric Calculations"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "total = len(df)\n",
            "summary = {\n",
            "    \"Baseline (No RAG)\": {\n",
            "        \"Hallucination Rate\": f\"{(df['baseline_hallucinated'].sum()/total)*100:.1f}%\",\n",
            "        \"Faithfulness\": f\"{df['baseline_faithfulness'].mean()*100:.1f}%\"\n",
            "    },\n",
            "    \"RAG (Top-3 Strict)\": {\n",
            "        \"Hallucination Rate\": f\"{(df['rag_k3_hallucinated'].sum()/total)*100:.1f}%\",\n",
            "        \"Faithfulness\": f\"{df['rag_k3_faithfulness'].mean()*100:.1f}%\"\n",
            "    },\n",
            "    \"RAG (Top-5 Strict)\": {\n",
            "        \"Hallucination Rate\": f\"{(df['rag_k5_hallucinated'].sum()/total)*100:.1f}%\n",
            "    },\n",
            "    \"RAG (Top-3 Loose)\": {\n",
            "        \"Hallucination Rate\": f\"{(df['rag_loose_hallucinated'].sum()/total)*100:.1f}%\n",
            "    }\n",
            "}\n",
            "pd.DataFrame(summary).T"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Visualizing Benchmark Plots"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import matplotlib.image as mpimg\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(10, 6))\n",
            "img = mpimg.imread(str(PLOTS_DIR / 'hallucination_reduction.png'))\n",
            "ax.imshow(img)\n",
            "ax.axis('off')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Qualitative Failure Case Analysis\n",
            "Let's inspect questions where Baseline hallucinated vs how RAG resolved or handled it."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "halluc_cases = df[df['baseline_hallucinated'] == 1][['id', 'category', 'question', 'baseline_answer', 'rag_k3_answer', 'ground_truth']].head(5)\n",
            "for _, r in halluc_cases.iterrows():\n",
            "    print(f\"\\n[{r['id']}] Category: {r['category']}\")\n",
            "    print(f\"Question: {r['question']}\")\n",
            "    print(f\"🔴 Baseline: {r['baseline_answer']}\")\n",
            "    print(f\"🟢 RAG: {r['rag_k3_answer']}\")\n",
            "    print(f\"🎯 Ground Truth: {r['ground_truth']}\")\n",
            "    print(\"-\"*60)"
        ]
    }
]

create_notebook(nb3_cells, NOTEBOOKS_DIR / "03_evaluation.ipynb")
