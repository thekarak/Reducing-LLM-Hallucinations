# 🔬 Evaluating the Impact of Retrieval-Augmented Generation (RAG) on Reducing LLM Hallucinations

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit Demo](https://img.shields.io/badge/Demo-Streamlit-FF4B4B.svg)](app.py)
[![Colab Ready](https://img.shields.io/badge/Google%20Colab-Ready-orange.svg)](notebooks/)
[![Author](https://img.shields.io/badge/Author-Sourasis%20Karak-blueviolet.svg)](https://github.com/thekarak)

An empirical research study and evaluation benchmark where I quantitatively measure and analyze how **Retrieval-Augmented Generation (RAG)** mitigates factual hallucinations, enforces contextual faithfulness, and improves uncertainty handling compared to standard unaugmented Large Language Models.

---

## 📌 Table of Contents
1. [Why I Built This (Motivation & Background)](#-why-i-built-this-motivation--background)
2. [Core Architecture & How My System Works](#-core-architecture--how-my-system-works)
3. [Tech Stack](#-tech-stack)
4. [My Evaluation Benchmark (60 Questions, 4 Categories)](#-my-evaluation-benchmark)
5. [Key Empirical Results](#-key-empirical-results)
6. [Visualizations & Plots](#-visualizations--plots)
7. [Qualitative Analysis & Case Studies](#-qualitative-analysis--case-studies)
8. [Failure Modes: When & Why RAG Still Fails](#-failure-modes-when--why-rag-still-fails)
9. [Project Structure](#-project-structure)
10. [How to Run (CLI, Web App & Colab)](#-how-to-run-cli-web-app--colab)
11. [Supported LLM Providers](#-supported-llm-providers)
12. [Next Steps & Future Work](#-next-steps--future-work)

---

## 💡 Why I Built This (Motivation & Background)

As LLMs have evolved, their fluency has often outpaced their factual grounding. When deployed in production environments, standard unaugmented models frequently suffer from:
1. **Parametric Memorization Decay**: Confabulating fine-grained historical dates, numerical values, and technical specifications.
2. **Epistemic Overconfidence**: Making up answers to unanswerable or fictional questions rather than simply admitting ignorance.
3. **Adversarial Sycophancy**: Falsely agreeing with leading user questions that embed incorrect factual premises.

I designed and built this project to answer three practical engineering questions with real numbers:
* *How much does dense semantic grounding actually reduce hallucination rates compared to plain LLM generation?*
* *Does increasing retrieval context depth ($k=3$ vs $k=5$) improve multi-hop reasoning or just add noise?*
* *How critical is prompt grounding strictness (negative constraint phrasing) in preventing context extrapolation?*

---

## ⚙️ Core Architecture & How My System Works

I structured the pipeline into five distinct stages:

```
[51 Knowledge Documents] ──► [Recursive Chunker (1,240 Chunks)] ──► [Dense Embeddings (MiniLM)] ──► [Persistent Vector Store]
                                                                                                          │
[60 Evaluation Queries] ──────────────────────────────────────────────────────────────────────────────────┼──► [System A: Baseline LLM]
        │                                                                                                 │
        └──────► [Cosine Similarity Retrieval (Top-K)] ──► [Strict Grounding Prompt] ─────────────────────┴──► [System B: Grounded RAG]
                                                                                                                    │
                                                                                                                    ▼
                                                                                   [Automated Multi-Metric Evaluation Engine]
                                                                                   (Faithfulness, Hallucination, F1, Accuracy)
```

### 1. Domain Corpus & Semantic Chunking
* Curated **51 domain documents** spanning deep-space missions, Mars exploration rovers, space telescopes, and astrobiology.
* Implemented a `RecursiveCharacterTextSplitter` with a chunk size of **400 characters** and **60 characters overlap** ($L=400, O=60$). This produced **1,240 semantic chunks**, ensuring complete thoughts and numerical figures were not split mid-sentence.

### 2. Dense Embeddings & Vector Indexing
* Generated 384-dimensional dense vector embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
* Built an in-memory normalized cosine similarity search engine with disk serialization for instant warm starts.

### 3. Dual-System Pipeline Implementation
I evaluated every query across 4 distinct configurations:
* **System A (Baseline LLM - No RAG)**: Direct zero-shot query passing to the model without external context.
* **System B (RAG Top-3 Strict Grounding)**: Retrieves top-3 chunks; prompt mandates: *"Answer strictly and ONLY using the provided context. If the answer is not present, state 'I do not have enough information'."*
* **System C (RAG Top-5 Strict Grounding)**: Retrieves top-5 chunks to test expanded multi-hop context depth.
* **System D (RAG Top-3 Loose Grounding - Ablation)**: Retrieves top-3 chunks with a permissive prompt that allows external knowledge extrapolation.

### 4. Multi-Metric Evaluation Engine
To ensure unbiased evaluation, each generated answer is scored against ground truth across 5 automated metrics:
* **Hallucination Flag**: Binary flag indicating whether the output contains unsupported or fabricated claims.
* **Context Faithfulness Score**: Proportion of statements in the answer directly verifiable from the retrieved context.
* **Factual Correctness**: Multi-class metric (*Correct*, *Partially Correct*, *Incorrect*).
* **Token F1 Score**: Word overlap precision and recall against verified ground truth.
* **Refusal Precision**: Checks whether out-of-corpus queries were correctly refused without fabrication.

---

## 🛠️ Tech Stack

| Component | Tool / Library | Rationale |
|---|---|---|
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Fast, lightweight 384-dim dense embeddings with high semantic fidelity. |
| **Vector Store** | Custom Cosine Vector Index with Disk Persistence | Zero-dependency, portable, deterministic, and easily swappable with FAISS or Chroma. |
| **LLMs Supported** | **OpenCode Zen**, **Groq (Llama-3.1-8B/70B)**, **Gemini 1.5**, **OpenAI**, **Local Mock** | Seamless multi-provider support with zero-cost offline reproduction out of the box. |
| **Data Processing** | `pandas`, `numpy`, `tqdm` | High-performance batch evaluation and structured tabular dataset management. |
| **Visualizations** | `matplotlib`, `seaborn` | Generates 300 DPI publication-grade comparison plots. |
| **Demo Application** | `streamlit` | Interactive side-by-side testbench, live chunk inspector, and analytics explorer. |

---

## 🧪 My Evaluation Benchmark

I created a custom dataset of **60 evaluation questions** categorized to test different reasoning capabilities and failure modes:

| Category | Count | What It Tests | Example Query |
|---|:---:|---|---|
| **Direct Fact Extraction** | 18 | Exact dates, specs, instruments, budgets | *"What is the primary power source for the Curiosity Mars rover?"* |
| **Multi-Hop Synthesis** | 12 | Synthesizing facts across multiple chunks | *"Compare the power sources of the Curiosity rover, Juno spacecraft, and Voyager 1 probe."* |
| **Out-of-Corpus (Traps)** | 15 | Fictional or unrecorded facts (Refusal testing) | *"What was the battery capacity of the fictional Apollo 18 Mars module?"* |
| **Adversarial Misconceptions** | 15 | Queries embedding false factual premises | *"Did the Viking biological experiments in 1976 definitively prove life on Mars?"* |

---

## 📊 Key Empirical Results

Here is what the numbers showed across all 60 benchmark questions:

| System Configuration | Hallucination Rate ↓ | Context Faithfulness ↑ | Factual Accuracy ↑ | Token F1 Match ↑ |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline LLM (No RAG)** | **83.3%** | 10.0% | 14.2% | 0.22 |
| **RAG (Top-3 Strict Grounding)** | **3.3%** | **69.8%** | **96.7%** | **0.86** |
| **RAG (Top-5 Strict Grounding)** | **3.3%** | **72.8%** | **96.7%** | **0.88** |
| **RAG (Top-3 Loose Grounding)** *(Ablation)* | **28.3%** | 44.8% | 71.7% | 0.65 |

### 🚀 Key Takeaways from My Experiments:
1. **96.0% Relative Reduction in Hallucinations**: Grounding with strict RAG caused hallucination rates to plunge from **83.3% to 3.3%**.
2. **Prompt Strictness is as Important as Retrieval**: When I relaxed the prompt instructions (*"Loose Grounding"*), hallucinations jumped from **3.3% to 28.3%**, showing that LLMs will extrapolate if not explicitly restricted to context.
3. **Top-5 vs Top-3 Context Depth**: Expanding retrieval depth to $k=5$ boosted multi-hop synthesis faithfulness from **69.8% to 72.8%** without introducing context distraction.

---

## 📈 Visualizations & Plots

The benchmark exports 3 publication-ready figures located in [`results/plots/`](file:///c:/Users/sfors/OneDrive/Desktop/Project%20Ze/results/plots):

| Figure | Description |
|---|---|
| **`hallucination_reduction.png`** | Direct comparison of Hallucination Rate, Faithfulness, and Accuracy across all 4 system configurations. |
| **`faithfulness_by_category.png`** | Granular performance breakdown across Direct Fact, Multi-Hop, Out-of-Corpus, and Adversarial categories. |
| **`top_k_ablation.png`** | Ablation study analyzing context depth ($k=3$ vs $k=5$) and prompt instruction strictness. |

---

## 🔬 Qualitative Analysis & Case Studies

Here are 3 representative examples illustrating how RAG corrected baseline failures:

### Case 1: Out-of-Corpus Trap (Apollo 18 Mars Module)
* **Question**: *"What was the exact battery capacity in kilowatt-hours of the fictional Apollo 18 landing module on Mars?"*
* 🔴 **Baseline LLM**: *"The Apollo 18 Mars landing module was equipped with an experimental 64 kilowatt-hour silver-zinc secondary battery pack..."* *(Completely fabricated)*
* 🟢 **RAG (Top-3 Strict)**: *"I do not have enough information in the provided context to answer this question."* *(Correct uncertainty calibration)*

### Case 2: Adversarial Misconception (Viking Life Detection)
* **Question**: *"Did the Viking biological experiments in 1976 definitively prove the existence of living microorganisms on Mars?"*
* 🔴 **Baseline LLM**: *"Yes, the Viking 1 and 2 landers confirmed life on Mars in 1976 when the Labeled Release experiment detected metabolic respiration."* *(Affirmed false premise)*
* 🟢 **RAG (Top-3 Strict)**: *"No. While the Labeled Release test gave a positive signal, the GCMS found no organics, and scientists attributed the reaction to soil oxidants like perchlorates."* *(Grounded rejection)*

### Case 3: Exact Numeric Specification (OSIRIS-REx Sample Mass)
* **Question**: *"What total mass of asteroid sample did OSIRIS-REx deliver back to Earth from asteroid Bennu?"*
* 🔴 **Baseline LLM**: *"OSIRIS-REx returned approximately 2.5 kilograms of Bennu regolith."* *(Inaccurate estimation)*
* 🟢 **RAG (Top-3 Strict)**: *"OSIRIS-REx delivered 121.6 grams (4.29 oz) of asteroid sample from Bennu."* *(Exact match)*

---

## ⚠️ Failure Modes: When & Why RAG Still Fails

In my analysis, I categorized residual RAG errors into four primary mechanisms:
1. **Semantic Chunk Fragmentation**: When an answer requires synthesizing facts split across chunk boundaries, standard top-3 retrieval may retrieve one chunk and omit the other.
2. **Context Extrapolation**: Without explicit negative constraints (*"Answer ONLY from context"*), models blend retrieved text with ungrounded pretraining priors.
3. **Retrieval Semantic Drift**: Queries with vocabulary mismatch or low keyword overlap can retrieve adjacent but non-pertinent chunks.
4. **Entity Swapping in Multi-Hop Queries**: Complex queries comparing multiple entities may lead to attribute confusion if the context is dense.

---

## 📂 Project Structure

```
Reducing-LLM-Hallucinations/
├── README.md                          # Comprehensive research report & user guide
├── requirements.txt                   # Core Python dependencies
├── .env.example                       # API key configuration template
├── run_experiments.py                 # Master CLI experiment & benchmark runner
├── app.py                             # Interactive Streamlit Demo Application
├── data/
│   ├── documents/                     # 51 curated domain knowledge documents
│   └── questions.csv                  # 60 benchmark questions across 4 categories
├── src/
│   ├── __init__.py
│   ├── config.py                      # Paths, parameters, and model options
│   ├── data_loader.py                 # Document loader & recursive character chunker
│   ├── vector_store.py                # Embedding engine & persistent vector store
│   ├── llm_client.py                  # Multi-provider client (OpenCode Zen, Groq, Gemini, OpenAI, Mock)
│   ├── rag_pipeline.py                # Baseline LLM vs Grounded RAG pipelines
│   ├── evaluator.py                   # Automated Faithfulness & Hallucination metrics
│   └── visualization.py               # Generates 300 DPI publication plots
├── notebooks/
│   ├── 01_build_rag.ipynb             # Step 1: Chunking, Embeddings, & Retrieval
│   ├── 02_run_experiments.ipynb       # Step 2: Executing benchmark across 60 questions
│   └── 03_evaluation.ipynb           # Step 3: Metric analysis & error case studies
└── results/
    ├── results.csv                    # Complete 60-row evaluation dataset
    └── plots/                         # Generated comparison figures (.png)
```

---

## 🚀 How to Run (CLI, Web App & Colab)

### 1. Installation
```bash
git clone https://github.com/thekarak/Reducing-LLM-Hallucinations.git
cd Reducing-LLM-Hallucinations
pip install -r requirements.txt
```

### 2. Run the Benchmark Suite via CLI
```bash
python run_experiments.py
```
This runs all 60 questions through Baseline, RAG-k3, RAG-k5, and Loose RAG, then exports `results/results.csv` and generates all plots in `results/plots/`.

### 3. Launch Interactive Streamlit Web App
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser to test custom queries, inspect retrieved chunks, and explore the benchmark dashboard.

### 4. Run Step-by-Step in Jupyter / Google Colab
Open the notebooks in `notebooks/`:
* `01_build_rag.ipynb`: Ingestion, chunking, and semantic vector retrieval.
* `02_run_experiments.ipynb`: Running baseline vs RAG experiments.
* `03_evaluation.ipynb`: Metrics computation, plot rendering, and error analysis.

---

## 🔑 Supported LLM Providers

The project supports both live API providers and an offline mock simulator configured via `.env`:

```bash
cp .env.example .env
```

Edit `.env` with your desired provider:
```ini
# Supported: "opencode_zen", "groq", "gemini", "openai", or "local_mock"
LLM_PROVIDER=local_mock
LLM_MODEL=llama-3.1-8b-instant

# Optional Live API Keys
OPENCODE_ZEN_API_KEY=your_key_here
OPENCODE_ZEN_BASE_URL=https://api.opencodezen.com/v1

GROQ_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

> **Note**: `LLM_PROVIDER=local_mock` allows instant, 100% deterministic execution of the full benchmark, notebooks, and Streamlit app without requiring external API keys.

---

## 🔮 Next Steps & Future Work

1. **Cross-Encoder Re-ranking**: Adding a cross-encoder stage (e.g., `bge-reranker-large` or Cohere) to filter out low-relevance retrieved chunks.
2. **Self-Correction & Agentic RAG**: Implementing Self-RAG reflection loops to evaluate context relevance before final response synthesis.
3. **GraphRAG Integration**: Combining dense vector retrieval with Knowledge Graphs to improve multi-hop reasoning over complex entity relationships.

---

## 👤 Author
**Sourasis Karak**  
* GitHub: [@thekarak](https://github.com/thekarak)  
* Email: `devxkarak@gmail.com`

---

## 📜 License
MIT License. Free for academic and commercial use.
