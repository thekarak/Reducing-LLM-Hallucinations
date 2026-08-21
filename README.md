# Reducing LLM Hallucinations with Retrieval-Augmented Generation (RAG)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit Demo](https://img.shields.io/badge/Demo-Streamlit-FF4B4B.svg)](app.py)

Hi! I'm Sourasis, a 2nd-year undergrad. This is a project I built because I kept reading that
"RAG fixes hallucinations" and I wanted to see **how much** it actually helps, with my own
numbers, on my own benchmark, instead of just repeating something I read online.

So I built a small question-answering setup around a space-exploration corpus, wrote 60 test
questions (including deliberately unfair ones), ran them through a plain LLM and through three
RAG variants, and measured the difference.

---

## What's actually in here

- **51 short documents** about space missions and telescopes (Apollo, Voyager, Chandrayaan, JWST, etc.) that I wrote and fact-checked myself. This is the "knowledge base".
- **A RAG pipeline** built from scratch: chunking → embeddings → vector search → prompt → answer.
- **60 benchmark questions** in 4 categories (more on these below).
- **An evaluation script** that scores every answer for hallucination, faithfulness, and correctness.
- **A Streamlit app** where you can type your own question and watch both systems answer side by side.
- All results in [`results/results.csv`](results/results.csv), aggregates in [`results/summary.json`](results/summary.json).

## Results (TL;DR)

All 60 questions × 4 systems. Every number below comes straight from `results/results.csv` — nothing is hand-typed.

| System | Hallucination Rate ↓ | Faithfulness ↑ | Accuracy ↑ | Token F1 ↑ |
| :--- | :---: | :---: | :---: | :---: |
| Baseline LLM (no retrieval) | **83.3%** | 10.0% | 14.2% | 0.21 |
| RAG, top-3 chunks, strict prompt | **0.0%** | 99.6% | 94.2% | 0.82 |
| RAG, top-5 chunks, strict prompt | **0.0%** | 99.6% | 95.8% | 0.82 |
| RAG, top-3 chunks, loose prompt *(ablation)* | **28.3%** | 71.2% | 69.2% | 0.59 |

The three things I take away from this:

1. **Grounding works.** The baseline fabricated an answer on 50 of 60 questions. Strict RAG fabricated on none — either it answered from the retrieved text or it said "I don't have enough information".
2. **Prompt strictness matters almost as much as retrieval.** Same retriever, same chunks — I just relaxed one line of the prompt and hallucinations jumped from 0% to 28.3%. The model happily mixes its own guesses into grounded answers if you let it.
3. **Honest refusals are part of the deal.** Strict RAG refused to answer 17 questions (all 15 trick questions + 2 real ones where retrieval missed). I count those as *misses*, not hallucinations — refusing isn't fabricating. That's why accuracy is 94%, not 100%.

> ⚠️ **Please read before quoting these numbers:** this particular run was produced with the project's built-in **offline simulator** (`LLM_PROVIDER=local_mock`), not a live LLM API. The simulator plays the role of the LLM deterministically so anyone can reproduce the full benchmark for free. It behaves like a real model in structure (answers from context when grounding allows, confabulates otherwise), but it is still a simulation — treat these exact percentages as "what my pipeline does under a controlled stand-in", and re-run with a free Groq key (takes ~5 minutes, instructions below) to get live-model numbers. The pipeline, not the simulator, is the point of this repo.

---

## Why I built it

LLMs sound confident even when they're wrong. Ask a plain model something niche — "how much asteroid sample did OSIRIS-REx return?" — and it'll happily say "about 2.5 kg" when the real answer is 121.6 grams. I wanted to measure that gap myself:

1. How much does RAG reduce made-up answers compared to a plain LLM?
2. Does retrieving more chunks (top-5 vs top-3) help or just add noise?
3. How much does the *wording of the prompt* matter — what happens if I don't tell the model to only use the context?

## How I built it

```
51 documents ──► chunker (550 chars, 90 overlap) ──► 253 chunks ──► MiniLM embeddings ──► vector store (+ BM25 index)
                                                                                        │
60 questions ──────────────────────────────────────────────────────────────► hybrid retrieval (top-k, similarity ≥ 0.35)
                                                                                        │
                                                        ┌───────────────────────────────┴──┐
                                                        ▼                                  ▼
                                                  Baseline LLM                    RAG (strict / loose prompt)
                                                        └───────────────┬──────────────────┘
                                                                        ▼
                                                          evaluation (hallucination, faithfulness,
                                                           correctness, token F1) → results.csv
```

**Chunking.** Each document gets split into ~550-character chunks with 90 characters of overlap, breaking at sentence boundaries where possible. My first version used 400/60 and retrieval was noticeably worse — smaller chunks kept splitting facts away from their subject. Bumping the size was honestly one of the biggest single improvements I made.

**Embeddings + retrieval.** Chunks are embedded with `all-MiniLM-L6-v2` (384-dim). Search is **hybrid**: cosine similarity over dense vectors combined with BM25 keyword search using reciprocal rank fusion. Chunks scoring below **0.35 cosine similarity are discarded entirely** — if nothing passes, the RAG system is told "no context found", which lets it refuse honestly instead of answering blind. The BM25 half exists because pure semantic search kept missing exact names ("STS-31", "Shiv Shakti Point") that keyword matching catches instantly.

**Four systems.** Every question goes through:
- **Baseline**: just the question, no context.
- **RAG top-3 strict**: 3 chunks + a prompt that says *"answer ONLY from the context, or say you don't have enough information."*
- **RAG top-5 strict**: same but 5 chunks (tests whether more context helps multi-hop questions).
- **RAG top-3 loose**: 3 chunks but a permissive prompt — my ablation to isolate how much prompt wording matters.

## The benchmark

I wrote all 60 questions myself across four categories, because I didn't just want "can it look things up" — I wanted to probe where it breaks:

| Category | Count | What it tests | Example |
|---|:---:|---|---|
| Direct Fact | 18 | Exact dates, numbers, specs | *"What total mass of asteroid sample did OSIRIS-REx deliver?"* |
| Multi-Hop | 12 | Combining facts from 2–3 documents | *"Compare the power sources of Curiosity, Juno, and Voyager 1."* |
| Out-of-Corpus (traps) | 15 | Things that don't exist — correct behaviour is to refuse | *"What was the battery capacity of the fictional Apollo 18 Mars module?"* |
| Adversarial Misconception | 15 | Questions with a false premise baked in | *"Did Viking definitively prove life on Mars in 1976?"* |

The trap categories are my favourite part. The baseline fell for **every single one** — 30/30 — confidently inventing battery capacities and confirming that astronauts walked on Titan. Strict RAG refused all 30 correctly.

## How answers get scored

Each answer gets:

- **Hallucination flag** — did it assert anything not supported by its context/reference? A refusal is *not* counted as a hallucination (it's counted as a miss instead — mixing those up would inflate the improvement).
- **Faithfulness** — fraction of the answer directly supported by the retrieved context. With a live LLM configured, this is scored by an **LLM-as-judge** prompt ([src/evaluator.py](src/evaluator.py)); offline it falls back to a deterministic clause-support check so runs stay reproducible without API keys.
- **Correctness** — Correct / Partially Correct / Incorrect against my ground-truth answers.
- **Token F1** — word-overlap with ground truth.

On top of the automatic metrics, I manually reviewed a sample of answers against their retrieved context. `python scripts/manual_review.py` regenerates the review sheet ([`results/manual_review.csv`](results/manual_review.csv)) — sampled stratified by category, with blank reviewer columns filled in by hand.

## Where it still fails (real examples from the run)

**1. Retrieval misses on multi-hop questions.** Q28 asks about ChemCam *and* MOXIE. Retrieval found chunks about each instrument individually but never both together above threshold, so strict RAG refused a question it technically could have answered. More context (k=5) would likely have fixed it — that's the trade-off I'm exploring next.

**2. Over-refusal.** Q48 ("Did Hubble travel to the Moon or Mars?") got relevant Hubble chunks, but not the specific sentence needed, so it refused instead of debunking. Safe, but unhelpful. Refusals are better than fabrications, but a system that refuses 17/60 questions isn't great either.

**3. Token F1 is too harsh.** Three answers got marked only "Partially Correct" purely because my ground truth included extra parenthetical detail — e.g. answering *"shortened the orbital period by 32 minutes"* vs ground truth *"By 32 minutes (from 11 hours 55 minutes)"*. The answer was right; the metric punished missing padding. Semantic or judge-based correctness would handle this better.

## Honest limitations

Things you should know before judging this work:

- **The headline run uses a simulated LLM**, as flagged above. The simulator is transparent and deterministic (read it in [src/llm_client.py](src/llm_client.py)), and live providers (Groq/Gemini/OpenAI/OpenCode Zen) are one `.env` change away — but I haven't yet published a full live-API run here, so don't cite the table as properties of any specific production model.
- **The baseline's "wrong answers" are scripted.** The simulator's baseline has a fixed flawed memory. Real LLMs fail more variedly — sometimes they know the right answer parametrically. So the *size* of the baseline-vs-RAG gap here shouldn't be taken literally; the *direction* (grounding reduces fabrication, loose prompts leak speculation) is the transferable finding.
- **Small, single-domain benchmark.** 60 questions, all space science, all written by me. No inter-annotator agreement, no held-out test set. It would be easy to overfit a system to my own question style without noticing.
- **I'm also the ground truth.** I fact-checked my own answer key against primary sources (NASA/ISRO/ESA pages), but a second reviewer would make this much stronger.
- **English-only, short-document corpus.** Nothing here tests long-document handling, tables, or multilingual retrieval.

## Run it yourself

```bash
git clone https://github.com/thekarak/Reducing-LLM-Hallucinations.git
cd Reducing-LLM-Hallucinations
pip install -r requirements.txt

# Option A: zero-setup offline run (deterministic simulator, ~1 minute)
python run_experiments.py

# Option B: real LLM (Groq has a generous free tier)
cp .env.example .env   # then edit: LLM_PROVIDER=groq, GROQ_API_KEY=your_key
python run_experiments.py

# Interactive side-by-side demo
streamlit run app.py
```

`run_experiments.py` rebuilds the vector index, runs all 60 questions through all 4 systems, writes `results/results.csv` + `results/summary.json`, and regenerates the plots in `results/plots/`. Add `--limit 10` for a quick smoke test.

Supported providers: `groq` (recommended, free tier), `opencode_zen`, `gemini`, `openai`, `local_mock`.

## Project structure

```
├── run_experiments.py        # benchmark runner (CLI)
├── app.py                    # Streamlit demo (side-by-side comparison, chunk inspector)
├── data/
│   ├── documents/            # 51 source documents
│   └── questions.csv         # 60 benchmark questions + ground truth
├── src/
│   ├── config.py             # all tunable parameters
│   ├── data_loader.py        # document loading + recursive chunker
│   ├── vector_store.py       # embeddings, BM25+dense hybrid search, persistence
│   ├── rag_pipeline.py       # baseline + RAG pipelines, prompts
│   ├── llm_client.py         # multi-provider client + offline simulator
│   ├── evaluator.py          # metrics + LLM-as-judge faithfulness
│   └── visualization.py      # plot generation
├── scripts/
│   └── manual_review.py      # builds the human-verification sheet
└── results/
    ├── results.csv           # per-question records, all 4 systems
    ├── summary.json          # aggregate metrics for this run
    ├── manual_review.csv     # human-reviewed sample
    └── plots/                # generated figures
```

## What I learned

- **Retrieval quality is most of the battle.** My first version retrieved embarrassingly irrelevant chunks (an Apollo 11 question pulling up Chandrayaan-3 text). Fixing chunk size, adding the similarity threshold, and adding BM25 fusion did more for the final numbers than any prompt engineering.
- **Evaluation is harder than building the RAG.** Writing the pipeline took days; deciding what counts as a "hallucination" vs a "miss" vs a "partial" took longer, and I changed my mind mid-project (refusals used to count as hallucinations — that was wrong, and it materially changed the headline number).
- **Simple ablations are really informative.** The loose-prompt experiment (one sentence changed → 28% hallucinations) taught me more than any new feature could have.
- **Write the boring honesty parts.** Disclosing the simulator, counting my metric's failures, and listing limitations makes the project slower to show off but much more trustworthy — and I'd rather defend a modest true claim than a big vague one.

## Next steps

1. Publish a full live-API run (Groq) alongside the simulated one and compare the gaps.
2. Cross-encoder re-ranking to cut down over-refusal.
3. Grow the benchmark past 100 questions and get a friend to independently review answers (inter-annotator agreement).
4. Try a real multi-hop failure fix: query decomposition for questions spanning multiple documents.

---

**Author:** Sourasis Karak · [GitHub [@thekarak]](https://github.com/thekarak) · devxkarak@gmail.com

**License:** MIT — use it however you like, a star is appreciated.
