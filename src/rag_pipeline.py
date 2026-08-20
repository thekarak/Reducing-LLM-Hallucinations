import time
from typing import List, Dict, Any, Optional
from src.data_loader import Document
from src.vector_store import SimpleVectorStore
from src.llm_client import LLMClient

STRICT_RAG_PROMPT_TEMPLATE = """You are a factual, concise question-answering assistant.
Context:
{context}

Question: {question}

Instructions:
1. Answer the question strictly and ONLY using the facts present in the provided context above.
2. Do not assume, extrapolate, or invent any outside knowledge.
3. If the context does not contain enough information to answer the question, state: "I do not have enough information in the provided context to answer this question."

Answer:"""

LOOSE_RAG_PROMPT_TEMPLATE = """You are a helpful assistant.
Context:
{context}

Question: {question}

Answer the question as best as you can based on the context or your general knowledge:"""

BASELINE_PROMPT_TEMPLATE = """You are a knowledgeable assistant. Answer the following question accurately and concisely.

Question: {question}

Answer:"""


class BaselinePipeline:
    """Baseline LLM pipeline without retrieval augmentation."""
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    def query(self, question: str) -> Dict[str, Any]:
        prompt = BASELINE_PROMPT_TEMPLATE.format(question=question)
        start_time = time.time()
        answer = self.llm.generate(prompt=prompt)
        elapsed_ms = (time.time() - start_time) * 1000.0

        return {
            "question": question,
            "answer": answer.strip(),
            "pipeline_type": "baseline",
            "retrieved_docs": [],
            "retrieved_context": "",
            "latency_ms": round(elapsed_ms, 2)
        }


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline with configurable Top-K and grounding prompts."""
    def __init__(self, vector_store: SimpleVectorStore, llm_client: Optional[LLMClient] = None, top_k: int = 3, strict_grounding: bool = True):
        self.vector_store = vector_store
        self.llm = llm_client or LLMClient()
        self.top_k = top_k
        self.strict_grounding = strict_grounding

    def retrieve(self, question: str, k: Optional[int] = None) -> List[Document]:
        k_val = k if k is not None else self.top_k
        return self.vector_store.similarity_search(question, k=k_val)

    def query(self, question: str, k: Optional[int] = None, strict: Optional[bool] = None) -> Dict[str, Any]:
        k_val = k if k is not None else self.top_k
        is_strict = strict if strict is not None else self.strict_grounding
        
        start_time = time.time()
        # 1. Retrieve relevant chunks
        retrieved_docs = self.vector_store.similarity_search(question, k=k_val)
        
        # 2. Format Context
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            src = doc.metadata.get("source", "doc")
            context_parts.append(f"[{i}] ({src})\n{doc.page_content}")
        context_text = "\n\n".join(context_parts)

        # 3. Construct Prompt
        template = STRICT_RAG_PROMPT_TEMPLATE if is_strict else LOOSE_RAG_PROMPT_TEMPLATE
        prompt = template.format(context=context_text, question=question)

        # 4. Generate Answer
        answer = self.llm.generate(prompt=prompt)
        elapsed_ms = (time.time() - start_time) * 1000.0

        return {
            "question": question,
            "answer": answer.strip(),
            "pipeline_type": f"rag_top{k_val}_{'strict' if is_strict else 'loose'}",
            "top_k": k_val,
            "strict_grounding": is_strict,
            "retrieved_docs": retrieved_docs,
            "retrieved_context": context_text,
            "latency_ms": round(elapsed_ms, 2)
        }
