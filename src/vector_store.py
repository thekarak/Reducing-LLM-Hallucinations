import os
import re
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from src.config import (
    VECTOR_STORE_DIR, EMBEDDING_MODEL_NAME,
    SIMILARITY_THRESHOLD, RETRIEVAL_MODE, BM25_CANDIDATES
)
from src.data_loader import Document


def _tokenize(text: str) -> List[str]:
    """Simple word tokenizer shared by BM25 and the TF-IDF fallback."""
    return re.findall(r"[a-z0-9]+", text.lower())


class EmbeddingEngine:
    """Dense embedding engine.

    Primary: sentence-transformers (all-MiniLM-L6-v2).
    Fallback: scikit-learn TF-IDF fitted on the corpus (deterministic, CPU-only).
    """
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self._tfidf = None          # fitted TfidfVectorizer (fallback mode)
        self._init_model()

    def _init_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.is_neural = True
            self.engine_type = "minilm"
        except Exception as e:
            print(f"[VectorStore] SentenceTransformer unavailable ({e}); using TF-IDF fallback.")
            print("[VectorStore] NOTE: TF-IDF retrieval is weaker. `pip install sentence-transformers` is recommended.")
            self.is_neural = False
            self.engine_type = "tfidf"

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        if self.is_neural and self.model is not None:
            return np.array(self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True))
        return self._tfidf_fit_transform(texts)

    def embed_query(self, text: str) -> np.ndarray:
        if self.is_neural and self.model is not None:
            emb = self.model.encode([text], show_progress_bar=False, normalize_embeddings=True)[0]
            return np.array(emb)
        return self._tfidf_transform(text)

    def _tfidf_fit_transform(self, texts: List[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._tfidf = TfidfVectorizer(tokenizer=_tokenize, lowercase=False, sublinear_tf=True)
        matrix = self._tfidf.fit_transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def _tfidf_transform(self, text: str) -> np.ndarray:
        if self._tfidf is None:
            raise RuntimeError("TF-IDF engine used before fitting on documents. Rebuild the vector store.")
        vec = self._tfidf.transform([text]).toarray()[0].astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


class SimpleVectorStore:
    """Vector store with cosine similarity search, optional BM25 hybrid fusion,
    a minimum-similarity threshold, and disk persistence."""

    def __init__(self, embedding_engine: EmbeddingEngine = None,
                 similarity_threshold: float = SIMILARITY_THRESHOLD,
                 retrieval_mode: str = RETRIEVAL_MODE):
        self.embedding_engine = embedding_engine or EmbeddingEngine()
        self.documents: List[Document] = []
        self.vectors: np.ndarray = np.empty((0, 384), dtype=np.float32)
        self.similarity_threshold = similarity_threshold
        self.retrieval_mode = retrieval_mode
        self._bm25 = None

    # ------------------------------------------------------------------ index
    def add_documents(self, docs: List[Document]):
        if not docs:
            return
        texts = [d.page_content for d in docs]
        new_vectors = self.embedding_engine.embed_documents(texts)

        self.documents.extend(docs)
        if self.vectors.shape[0] == 0:
            self.vectors = new_vectors
        else:
            self.vectors = np.vstack([self.vectors, new_vectors])
        self._rebuild_bm25()

    def _rebuild_bm25(self):
        if self.retrieval_mode != "hybrid":
            self._bm25 = None
            return
        try:
            from rank_bm25 import BM25Okapi
            corpus_tokens = [_tokenize(d.page_content) for d in self.documents]
            self._bm25 = BM25Okapi(corpus_tokens) if corpus_tokens else None
        except ImportError:
            print("[VectorStore] rank_bm25 not installed; falling back to dense-only retrieval.")
            print("[VectorStore] Enable hybrid search with: pip install rank-bm25")
            self._bm25 = None

    # ----------------------------------------------------------------- search
    def similarity_search_with_score(self, query: str, k: int = 3,
                                     threshold: Optional[float] = None) -> List[Tuple[Document, float]]:
        """Retrieve top-k chunks.

        - Dense candidates below `threshold` cosine similarity are discarded.
        - In hybrid mode, BM25 keyword candidates are fused with dense ranking
          via Reciprocal Rank Fusion before selecting the final top-k.
        Returns [] when nothing clears the threshold (caller should treat this
        as "no relevant context found").
        """
        if len(self.documents) == 0:
            return []

        thr = self.similarity_threshold if threshold is None else threshold
        scores = np.dot(self.vectors, self.embedding_engine.embed_query(query))

        # Dense candidates that clear the relevance threshold.
        dense_order = np.argsort(scores)[::-1]
        dense_hits = [(int(i), float(scores[i])) for i in dense_order if scores[i] >= thr]

        candidate_ids: Dict[int, float] = {i: s for i, s in dense_hits}

        if self.retrieval_mode == "hybrid" and self._bm25 is not None:
            bm25_scores = self._bm25.get_scores(_tokenize(query))
            bm25_order = np.argsort(bm25_scores)[::-1][:BM25_CANDIDATES]
            # Keep only keyword hits with a non-trivial score.
            bm25_hits = [(int(i), float(bm25_scores[i])) for i in bm25_order if bm25_scores[i] > 0]
            # Reciprocal Rank Fusion over the union of both rankings.
            fused: Dict[int, float] = {}
            for rank, (i, _) in enumerate(dense_hits):
                fused[i] = fused.get(i, 0.0) + 1.0 / (60 + rank + 1)
            for rank, (i, _) in enumerate(bm25_hits):
                fused[i] = fused.get(i, 0.0) + 1.0 / (60 + rank + 1)
            ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
            # Report the underlying cosine score per selected chunk.
            return [(self.documents[i], candidate_ids.get(i, float(scores[i]))) for i, _ in ranked]

        return [(self.documents[i], s) for i, s in dense_hits[:k]]

    def similarity_search(self, query: str, k: int = 3,
                          threshold: Optional[float] = None) -> List[Document]:
        results = self.similarity_search_with_score(query, k=k, threshold=threshold)
        return [doc for doc, _ in results]

    # ------------------------------------------------------------- persistence
    def save(self, directory: Path = VECTOR_STORE_DIR):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        docs_data = [
            {"page_content": d.page_content, "metadata": d.metadata}
            for d in self.documents
        ]
        index_meta = {
            "engine_type": self.embedding_engine.engine_type,
            "model_name": self.embedding_engine.model_name,
            "num_chunks": len(self.documents),
            "chunk_size": self.documents[0].metadata.get("chunk_size") if self.documents else None,
        }
        with open(directory / "documents.json", "w", encoding="utf-8") as f:
            json.dump({"index_meta": index_meta, "documents": docs_data}, f, indent=2)

    @classmethod
    def load(cls, directory: Path = VECTOR_STORE_DIR, embedding_engine: EmbeddingEngine = None,
             similarity_threshold: float = SIMILARITY_THRESHOLD,
             retrieval_mode: str = RETRIEVAL_MODE):
        directory = Path(directory)
        store = cls(embedding_engine=embedding_engine,
                    similarity_threshold=similarity_threshold,
                    retrieval_mode=retrieval_mode)

        vec_file = directory / "vectors.npy"
        docs_file = directory / "documents.json"

        if vec_file.exists() and docs_file.exists():
            store.vectors = np.load(vec_file)
            with open(docs_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            # Backwards compatibility with the old plain-list format.
            if isinstance(payload, list):
                docs_data, meta = payload, {}
            else:
                docs_data, meta = payload["documents"], payload.get("index_meta", {})

            stored_engine = meta.get("engine_type")
            if stored_engine and stored_engine != store.embedding_engine.engine_type:
                raise RuntimeError(
                    f"Vector store was built with engine '{stored_engine}' but current "
                    f"engine is '{store.embedding_engine.engine_type}'. Delete the "
                    f"'{directory}' folder and rebuild the index for consistent retrieval."
                )

            store.documents = [
                Document(page_content=item["page_content"], metadata=item["metadata"])
                for item in docs_data
            ]
            store._rebuild_bm25()
        return store
