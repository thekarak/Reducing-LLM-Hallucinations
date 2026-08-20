import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from src.config import VECTOR_STORE_DIR, EMBEDDING_MODEL_NAME
from src.data_loader import Document

class EmbeddingEngine:
    """Wrapper around sentence-transformers with a lightweight fallback."""
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self._init_model()

    def _init_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.is_neural = True
        except Exception as e:
            # Lightweight deterministic vectorizer fallback
            print(f"[VectorStore] SentenceTransformer not loaded ({e}), using TF-IDF / hashing fallback.")
            self.is_neural = False

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        if self.is_neural and self.model is not None:
            return np.array(self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True))
        else:
            return self._fallback_embed(texts)

    def embed_query(self, text: str) -> np.ndarray:
        if self.is_neural and self.model is not None:
            emb = self.model.encode([text], show_progress_bar=False, normalize_embeddings=True)[0]
            return np.array(emb)
        else:
            return self._fallback_embed([text])[0]

    def _fallback_embed(self, texts: List[str], dim: int = 384) -> np.ndarray:
        """Deterministic character & token hashing embedding for offline execution."""
        embeddings = []
        for text in texts:
            vec = np.zeros(dim, dtype=np.float32)
            tokens = text.lower().replace(",", " ").replace(".", " ").replace("(", " ").replace(")", " ").split()
            for token in tokens:
                # Hash token into vector dimensions
                h = abs(hash(token)) % dim
                vec[h] += 1.0
                # Bigram hash
                if len(token) >= 3:
                    h2 = abs(hash(token[:3])) % dim
                    vec[h2] += 0.5
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec)
        return np.array(embeddings, dtype=np.float32)


class SimpleVectorStore:
    """Fast, reliable in-memory Vector Store with disk persistence and cosine similarity."""
    def __init__(self, embedding_engine: EmbeddingEngine = None):
        self.embedding_engine = embedding_engine or EmbeddingEngine()
        self.documents: List[Document] = []
        self.vectors: np.ndarray = np.empty((0, 384), dtype=np.float32)

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

    def similarity_search_with_score(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        if len(self.documents) == 0:
            return []
        
        query_vec = self.embedding_engine.embed_query(query)
        # Cosine similarity (vectors are normalized)
        scores = np.dot(self.vectors, query_vec)
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            results.append((self.documents[idx], score))
        return results

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        results = self.similarity_search_with_score(query, k=k)
        return [doc for doc, _ in results]

    def save(self, directory: Path = VECTOR_STORE_DIR):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Save vectors
        np.save(directory / "vectors.npy", self.vectors)
        
        # Save document metadata and content
        docs_data = [
            {"page_content": d.page_content, "metadata": d.metadata}
            for d in self.documents
        ]
        with open(directory / "documents.json", "w", encoding="utf-8") as f:
            json.dump(docs_data, f, indent=2)

    @classmethod
    def load(cls, directory: Path = VECTOR_STORE_DIR, embedding_engine: EmbeddingEngine = None):
        directory = Path(directory)
        store = cls(embedding_engine=embedding_engine)
        
        vec_file = directory / "vectors.npy"
        docs_file = directory / "documents.json"
        
        if vec_file.exists() and docs_file.exists():
            store.vectors = np.load(vec_file)
            with open(docs_file, "r", encoding="utf-8") as f:
                docs_data = json.load(f)
            store.documents = [
                Document(page_content=item["page_content"], metadata=item["metadata"])
                for item in docs_data
            ]
        return store
