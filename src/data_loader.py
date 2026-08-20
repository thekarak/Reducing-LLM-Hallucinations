import os
import re
import csv
from pathlib import Path
from typing import List, Dict, Any
from src.config import DOCUMENTS_DIR, QUESTIONS_FILE, CHUNK_SIZE, CHUNK_OVERLAP

class Document:
    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        self.page_content = page_content
        self.metadata = metadata or {}

    def __repr__(self):
        source = self.metadata.get("source", "unknown")
        doc_id = self.metadata.get("id", "none")
        return f"<Document id={doc_id} source={source} len={len(self.page_content)}>"

class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP, separators: List[str] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        text = text.strip()
        if len(text) <= self.chunk_size:
            return [text] if text else []
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            if end >= len(text):
                chunks.append(text[start:].strip())
                break
            
            # Find best split point near end
            split_pos = -1
            for sep in self.separators:
                if sep == "":
                    split_pos = end
                    break
                pos = text.rfind(sep, start, end)
                if pos != -1 and pos > start:
                    split_pos = pos + len(sep)
                    break
            
            if split_pos == -1 or split_pos <= start:
                split_pos = end
            
            chunk = text[start:split_pos].strip()
            if chunk:
                chunks.append(chunk)
            
            start = max(start + 1, split_pos - self.chunk_overlap)
        
        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        split_docs = []
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for idx, chunk in enumerate(chunks):
                meta = doc.metadata.copy()
                meta["chunk_id"] = f"{meta.get('doc_id', 'doc')}_{idx}"
                meta["chunk_index"] = idx
                split_docs.append(Document(page_content=chunk, metadata=meta))
        return split_docs


def load_documents(directory: Path = DOCUMENTS_DIR) -> List[Document]:
    """Load all text documents from the specified directory."""
    documents = []
    if not directory.exists():
        return documents
    
    files = sorted(list(directory.glob("*.txt")) + list(directory.glob("*.md")))
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                doc_id = file_path.stem
                meta = {
                    "doc_id": doc_id,
                    "source": file_path.name,
                    "file_path": str(file_path)
                }
                documents.append(Document(page_content=content, metadata=meta))
    return documents


def load_questions(csv_path: Path = QUESTIONS_FILE) -> List[Dict[str, Any]]:
    """Load evaluation questions from CSV."""
    questions = []
    if not csv_path.exists():
        return questions
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(row)
    return questions
