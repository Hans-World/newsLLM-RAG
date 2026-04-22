"""
Embedder Classes — single source of truth for all embedding.

Dense (DenseEmbedder ABC):
    - encode_documents(texts)  → used by the indexing pipeline
    - encode_query(query)      → used by the generation pipeline
    - encode_chunks(chunks)    → convenience wrapper for indexing pipeline

Sparse (BM25SparseEmbedder):
    - embed_documents(texts)   → used by the indexing pipeline
    - embed_query(query)       → used by the generation pipeline

To switch models: instantiate a different class — no other code changes needed.
"""
import torch
import numpy as np
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# RTX 3090 (24GB) comfortably fits E5-large (~2GB) with batch_size=256
ENCODE_BATCH_SIZE = 256

class DenseEmbedder(ABC):
    @abstractmethod
    def encode_documents(self, texts: list[str]) -> np.ndarray: ...

    @abstractmethod
    def encode_query(self, query: str) -> list[float]: ...

class BM25SparseEmbedder:
    def __init__(self):
        self.model = SparseTextEmbedding(model_name="Qdrant/bm25")

    def encode_documents(self, texts: list[str]):
        return list(self.model.embed(texts))

    def encode_query(self, query: str):
        return list(self.model.embed([query]))[0]

class E5Embedder(DenseEmbedder):
    """
    intfloat/multilingual-e5-large-instruct
    License: MIT
    Requires an instruction prefix on queries for best retrieval performance.
    """
    MODEL = "intfloat/multilingual-e5-large-instruct"
    QUERY_PREFIX = "Instruct: Given a news article query, retrieve relevant news passages\nQuery: "

    def __init__(self):
        self.model = SentenceTransformer(self.MODEL, trust_remote_code=True, device=DEVICE)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=ENCODE_BATCH_SIZE)

    def encode_query(self, query: str) -> list[float]:
        return self.model.encode(self.QUERY_PREFIX + query, normalize_embeddings=True).tolist()
    
    def encode_chunks(self, chunks) -> np.ndarray:
        """Convenience method for the indexing pipeline — extracts text from Chunk objects."""
        texts = [c.text for c in chunks]
        return self.encode_documents(texts)