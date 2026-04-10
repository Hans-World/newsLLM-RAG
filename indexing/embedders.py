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
import numpy as np
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding

class DenseEmbedder(ABC):
    @abstractmethod
    def encode_documents(self, texts: list[str]) -> np.ndarray: ...

    @abstractmethod
    def encode_query(self, query: str) -> list[float]: ...


class E5Embedder(DenseEmbedder):
    """
    intfloat/multilingual-e5-large-instruct
    License: MIT
    Requires an instruction prefix on queries for best retrieval performance.
    """
    MODEL = "intfloat/multilingual-e5-large-instruct"
    QUERY_PREFIX = "Instruct: Given a news article query, retrieve relevant news passages\nQuery: "

    def __init__(self):
        self.model = SentenceTransformer(self.MODEL, trust_remote_code=True)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    def encode_query(self, query: str) -> list[float]:
        return self.model.encode(self.QUERY_PREFIX + query, normalize_embeddings=True).tolist()
    
    def encode_chunks(self, chunks) -> np.ndarray:
        """Convenience method for the indexing pipeline — extracts text from Chunk objects."""
        texts = [c.text for c in chunks]
        return self.encode_documents(texts)


# class JinaV5Embedder(DenseEmbedder):
#     """
#     jinaai/jina-embeddings-v5-text-small
#     License: CC BY-NC 4.0 (non-commercial only)
#     Uses task/prompt_name parameters instead of instruction prefix.
#     """
#     MODEL = "jinaai/jina-embeddings-v5-text-small"
#
#     def __init__(self):
#         self.model = SentenceTransformer(self.MODEL, trust_remote_code=True)
#
#     def encode_documents(self, texts: list[str]) -> np.ndarray:
#         return self.model.encode(texts, task="retrieval", prompt_name="document", show_progress_bar=True)
#
#     def encode_query(self, query: str) -> list[float]:
#         return self.model.encode([query], task="retrieval", prompt_name="query")[0].tolist()


class BM25SparseEmbedder:
    def __init__(self):
        self.model = SparseTextEmbedding(model_name="Qdrant/bm25")

    def encode_documents(self, texts: list[str]):
        return list(self.model.embed(texts))

    def encode_query(self, query: str):
        return list(self.model.embed([query]))[0]