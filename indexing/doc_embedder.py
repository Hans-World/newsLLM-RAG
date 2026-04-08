"""
Stage 3 — Embed
Convert each chunk's text into dense and sparse vectors for hybrid search.

Dense vectors  (SentenceTransformer) capture semantic meaning.
Sparse vectors (BM25) capture keyword frequency for exact term matching.

Note: model names must stay in sync with generation/query_embedder.py —
both pipelines must use the same models or retrieval silently breaks.
"""
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding

DENSE_MODEL = "intfloat/multilingual-e5-large-instruct" # "intfloat/multilingual-e5-large-instruct" / "jinaai/jina-embeddings-v3"
SPARSE_MODEL = "Qdrant/bm25"

dense_embedder = SentenceTransformer(DENSE_MODEL, trust_remote_code=True) # Semantic Matching
sparse_embedder = SparseTextEmbedding(model_name=SPARSE_MODEL)  # Keyword Matching

def download_models():
    """Download and cache both embedding models. Call this before serving if ingest.py has not been run."""
    SentenceTransformer(DENSE_MODEL, trust_remote_code=True)
    list(SparseTextEmbedding(model_name=SPARSE_MODEL).embed(["test"]))
    print(f"✓ Models downloaded and cached")


def embed_chunks(chunks):
    texts = [c.text for c in chunks]
    dense_vectors = dense_embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    sparse_vectors = list(sparse_embedder.embed(texts))
    return dense_vectors, sparse_vectors