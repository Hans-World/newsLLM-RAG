"""
Generation Pipeline (called by api/app.py to serve user queries)

Responsibility:
    Converts a user query into a grounded LLM response by searching
    the Qdrant vector store and passing retrieved context to the LLM.

Pipeline:
    5. Embed Query    Convert the user's question into dense + sparse vectors
    6. Retrieve       Hybrid search (dense + sparse + RRF) → top-k chunks from Qdrant
    7. Generate       Build a grounded prompt and stream the LLM response to the caller

Usage:
    Called by api/app.py — not run directly.
"""
from generation import hybrid_search, generate
from indexing.embedders import DenseEmbedder, BM25SparseEmbedder

# from generation import embed_query, hybrid_search, generate
# COLLECTION = "testing_v1"
COLLECTION = "news_samples"

def run_pipeline(query: str, dense_embedder: DenseEmbedder, sparse_embedder: BM25SparseEmbedder, top_k: int = 10):
    # Stage 5: Embed Query
    # dense_vector, sparse_vector = embed_query(query, dense_embedder, sparse_embedder)
    dense_vector  = dense_embedder.encode_query(query)
    sparse_vector = sparse_embedder.embed_query(query)

    # Stage 6: Retrieve
    chunks = hybrid_search(COLLECTION, dense_vector, sparse_vector, top_k=top_k)

    # Stage 7: Generate — yields tokens for StreamingResponse
    return generate(query, chunks)