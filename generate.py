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
    Called by demo.py — not run directly.
"""
from generation import hybrid_search, generate
from indexing import E5Embedder, BM25SparseEmbedder

# from generation import embed_query, hybrid_search, generate
# COLLECTION = "testing_v1"
COLLECTION = "news_samples"

def run_pipeline(query: str, dense_embedder: E5Embedder, sparse_embedder: BM25SparseEmbedder, top_k: int = 10):
    # Stage 5: Embed Query
    dense_vector  = dense_embedder.encode_query(query)
    sparse_vector = sparse_embedder.encode_query(query)

    # Stage 6: Retrieve
    retrieved_chunks = hybrid_search(COLLECTION, dense_vector, sparse_vector, top_k=top_k)
    print(f"--- [Retrived {top_k} Chunks] ---")
    for i, rc in enumerate(retrieved_chunks):
        print(f"[{i+1}] {rc.chunk.title}  |  score: {rc.score:.4f}  |  {rc.chunk.publish_date}  |  {rc.chunk.source}")
        print(f"    {rc.chunk.url}")
        print(f"    {rc.chunk.text[:80].strip()}...")
        print()

    # Stage 7: Generate — yields tokens for StreamingResponse
    llm_response = generate(query, retrieved_chunks)
    return llm_response