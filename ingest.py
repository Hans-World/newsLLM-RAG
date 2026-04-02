"""
Indexing Pipeline (run once, or when new articles arrive)

Responsibility:
    Transforms raw news articles into searchable vectors stored in Qdrant.
    This is an offline, one-shot process — it does not serve user requests.

Pipeline:
    1. Load      Read raw articles from data/news.json → RawDocument[]
    2. Chunk     Split each article into sentence-window chunks → Chunk[]
    3. Embed     Convert each chunk into dense + sparse vectors
    4. Store     Upsert vectors and metadata into the Qdrant collection

When to run:
    - On first deployment to build the initial index
    - When new articles are added to the data source
    - When the embedding model is changed (requires full re-index)

Usage:
    uv run ingest.py
"""
from indexing import load, chunk, embed_chunks

if __name__ == "__main__":
    print("=== INDEXING PIPELINE ===")
    
    # Stage 1: Load
    PATH = "./notebooks/data/news.json"
    docs = load(PATH) # Potentially need to be replaced after we receive the data
    print(f"✓ Loaded {len(docs)} documents")
    
    # Stage 2: Chunk
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk(doc))
    print(f"✓ Turned {len(docs)} documents into {len(all_chunks)} chunks")
    
    # Stage 3: Embed
    dense_vectors, sparse_vectors = embed_chunks(all_chunks)
    print(f"✓ Dense:  {len(dense_vectors)} chunks, dimension={len(dense_vectors[0])}")
    print(f"✓ Sparse: {len(sparse_vectors)} chunks, example nnz={len(sparse_vectors[0].indices)} non-zero values")
    
    # Stage 4: Store
    