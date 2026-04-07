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

Note: 
    Call the Qdrant API and show me all the collections
    command: curl -s http://localhost:6333/collections
             curl -s http://localhost:6333/healthz
"""
from indexing import load, chunk, embed_chunks, create_collection, delet_collection, store_chunks

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
    COLLECTION = "testing_v1"
    DENSE_VECTOR_DIM = len(dense_vectors[0])
    create_collection(COLLECTION, DENSE_VECTOR_DIM)
    store_chunks(COLLECTION, all_chunks, dense_vectors, sparse_vectors)
    print(f"✓ Stored {len(all_chunks)} chunks into '{COLLECTION}'")
    # delete_collection(COLLECTION)