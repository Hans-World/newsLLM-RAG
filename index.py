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
from pathlib import Path
# from fastembed import SparseTextEmbedding
# from indexing import load, chunk, embed_chunks, create_collection, delete_collection, store_chunks
from indexing import load, chunk, create_collection, delete_collection, store_chunks
from indexing.embedders import E5Embedder, BM25SparseEmbedder

# SAMPLES_DIR = Path("./notebooks/data/news.json")
# COLLECTION  = "testing_v1"
SAMPLES_DIR  = Path("./notebooks/data/samples")
COLLECTION   = "news_samples"

if __name__ == "__main__":
    sample_files = sorted(SAMPLES_DIR.glob("*.json"))
    print(f"=== INDEXING PIPELINE ===")
    print(f"Found {len(sample_files)} sample files\n")

    dense_embedder  = E5Embedder()
    sparse_embedder = BM25SparseEmbedder()
    global_id = 0

    for filepath in sample_files:
        source = filepath.stem
        print(f"--- [{source}] ---")

        # Stage 1: Load
        docs = load(str(filepath))
        print(f"✓ Loaded {len(docs)} documents")

        # Stage 2: Chunk
        all_chunks = []
        for doc in docs:
            all_chunks.extend(chunk(doc))
        print(f"✓ Chunked into {len(all_chunks)} chunks")

        # Stage 3: Embed
        # dense_vectors, sparse_vectors = embed_chunks(all_chunks)
        dense_vectors  = dense_embedder.encode_chunks(all_chunks)
        sparse_vectors = sparse_embedder.embed_documents([c.text for c in all_chunks])
        print(f"✓ Dense:  {len(dense_vectors)} vectors, dim={len(dense_vectors[0])}")
        print(f"✓ Sparse: {len(sparse_vectors)} vectors, nnz={len(sparse_vectors[0].indices)}")

        # Stage 4: Store
        DENSE_VECTOR_DIM = len(dense_vectors[0])
        create_collection(COLLECTION, DENSE_VECTOR_DIM)
        store_chunks(COLLECTION, all_chunks, dense_vectors, sparse_vectors, start_id=global_id)
        global_id += len(all_chunks)
        print(f"✓ Stored {len(all_chunks)} chunks into '{COLLECTION}'\n")