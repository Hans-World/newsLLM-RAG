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
    uv run index.py
    uv run index.py --data-dir /path/to/articles

Note:
    Call the Qdrant API and show me all the collections
    command: curl -s http://localhost:6333/collections
             curl -s http://localhost:6333/healthz
"""
import argparse
from pathlib import Path
from tqdm import tqdm
from indexing import load, chunk, create_collection, delete_collection, store_chunks, E5Embedder, BM25SparseEmbedder

DEFAULT_SAMPLES_DIR = Path("./notebooks/data/samples")
COLLECTION  = "公視" # "news_all", "news_samples", "testing_v1", "公視"

if __name__ == "__main__":
    # CLI arguments for custom data source 
    parser = argparse.ArgumentParser(description="NewsLLM Indexing Pipeline")
    parser.add_argument("--data-dir",   type=Path, default=DEFAULT_SAMPLES_DIR, help="Directory containing JSON article files")
    args = parser.parse_args()

    SAMPLES_DIR = args.data_dir

    # Accept either a single .json file or a directory of .json files
    sample_files = [SAMPLES_DIR] if SAMPLES_DIR.is_file() else sorted(SAMPLES_DIR.glob("*.json"))
    print(f"=== INDEXING PIPELINE ===")
    print(f"Found {len(sample_files)} sample files\n")

    dense_embedder  = E5Embedder()
    sparse_embedder = BM25SparseEmbedder()

    for filepath in sample_files:
        source = filepath.stem
        print(f"--- [{source}] ---")

        # Stage 1: Load
        docs = load(str(filepath))
        print(f"✓ Loaded {len(docs)} documents")

        # Stage 2: Chunk
        all_chunks = []
        for doc in tqdm(docs, desc="Chunking", unit="doc"):
            all_chunks.extend(chunk(doc))
        print(f"✓ Chunked into {len(all_chunks)} chunks")

        # Stage 3: Embed
        # dense_vectors, sparse_vectors = embed_chunks(all_chunks)
        dense_vectors  = dense_embedder.encode_chunks(all_chunks)
        sparse_vectors = sparse_embedder.encode_documents([c.text for c in all_chunks])
        print(f"✓ Dense:  {len(dense_vectors)} vectors, dim={len(dense_vectors[0])}")
        avg_nnz = sum(len(v.indices) for v in sparse_vectors) / len(sparse_vectors)
        print(f"✓ Sparse: {len(sparse_vectors)} vectors, avg_nnz={avg_nnz:.1f}")
        # print(f"✓ Sparse: {len(sparse_vectors)} vectors, nnz={len(sparse_vectors[0].indices)}")

        # Stage 4: Store
        DENSE_VECTOR_DIM = len(dense_vectors[0])
        create_collection(COLLECTION, DENSE_VECTOR_DIM)
        store_chunks(COLLECTION, all_chunks, dense_vectors, sparse_vectors)
        print(f"✓ Stored {len(all_chunks)} chunks into '{COLLECTION}'\n")