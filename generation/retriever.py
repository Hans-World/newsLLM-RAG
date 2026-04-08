"""
Stage 6 — Retrieve
Hybrid search (dense + sparse + RRF) against Qdrant to find the most relevant chunks.

Approach:
    Step 1a: Dense search   — finds semantically similar chunks
    Step 1b: Sparse search  — finds keyword matching chunks
    Step 2:  RRF fusion     — merges both ranked lists into one final ranking
"""
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector, Prefetch, FusionQuery, Fusion
from indexing.chunker import Chunk

client = QdrantClient(host="localhost", port=6333)


class RetrievedChunk:
    def __init__(self, chunk, score):
        self.chunk = chunk
        self.score = score


def retrieve(collection, query_dense_vector, query_sparse_vector, top_k=10):
    results = client.query_points(
        collection_name=collection,
        prefetch=[
            # Step 1a: dense search
            Prefetch(
                query=query_dense_vector,
                using="dense",
                limit=top_k * 2,  # fetch more candidates than needed
            ),
            # Step 1b: sparse search
            Prefetch(
                query=SparseVector(
                    indices=query_sparse_vector.indices.tolist(),
                    values=query_sparse_vector.values.tolist(),
                ),
                using="sparse",
                limit=top_k * 2,
            ),
        ],
        # Step 2: Reciprocal Rank Fusion — merges both ranked lists into one final ranking
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    ).points

    return [
        RetrievedChunk(
            chunk=Chunk(
                chunk_id=r.payload["chunk_id"],
                source_id=r.payload["source_id"],
                text=r.payload["text"],
                title=r.payload["title"],
                url=r.payload["url"],
                publish_date=datetime.fromisoformat(r.payload["publish_date"]),
            ),
            score=r.score,
        )
        for r in results
    ]