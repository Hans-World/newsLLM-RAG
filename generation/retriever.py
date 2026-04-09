"""
Stage 6 — Retrieve
Hybrid search (dense + sparse + RRF) against Qdrant to find the most relevant chunks.

Note: 
    we can also set the 'score_threshold' - Minimum similarity score for returned points

Approach:
    Step 1a: Dense search   — finds semantically similar chunks
    Step 1b: Sparse search  — finds keyword matching chunks
    Step 2:  RRF fusion     — merges both ranked lists into one final ranking
"""
import os
from datetime import datetime
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector, Prefetch, FusionQuery, Fusion
from indexing.chunker import Chunk

load_dotenv()
client = QdrantClient(host=os.getenv("QDRANT_HOST"), port=int(os.getenv("QDRANT_PORT")))


class RetrievedChunk:
    def __init__(self, chunk, score):
        self.chunk = chunk
        self.score = score

 
def keyword_search(collection, query_sparse_vector, top_k=10):
    results = client.query_points(
        collection_name=collection,
        query=SparseVector(
            indices=query_sparse_vector.indices.tolist(),  # WHICH positions have values
            values=query_sparse_vector.values.tolist(),    # WHAT those values are
        ),
        using="sparse",
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


def semantic_search(collection, query_dense_vector, top_k=10):
    results = client.query_points(
        collection_name=collection,
        query=query_dense_vector,
        using="dense",
        limit=top_k,
        with_payload=True
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


def hybrid_search(collection, query_dense_vector, query_sparse_vector, top_k=10):
    results = client.query_points(
        collection_name=collection,
        prefetch=[
            # Step 1a: dense search - finds semantically similar chunks
            Prefetch(
                query=query_dense_vector,
                using="dense",
                limit=top_k * 2,  # fetch more candidates than needed
            ),
            # Step 1b: sparse search - finds keyword matching chunks
            Prefetch(
                query=SparseVector(
                    indices=query_sparse_vector.indices.tolist(), # WHICH positions have values
                    values=query_sparse_vector.values.tolist() # WHAT those values are
                ),
                using="sparse",
                limit=top_k * 2,
            ),
        ],
        # Step 2: Reciprocal Rank Fusion — merges both ranked lists into one final ranking
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    ).points # sorted by score descending

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