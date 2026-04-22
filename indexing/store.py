"""
Stage 4 — Store
Save vectors + chunk metadata into Qdrant.
"""

import os
import uuid
from tqdm import tqdm
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, SparseVector, SparseVectorParams, SparseIndexParams, Modifier

load_dotenv()
client = QdrantClient(host=os.getenv("QDRANT_HOST"), port=int(os.getenv("QDRANT_PORT")))

def delete_collection(collection):
    """ Delete the specified collection """
    client.delete_collection(collection_name=collection)


def create_collection(collection, dense_vector_dimension):
    """
    A collection is a named set of points (vectors with a payload) among which you can search
    """
    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config={
                # Dense Vector captures semantic meaning
                "dense": VectorParams(size=dense_vector_dimension, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                # Sparse Vector captures keyword 
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                    modifier=Modifier.IDF
                )
            }
        )

UPSERT_BATCH_SIZE = 256

def store_chunks(collection, chunks, dense_vectors, sparse_vectors):
    # Outer: sends the amount of batch size points, then moves to next slice
    total_batches = (len(chunks) + UPSERT_BATCH_SIZE - 1) // UPSERT_BATCH_SIZE
    for start in tqdm(range(0, len(chunks), UPSERT_BATCH_SIZE), total=total_batches, desc="Storing", unit="batch"):
        end = start + UPSERT_BATCH_SIZE
        batch = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{chunk.source}_{chunk.chunk_id}")),
                vector={
                    "dense": dense_vectors[i].tolist(),
                    "sparse": SparseVector(
                        indices=sparse_vectors[i].indices.tolist(), # WHICH positions have values
                        values=sparse_vectors[i].values.tolist()    # WHAT those values are
                    )
                },
                payload={
                    "chunk_id":     chunk.chunk_id,
                    "source_id":    chunk.source_id,
                    "text":         chunk.text,
                    "title":        chunk.title,
                    "url":          chunk.url,
                    "publish_date": chunk.publish_date.isoformat() if chunk.publish_date else None,
                    "source":       chunk.source,
                }
            )
            for i, chunk in enumerate(chunks[start:end], start=start) # Inner: iterates each chunk in that slice
        ]
        client.upsert(collection_name=collection, points=batch)