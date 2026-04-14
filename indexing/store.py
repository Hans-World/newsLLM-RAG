"""
Stage 4 — Store
Save vectors + chunk metadata into Qdrant.
"""

import os
import uuid
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


def store_chunks(collection, chunks, dense_vectors, sparse_vectors):
    points = []
    for i, chunk in enumerate(chunks):
        points.append(
            # Each chunk is stored as a record in the database (id + vector + metadata)
            PointStruct(
                # uuid is used for creating a unique chunk id
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{chunk.source}_{chunk.chunk_id}")),
                vector={
                    "dense": dense_vectors[i].tolist(),
                    "sparse": SparseVector(
                        indices=sparse_vectors[i].indices.tolist(), # WHICH positions have values
                        values=sparse_vectors[i].values.tolist() # WHAT those values are
                    )
                },
                # metadata for citation
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
        )
    
    # Update or insert a new point to a collection
    client.upsert(collection_name=collection, points=points)