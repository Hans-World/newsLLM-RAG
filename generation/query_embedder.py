"""
Stage 5 — Embed Query
Convert the user's query into dense and sparse vectors for hybrid search.

Note: imports embedders directly from indexing/doc_embedder.py to guarantee
the same models are used in both pipelines. Using different models here would
place query vectors in a different space and silently break retrieval.
"""
def embed_query(query: str, dense_embedder, sparse_embedder):
    """Uses the same embedders from the indexing pipeline — same model guaranteed."""
    query_dense_vector = dense_embedder.encode(query, normalize_embeddings=True).tolist()
    # list(...) — forces the generator to compute and collect all results:
    # We only embed one query so we need [0] to unwrap it from the list
    query_sparse_vector = list(sparse_embedder.embed([query]))[0]
    return query_dense_vector, query_sparse_vector