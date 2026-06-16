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
from generation import hybrid_search, generate, rewrite_query
from indexing import fetch_articles, E5Embedder, BM25SparseEmbedder

# Should share the same COLLECTION with index.py
COLLECTION = "all_news" # "test_all_media", "all_news"

def run_pipeline(query: str, dense_embedder: E5Embedder, sparse_embedder: BM25SparseEmbedder, history: list[dict] | None=None, top_k: int = 10, isQueryRewrite = False):
    """
    Full RAG pipeline for demo and testing purposes.
    Retrieved relevant chunks and generates an LLM response internally.
    
    Use this for:
        - Streamlit demo
        - Local testing and evaluation
    """
    # Query Rewrite if needed
    print(f"--- [User Query] ---\n[Original] : {query}")
    if isQueryRewrite:
        query = rewrite_query(query)
        print(f"[Rewritten]: {query}")
    print()
        
    # Stage 5: Embed Query
    dense_vector  = dense_embedder.encode_query(query)
    sparse_vector = sparse_embedder.encode_query(query)
        
    # Stage 6: Retrieve
    retrieved_chunks = hybrid_search(COLLECTION, dense_vector, sparse_vector, top_k=top_k)
    print(f"--- [Retrived {top_k} Chunks] ---")
    source_ids = []
    for i, rc in enumerate(retrieved_chunks):
        source_ids.append(rc.chunk.source_id)
        print(f"[{i+1}] 標題：{rc.chunk.title}  |  score: {rc.score:.4f}")
        print(f"    來源：{rc.chunk.source}")
        print(f"    報導時間：{rc.chunk.publish_date}")
        print(f"    連結：{rc.chunk.url}")
        # print(f"    內容：{rc.chunk.text[:80].strip()}...")
        print(f"    內容：{rc.chunk.text}")
        print()
        
    # Stage 6.5: Fetch Parent Documents from Retrieved Chunks
    parent_docs = fetch_articles(source_ids)

    # Stage 7: Generate — yields tokens for StreamingResponse
    llm_response = generate(query, retrieved_chunks, history)
    return llm_response, retrieved_chunks # expose chunks for eval


def run_RAG(query: str, dense_embedder: E5Embedder, sparse_embedder: BM25SparseEmbedder, top_k: int = 10):
    """
    Pure retrieval pipeline for production deployment.
    Decouples retrieval from generation - caller brings their own LLM
    
    Use this for:
        - FastAPI deployment
        - Downstream task 1: pipe retrieved_chunks to a pre-trained and fine-tuned LLM for Q&A
        - Downstream task 2: pipe parent_docs to a news analysis/explanation module (Polar-Chain)
    """
    # Stage 5: Embed Query
    dense_vector  = dense_embedder.encode_query(query)
    sparse_vector = sparse_embedder.encode_query(query)
    
    # Stage 6: Retrieve
    retrieved_chunks = hybrid_search(COLLECTION, dense_vector, sparse_vector, top_k=top_k)
    
    # Stage 6.5: Fetch Parent Documents
    source_ids = list({rc.chunk.source_id for rc in retrieved_chunks}) # removes duplicates before hitting SQLite
    parent_docs = fetch_articles(source_ids)
    
    return retrieved_chunks, parent_docs