"""
This file takes core pipeline (run_RAG from generate) and makes it callable via HTTP.
Now this file runs as a server, and any client(browser, app, another service) can send 
a query and get back news chunks. Exposes run_RAG as a REST endpoint for downstream LLM tasks.

Run with: uv run fastapi dev api/app.py
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from generate import run_RAG
from indexing import E5Embedder, BM25SparseEmbedder

# --- [How API, REST, and FastAPI fit together] ---
# 1. API (Application Programming Interface) is the general concept: 
# a defined cnotract for how two pieces of software talk to each other
# 2. REST (Representation State Transfer): is the common style for web API 
# design. It uses HTTP methods as verbs: 
# - GET /health - read-only check, no body needed
# - POST /query - send data (query string), get data back
# 2. FastAPI: is the Python framework that handles all the HTTP plumbing

# --- Request / Response models --- #

class QueryRequest(BaseModel):
    query:  str
    top_k:  int = 10

class ChunkResponse(BaseModel):
    chunk_id:     str
    source_id:    str
    text:         str
    title:        str
    url:          str | None
    publish_date: str | None
    source:       str
    score:        float

class RAGResponse(BaseModel):
    chunks:      list[ChunkResponse]
    parent_docs: dict[str, str]


# --- Load embedders once at startup --- #

embedders = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    embedders["dense"]   = E5Embedder()
    embedders["sparse"]  = BM25SparseEmbedder()
    yield   # server runs here, embedders stay loaded in memory

app = FastAPI(lifespan=lifespan)


# --- Endpoints --- #

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query", response_model=RAGResponse)
def query(request: QueryRequest):
    retrieved_chunks, parent_docs = run_RAG(
        query           = request.query,
        dense_embedder  = embedders["dense"],
        sparse_embedder = embedders["sparse"],
        top_k           = request.top_k,
    )
    
    # Unpack RetrievedChunk objects (rc.chunk, rc.score) into serializable Pydantic models for the HTTP response
    chunks = [
        ChunkResponse( 
            chunk_id     = rc.chunk.chunk_id,
            source_id    = rc.chunk.source_id,
            text         = rc.chunk.text,
            title        = rc.chunk.title,
            url          = rc.chunk.url,
            publish_date = rc.chunk.publish_date.isoformat() if rc.chunk.publish_date else None,
            source       = rc.chunk.source,
            score        = rc.score,
        )
        for rc in retrieved_chunks
    ]
    return RAGResponse(chunks=chunks, parent_docs=parent_docs)