"""
FastAPI wrapper for the RAG retrieval pipeline.
Exposes run_RAG as a REST endpoint for downstream LLM tasks.

Run with: uv run fastapi dev api/app.py
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from generate import run_RAG
from indexing import E5Embedder, BM25SparseEmbedder


# --- Request / Response models --- #

class QueryRequest(BaseModel):
    query:  str
    top_k:  int = 10

class ChunkResponse(BaseModel):
    chunk_id:     str
    source_id:    str
    text:         str
    title:        str
    url:          str
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
