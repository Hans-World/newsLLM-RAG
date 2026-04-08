"""
FastAPI application — defines all routes.

Routes:
    GET  /health   Check if the server is alive
    POST /ask      Accept a user query and stream back a grounded LLM response
"""
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
# from generation import embed_query, retrieve, generate

app = FastAPI()


class QueryRequest(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
async def ask(request: QueryRequest):
    # Stage 5: Embed Query
    # dense_vector, sparse_vector = embed_query(request.query)

    # Stage 6: Retrieve
    # chunks = retrieve(dense_vector, sparse_vector)

    # Stage 7: Generate (streaming)
    # return StreamingResponse(generate(request.query, chunks), media_type="text/plain")
    return {"message": "Generation pipeline not yet implemented"}