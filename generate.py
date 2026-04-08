"""
Generation Pipeline (long-running API server)

Responsibility:
    Serves user queries by retrieving relevant news chunks from Qdrant
    and generating grounded answers via the LLM. Runs continuously after deployment.

Pipeline:
    5. Embed Query    Convert the user's question into dense + sparse vectors
    6. Retrieve       Hybrid search (dense + sparse + RRF) → top-k chunks from Qdrant
    7. Generate       Build a grounded prompt and stream the LLM response to the caller

When to run:
    - On deployment, and kept alive as a long-running service
    - Does NOT need to restart when new articles are ingested (Qdrant is shared)
    - Restart only when generation logic or the LLM prompt changes

Usage:
    uv run serve.py
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)