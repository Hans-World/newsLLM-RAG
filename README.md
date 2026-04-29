# newsLLM-RAG

A RAG (Retrieval-Augmented Generation) pipeline for Chinese news articles using hybrid search (dense + sparse vectors) with Qdrant.

## Pipeline Overview

![RAG Pipeline](images/RAG_pipeline.png)

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) — Python package manager
- [Docker](https://docs.docker.com/get-docker/) — for running Qdrant

## Setup

### 1. Install Python dependencies

```bash
uv sync
```

`uv sync` reads `pyproject.toml` and installs all required Python packages into an isolated virtual environment automatically.

### 2. Start Qdrant

```bash
docker compose up -d
```

This starts a Qdrant vector database in the background on port `6333`. Data is persisted in a Docker-managed volume so it survives container restarts.

To verify Qdrant is running:
```bash
curl -s http://localhost:6333/healthz
```

### 3. Run the indexing pipeline

```bash
uv run index.py
```

This runs the full indexing pipeline:
1. Load raw news articles from `notebooks/data/news.json`
2. Chunk each article into sentence windows
3. Embed chunks into dense + sparse vectors
4. Store vectors and metadata into Qdrant

### 4. Run the demo

```bash
uv run streamlit run demo.py
```

This launches the NewsLLM conversational demo in your browser.

### 5. Stop Qdrant

```bash
docker compose down
```

## Running long jobs with tmux

For large indexing jobs that take a long time, use tmux so the process keeps running even if you close your terminal.

```bash
# 1. Start a named tmux session
tmux new -s indexing

# 2. Run the indexing pipeline inside the session
uv run index.py --data-dir /path/to/your/data.json

# 3. Detach from the session (job keeps running in background)
# Press: Ctrl+b  then  d

# 4. Come back later and reattach to check progress
tmux attach -t indexing
```

## Useful commands

```bash
# Check all Qdrant collections
curl -s http://localhost:6333/collections | python3 -m json.tool

# Check Qdrant version
curl -s http://localhost:6333

# Delete Collection e.g. news_samples
curl -X DELETE http://localhost:6333/collections/news_samples
```