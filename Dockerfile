FROM python:3.13-slim 

# Copy uv directly from its official image — no pip needed (UV Python Package Manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# All following commands happen inside the /app folder
WORKDIR /app

# Copy dependency files + lockfile first (Docker caches this layer separately)
COPY pyproject.toml uv.lock ./

# Install exact versions from the lockfile — reproducible every time
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code (Copy only what the app needs to run)
# Notebooks, evaluation scripts, images - none of that belongs in production.
COPY api.py generate.py demo.py index.py ./
COPY images/ ./images/
COPY generation/ ./generation/
COPY indexing/ ./indexing/
COPY notebooks/evaluation/rag_eval.ipynb ./evaluation/

# Copy the SQLite database into the box (This is the parent-document retrieval data.)
# Baked directly into the image so the container has everything it needs — no external database calls for this part.
COPY notebooks/evaluation/drcd_articles.db ./database/drcd_articles.db
COPY notebooks/data/articles.db ./database/test_all_media_articles.db

EXPOSE 8080

# uv run automatically activates the virtual environment
CMD ["uv", "run", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]