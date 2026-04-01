"""
Indexing Pipeline (run once, or when new articles arrive)

Responsibility:
    Transforms raw news articles into searchable vectors stored in Qdrant.
    This is an offline, one-shot process — it does not serve user requests.

Pipeline:
    1. Load      Read raw articles from data/news.json → RawDocument[]
    2. Chunk     Split each article into sentence-window chunks → Chunk[]
    3. Embed     Convert each chunk into dense + sparse vectors
    4. Store     Upsert vectors and metadata into the Qdrant collection

When to run:
    - On first deployment to build the initial index
    - When new articles are added to the data source
    - When the embedding model is changed (requires full re-index)

Usage:
    uv run ingest.py
"""