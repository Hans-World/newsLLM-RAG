"""
Stage 1.5 - Store Parent Document
Save full articles to SQLite so they can be fetched after retrieval for parent-document retrieval
"""
import sqlite3, os 
from pathlib import Path
from indexing.loader import RawDocument
from dotenv import load_dotenv
from tqdm import tqdm

# Anchor the .db file to the project root regardless of where you run the script from
_PROJECT_ROOT = Path(__file__).parent.parent # Project Root

# os.getenv(key, default) checks for an environment variable first, and falss back to a default if it's not set.
load_dotenv() # Read the .env file 
# DB_PATH = os.getenv("ARTICLE_DB_PATH", str(_PROJECT_ROOT / "notebooks" / "data" / "articles.db"))

def _connect() -> sqlite3.Connection:
    """
    One place to change connection settings if needed later
    This creates the .db file on disk if it doesn't exist yet
    """
    db_path = os.getenv("ARTICLE_DB_PATH", str(_PROJECT_ROOT / "notebooks" / "data" / "articles.db"))
    return sqlite3.connect(db_path)

def init_db():
    # open connection, auto-commit on exit
    with _connect() as conn:
        # PRAGMA are SQLite-specific config knobs, not standard SQL.
        conn.execute("PRAGMA journal_mode=WAL") # WAL (Write-Ahead Log): allows reads while a write is happening
        conn.execute("PRAGMA synchronous=NORMAL")
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                source_id  TEXT PRIMARY KEY,
                text       TEXT
            )
        """)
        
        # INDEX on source_id: without this, fetch_articles() scans all rows
        # With it, SQLite jumps directly to the metching row
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_source_id ON articles (source_id)
        """)
        
def save_articles(docs: list[RawDocument]):
    BATCH_SIZE = 1000
    # open connection, auto-commit on exit
    with _connect() as conn:
        for i in tqdm(range(0, len(docs), BATCH_SIZE), desc="Saving articles", unit="batch"):
            batch = docs[i: i + BATCH_SIZE]
            # Safe to re-run the indexing pipeline without creating duplicates
            conn.executemany(
                "INSERT OR REPLACE INTO articles (source_id, text) VALUES (?, ?)",
                [(doc.id, doc.text) for doc in batch]
            )
            
def fetch_articles(source_ids: list[str]):
    """
    Look up full parent articles from SQLite using source_ids retrieved from Qdrant
    
    Args: 
        source_ids: list of source id from retrieved chunks 
    Returns:
        dict mapping source_id to full article text
    """
    placeholders = ",".join("?" * len(source_ids))
    # print(source_ids)
    # print(placeholders)
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT source_id, text FROM articles WHERE source_id IN ({placeholders})",
            source_ids
        ).fetchall()
    # for r in rows:
    #     print(r)
    return {row[0]: row[1] for row in rows} # converts a list of tuples into a dictionary {source_id: parent article}