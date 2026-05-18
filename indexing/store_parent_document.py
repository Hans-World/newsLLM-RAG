"""
Stage 1.5 - Store Parent Document
Save full articles to SQLite so they can be fetched after retrieval for parent-document retrieval
"""
import sqlite3, os 
from pathlib import Path
from indexing.loader import RawDocument

# Anchor the .db file to the project root regardless of where you run the script from
_PROJECT_ROOT = Path(__file__).parent.parent # Project Root

DB_PATH = os.getenv("ARTICLE_DB_PATH", str(_PROJECT_ROOT / "notebooks" / "data" / "articles.db"))

def _connect() -> sqlite3.Connection:
    """
    One place to change connection settings if needed later
    This creates the .db file on disk if it doesn't exist yet
    """
    return sqlite3.connect(DB_PATH)

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
    # open connection, auto-commit on exit
    with _connect() as conn:
        # Safe to re-run the indexing pipeline without creating duplicates
        conn.executemany(
            "INSERT OR REPLACE INTO articles (source_id, text) VALUES (?, ?)",
            [(doc.id, doc.text) for doc in docs]
        )
        
def fetch_articles():
    """
    
    """
    return