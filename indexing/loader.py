"""
Stage 1 — Load
Read raw JSON articles and return a list of RawDocument objects.
"""
import json
from datetime import datetime


class RawDocument:
    def __init__(self, id, title, text, url, publish_date, source):
        self.id = id
        self.title = title
        self.text = text
        self.url = url
        self.publish_date = publish_date
        self.source = source


def load(path: str) -> list[RawDocument]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    docs = []
    for doc in raw:
        raw_date = doc.get("publish_date")
        docs.append(RawDocument(
            id=str(doc.get("id", "")), # Force the source_id to be string type
            title=doc.get("title", ""),
            text=doc.get("content", ""),
            url=doc.get("url", ""),
            publish_date=datetime.fromisoformat(raw_date) if raw_date else None,
            source=doc.get("source", ""),
        ))

    return docs