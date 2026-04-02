"""
Stage 1 — Load
Read raw JSON articles and return a list of RawDocument objects.
"""
import json
from datetime import datetime


class RawDocument:
    def __init__(self, id, title, text, url, publish_date):
        self.id = id
        self.title = title
        self.text = text
        self.url = url
        self.publish_date = publish_date


def load(path: str) -> list[RawDocument]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    docs = []
    for doc in raw:
        docs.append(RawDocument(
            id=doc["id"],
            title=doc["title"],
            text=doc["content"],
            url=doc["url"],
            publish_date=datetime.fromisoformat(doc["publish_date"]),
        ))

    return docs