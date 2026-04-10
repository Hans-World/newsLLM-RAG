"""
Stage 2 — Chunk
Split each RawDocument into overlapping sentence-window chunks.

Each chunk preserves the source metadata (title, url, publish_date)
so citations remain traceable through retrieval and generation.
"""
import re
from indexing.loader import RawDocument

class Chunk:
    def __init__(self, source_id, chunk_id, text, title, url, publish_date, source):
        self.source_id = source_id
        self.chunk_id = chunk_id
        self.text = text
        self.title = title
        self.url = url
        self.publish_date = publish_date
        self.source = source
        
        
def split_RawDocument_to_sentences(doc:RawDocument):
    # Split on Chinese punctuation
    sentences = re.findall(r".+?[。！？]」?|.+?」", doc.text, re.DOTALL)
    cleaned_sentences = []
    for s in sentences: 
        s = s.strip()      # remove leading/trailing whitespace and newlines
        if s:   # only keep it if it's not empty after stripping
            cleaned_sentences.append(s)
    return cleaned_sentences
        
        
def sliding_window(sentence_lst, max_sentences, overlap_sentences):
    chunk_lst = [] 
    i = 0
    step = max_sentences - overlap_sentences # how far to advance each time
    while i < len(sentence_lst):
        window = sentence_lst[i: i + max_sentences] # take up to max_sentences cards
        chunk_lst.append("".join(window))
        i += step
    return chunk_lst
        
        
def chunk(doc:RawDocument, max_sentences=3, overlap_sentences=1): 
    """
    Sentence boundaries in Chinese natrually align with meaning boundaries.
    Approach: 
        Chunk by sentences
    Method: 
        Slide a window of max_sentences over the sentence list.
        Each step moves forward by (max_sentences - overlap_sentences), 
        so that last overlap_sentences cards are always carried into the next chunk.
    """
    sentence_lst = split_RawDocument_to_sentences(doc)
    chunk_lst = sliding_window(sentence_lst, max_sentences, overlap_sentences)
    
    chunks = []
    for i, text in enumerate(chunk_lst):
        chunks.append(Chunk(
            source_id=doc.id,
            chunk_id=f"{doc.id}_{i}",
            text=text,
            title=doc.title,
            url=doc.url,
            publish_date=doc.publish_date,
            source=doc.source,
        ))
    
    return chunks