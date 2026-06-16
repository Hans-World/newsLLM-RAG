"""
Stage 7 — Generate
Build a grounded prompt from retrieved chunks and stream the LLM response.

Approach:
    Evidence-first prompt: show sources first, then ask the question.
    Stream tokens as they arrive so the UI can render word by word.
"""
import os
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from generation.retriever import RetrievedChunk

load_dotenv()
llm = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),  # swap provider by changing .env — no code change needed
)

def format_date(dt: datetime | None) -> str:
    if dt is None:
        return ""
    return dt.strftime("%Y-%m-%d")

def build_system_prompt() -> str:
    return """你是一個專業的新聞助理。請根據以下規則回答使用者的問題：

1. 若問題可以從參考資料中回答：
   - 在每個論點後標註對應編號，例如 [1]、[2]。
   - 回答結尾必須附上「資料來源」清單，每筆包含：標題、來源、連結、報導時間。
   - 連結與報導時間「只能」使用參考資料中實際提供的內容，
     「嚴禁」自行編造、推測、修改或從記憶中補齊任何網址與日期。
   - 若某筆資料標示為「（此筆無連結）」或「（此筆無報導時間）」，
     請在來源清單中原樣保留此標示，不得填入任何網址或日期。

2. 若問題無法從參考資料中回答，請用你的知識直接回答，
   並在開頭註明「（此回答未引用參考資料）」，且不附任何來源清單。

輸出格式範例（以下為示意引用格式，非固定句型）：
根據報導[1]，……；另有分析指出[2]，……。

資料來源：
[1] [標題A](https://example.com/article-1)  <2026-06-10> — 來源
[2] 標題B (此筆無連結)  <此筆無報導時間> — 來源
"""
    

def build_user_message(query: str, retrievedChunks: list[RetrievedChunk], parent_docs: dict) -> str:
    # deduplicate by source_id; first seen = highest score (chunks pre-sorted desc by RRF score)
    seen: dict[str, RetrievedChunk] = {}
    for rc in retrievedChunks:
        if rc.chunk.source_id not in seen:
            seen[rc.chunk.source_id] = rc
    
    # Notice: retrievedChunks has already been sorted in a descending order by score
    evidence = "\n\n".join(
        f"[{i+1}] 標題：{rc.chunk.title}\n"
        f"    來源：{rc.chunk.source}\n"
        f"    連結：{rc.chunk.url}\n"
        f"    內容：{parent_docs.get(rc.chunk.source_id, rc.chunk.text)}\n" # get the full news article using source_id from retrievedChunk
        f"    報導時間：{format_date(rc.chunk.publish_date)}"
        for i, rc in enumerate(seen.values())
    )
    return f"""=== 問題 ===
{query}

=== 參考資料 ===
{evidence}
"""


def generate(query: str, chunks: list[RetrievedChunk], parent_docs:dict, history: list[dict] | None = None):
    messages = [{"role": "system", "content": build_system_prompt()}]
    if history:
        # Both += and .extend() produce the same result
        messages += history # Prior User/Assistant turns (raw queries, no evidence)
    messages.append({"role": "user", "content": build_user_message(query, chunks, parent_docs)})
    
    response = llm.chat.completions.create(
        model=os.getenv("LLM_MODEL"),
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
        stream=True,  # stream tokens as they arrive
    )
    for chunk in response:
        token = chunk.choices[0].delta.content or "" # chunk.choices[0].delta.content is the actual text piece 
        yield token  # immediately send it to whoever called generate()
