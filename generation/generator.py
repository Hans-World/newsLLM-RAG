"""
Stage 7 — Generate
Build a grounded prompt from retrieved chunks and stream the LLM response.

Approach:
    Evidence-first prompt: show sources first, then ask the question.
    Stream tokens as they arrive so the UI can render word by word.
"""
import os
from openai import OpenAI
from dotenv import load_dotenv
from generation.retriever import RetrievedChunk

load_dotenv()
llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def build_prompt(query: str, retrievedChunks: list[RetrievedChunk]) -> str:
    # Notice: retrievedChunks has already been sorted in a descending order by score
    evidence = "\n\n".join(
        f"[{i+1}] 標題：{rc.chunk.title}\n"
        f"    來源：{rc.chunk.url}\n"
        f"    內容：{rc.chunk.text}\n"
        f"    報導時間：{rc.chunk.publish_date}"
        for i, rc in enumerate(retrievedChunks)
    )
    return f"""你是一個可信新聞助理。請根據以下參考資料回答問題。若資料不足請說明無法確認。

=== 參考資料 ===
{evidence}

=== 問題 ===
{query}

=== 回答 ===
[標題]: URL來源
生成答案 """


def generate(query: str, chunks: list[RetrievedChunk]):
    prompt = build_prompt(query, chunks)
    response = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512,
        stream=True,  # stream tokens as they arrive
    )
    for chunk in response:
        token = chunk.choices[0].delta.content or ""
        yield token  # yield each token to StreamingResponse
