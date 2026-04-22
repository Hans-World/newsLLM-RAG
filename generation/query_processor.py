"""
Query Processor — Pre-Retrieval Query Gateway

Every user query passes through this module before embedding and retrieval.
    1. Guardrail   Reject harmful or abusive queries before any pipeline cost is incurred.
    2. Rewrite     Rewrite safe queries into keyword-rich form for better retrieval quality.
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
llm = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),  # swap provider by changing .env — no code change needed
)


def check_guardrail(query: str) -> tuple[bool, str]:
    """Return (True, "") if safe, or (False, reason) if the query should be blocked."""
    # TODO: implement guardrail logic
    pass


def rewrite_query(query: str) -> str:
    response = llm.chat.completions.create(
        model=os.getenv("LLM_MODEL"),
        messages=[{
            "role": "user",
            "content": (
                "你是一個搜尋查詢優化助理。請將以下使用者問題改寫成更精確、關鍵字豐富的搜尋查詢，"
                "以便在新聞文章資料庫中取得最佳檢索結果。只輸出改寫後的查詢，不要有任何解釋。\n\n"
                f"原始問題：{query}\n改寫查詢："
            )
        }],
        temperature=0.0,
        max_tokens=128,
        stream=False,
    )
    return response.choices[0].message.content.strip()