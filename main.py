import os, pprint, json, time
from common import *
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import Optional
from langchain_core.pydantic_v1 import BaseModel, Field

print("=" * 100)
start_time = time.time()    # 獲取開始時間
load_dotenv()

# Pydantic
class Result(BaseModel):
    items: list[str] = Field(..., title="項目", description="項目列表")
    answer: str = Field(
        ..., title="最終回答", description="根據項目列表，一句話總結回答問題"
    )

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
)

structured_llm = llm.with_structured_output(Result)

result = structured_llm.invoke("告訴我5個奧運會的項目")
# result = structured_llm.invoke("告訴我5個學習Python的步驟")
# result = structured_llm.invoke("告訴我5個日本旅遊的景點")

print(result)
print("-" * 100)
print("回答項目：", result.items)
print("-" * 100)
print("最終回答：", result.answer)
print("-" * 100)

print(evalEndTime(start_time))
