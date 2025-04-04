import os, pprint, json, time
from common import *
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

print("=" * 100)
start_time = time.time()    # 獲取開始時間
load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.5,
)
structured_llm = llm.with_structured_output(None, method="json_mode")

prompt = ChatPromptTemplate.from_template(
    """Question: {question}

Instructions: 使用json模式輸出，項目數組輸出到items[str]字段
Answer:
"""
)

result = structured_llm.invoke(prompt.invoke({"question": "告訴我5個奧運會的項目"}))
# result = structured_llm.invoke(prompt.invoke({"question": "告訴我5個Python的特點"}))
# result = structured_llm.invoke(prompt.invoke({"question": "告訴我5個日本旅遊的景點"}))
pprint.pprint(result)
# pprint.pprint(result["items"])

print(evalEndTime(start_time))
