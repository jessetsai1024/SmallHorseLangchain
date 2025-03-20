import os, pprint, json, time
from common import *
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

start_time = time.time()  # 獲取開始時間
load_dotenv()

messages = [
    SystemMessage("你是一位幽默大師,你的回答經常會讓客戶捧腹大笑。"),
    # HumanMessage("你好"),
    HumanMessage("我想去澳洲留學,給我一些建議好嗎?"),
]

model = ChatOpenAI(model="gpt-4o-mini")
result = model.invoke(messages)
print(result.content)

print()
print(evalEndTime(start_time))