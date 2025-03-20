import os, pprint, json, time
from common import *
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

start_time = time.time()  # 獲取開始時間

load_dotenv()  # 讀取.env文件

model = ChatOpenAI(model="gpt-4o-mini")

result = model.invoke("你好")
# result = model.invoke("我想去澳洲留學，給我一些建議好嗎？")
pprint.pprint(result)
# print(result.content)

print()
print(evalEndTime(start_time))
