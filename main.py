import os, pprint, json, time
from common import *
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

start_time = time.time()  # 獲取開始的時間
load_dotenv()

messages = [
    SystemMessage("你是一語言專家,精通英語和中文。"),
    HumanMessage("可以幫我翻譯一些英文成中文嗎?"),
    AIMessage("當然可以!請告訴我們需要翻譯的英文內容,我會盡力幫你翻譯成中文。"),
    HumanMessage("book"),
]

model = ChatOpenAI(model="gpt-4o-mini")

result = model.invoke(messages)
print(result.content)

print()
print(evalEndTime(start_time))