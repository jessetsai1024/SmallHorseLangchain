import os, pprint, json, time
from common import *
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

start_time = time.time()  # 獲取開始時間
load_dotenv()

messages = [
    SystemMessage("你是一語言專家，精通英語和中文。所有回答請限制在35個字以內。"),
]
model = ChatOpenAI(model="gpt-4o-mini")
while True:
    user_input = input("> ")
    if user_input.lower() == "exit":
        break
    elif len(user_input.strip()) == 0:
        continue

    messages.append(HumanMessage(user_input))
    result = model.invoke(messages)
    messages.append(AIMessage(result.content))
    print("AI:", result.content)

print(evalEndTime(start_time))