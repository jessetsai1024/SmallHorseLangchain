from langchain.prompts import ChatPromptTemplate

# 單參數
print("=" * 100)
template = "我想學習{language}語言，給我幾個開發框架好嗎?"
prompt_template = ChatPromptTemplate.from_template(template)

prompt = prompt_template.invoke({"language": "Python"})
print(prompt)
prompt = prompt_template.invoke({"language": "Java"})
print(prompt)
prompt = prompt_template.invoke({"language": "PHP"})
print(prompt)

# 多參數
print("=" * 100)
template = "我想學習{language}語言，給我幾個開發{target}好嗎?"
prompt_template = ChatPromptTemplate.from_template(template)

prompt = prompt_template.invoke({"language": "Python", "target": "框架"})
print(prompt)
prompt = prompt_template.invoke({"language": "Python", "target": "例子"})
print(prompt)

# 訊息陣列
print("=" * 100)
messages = [
    ("system", "你是一位{career}專家，你經常輔導你的學生。"),
    ("human", "我想學習{language}，給我幾個建議好嗎?"),
]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"career": "IT", "language": "Python"})
print(prompt)
prompt = prompt_template.invoke({"career": "醫學", "language": "按摩"})
print(prompt)

import time
from common import *
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate

start_time = time.time()  
load_dotenv()

from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o-mini")

result = model.invoke(prompt)
print(result.content)

# 打印结束时间
print(evalEndTime(start_time))
