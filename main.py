import time
from common import *
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser  # 字串輸出解析器

start_time = time.time()  # 獲取開始的時間
load_dotenv()
print("=" * 100)

# 提示詞模板
messages = [
    ("system", "你是一位{career}專家,你經常輔導你的學生。"),
    ("human", "我想學習{language},給我幾個建議好嗎?"),
]
prompt_template = ChatPromptTemplate.from_messages(messages)

# 大語言模型
model = ChatOpenAI(model="gpt-4o-mini")

# 鏈重點：使用Chain調用模型
chain = prompt_template | model | StrOutputParser()
result = chain.invoke({"career": "醫學", "language": "按摩"})
print(result)

# 打印結束時間
print(evalEndTime(start_time))