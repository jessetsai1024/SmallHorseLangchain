import time
from common import *
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableSequence

start_time = time.time()  # 獲取開始時間
load_dotenv()

# 訊息陣列
print("=" * 100)
messages = [
    ("system", "你是一位{career}專家，你經常輔導你的學生。"),
    ("human", "我想學習{language}，給我幾個建議好嗎？"),
]

prompt_template = ChatPromptTemplate.from_messages(messages)

# 語言模型
model = ChatOpenAI(model="gpt-4o-mini")

# 使用Chain調用模型
# chain = prompt_template | model | StrOutputParser()
# ↓
run_prompt = RunnableLambda(lambda x: prompt_template.invoke(x))
run_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
run_output = RunnableLambda(lambda x: StrOutputParser().invoke(x))

chain = RunnableSequence(first=run_prompt, middle=[run_model], last=run_output)
result = chain.invoke({"career": "醫學", "language": "按摩"})
print(result)

# 打印結束時間
print(evalEndTime(start_time))
