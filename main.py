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
    ("human", "我想學習{language}語言，請用英語回答我，並且控制在35個單字以內。"),
]
prompt_template = ChatPromptTemplate.from_messages(messages)
# 大語言模型
model = ChatOpenAI(model="gpt-4o-mini")
# 使用Chain呼叫模型
run_prompt = RunnableLambda(lambda x: prompt_template.invoke(x))
run_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
run_output = RunnableLambda(lambda x: StrOutputParser().invoke(x))
run_uppercase = RunnableLambda(lambda x: x.upper())  # 將輸出轉換為大寫
# run_countwords = RunnableLambda(lambda x: f"{x}\n共{len(x.split())}詞")  # 計算輸出的單字數
run_countwords = RunnableLambda(lambda x: print(x) or len(x))  # 計算輸出的單字數

# chain = RunnableSequence(first=run_prompt, middle=[run_model], last=run_output)
# ↓
# chain = run_prompt | run_model | run_output
chain = run_prompt | run_model | run_output | run_uppercase | run_countwords
result = chain.invoke({"language": "python"})
print(result)

# 印出總耗時
print(time.time() - start_time)
