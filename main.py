import time
from common import *
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel

start_time = time.time()  # 獲取開始的時間
load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

print("=" * 100)
prompt_input_chain = RunnableLambda(
    lambda input: "編程語言: {language}".format(language=input["language"])
)

def prompt_merit(language): # 優點提示詞
    messages = [
        ("system", "你是一位經驗豐富的編程高手。"),
        ("human", "我想學習{language}，請在100個單詞以內描述優點。"),
    ]
    prompt_template = ChatPromptTemplate.from_messages(messages)
    return prompt_template.format_prompt(language=language)

merit_branch_chain = ( # 優點分支鏈
    RunnableLambda(lambda x: prompt_merit(x)) | model | StrOutputParser()
)

def prompt_demerit(language): # 缺點提示詞
    messages = [
        ("system", "你是一位經驗豐富的編程高手。"),
        ("human", "我想學習{language}，請在100個單詞以內描述缺點。"),
    ]
    prompt_template = ChatPromptTemplate.from_messages(messages)
    return prompt_template.format_prompt(language=language)

demerit_branch_chain = ( # 缺點分支鏈
    RunnableLambda(lambda x: prompt_demerit(x)) | model | StrOutputParser()
)

def combine_merit_demerit(branch_outputs): # 合併分支輸出結果
    merit = branch_outputs["merit"]
    demerit = branch_outputs["demerit"]
    return f"[優點]:\n{merit}\n\n[缺點]:\n{demerit}"

combine_merit_demerit_chain = RunnableLambda( # 合併分支輸出結果 Chain 物件
    lambda x: combine_merit_demerit(x["branches"])
)

def output(x): # 僅作為中間過程的輸出
    print(">", x)
    return x

output_chain = RunnableLambda(lambda x: output(x)) # 輸出中間結果（日誌用）

# 使用 Chain 呼叫模型
chain = (
    prompt_input_chain |
    output_chain |
    RunnableParallel(
        branches={"merit": merit_branch_chain, "demerit": demerit_branch_chain}
    ) |
    combine_merit_demerit_chain |
    StrOutputParser()
)

result = chain.invoke({"language": "python"})
# result = chain.invoke({"language": "java"})
print(result)

# 列印結束時間
print(evalEndTime(start_time))
