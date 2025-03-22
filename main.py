import time
from common import *
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableBranch

print("=" * 100)
start_time = time.time()  # 取得開始時間
load_dotenv()

remove_spaces = lambda x: x.replace(" ", "") # 移除字串中的所有空格

# 客戶輸入
# client_prompt = "這飯真難吃"
# client_prompt = "好累啊"
client_prompt = "今天心情不錯"
# client_prompt = "你是誰"
# client_prompt = "明天什麼天氣？"
print(client_prompt, "\n")

# 建立模型
model = ChatOpenAI(model="gpt-4o-mini")

prompt_tpl = """
指令：請判斷下面的`文本`是正面、負面還是中性的。

文本：這東西真難吃  
結果：負面

文本：路太遠啦  
結果：負面

文本：今天心情不錯  
結果：正面

文本：時間太長了  
結果：負面

文本：我們出發吧  
結果：中性

文本：早睡早起  
結果：中性

文本：{content}
"""

# 定義一位客服AI
def customer_service_ai(content):
    messages = [
        (
            "system",
            """你是一位客服人員，你需要判斷客戶的情緒是正面、負面還是中性。"""
        ),
        ("human", prompt_tpl.format(content=content)),
    ]
    prompt_template = ChatPromptTemplate.from_messages(messages)
    return prompt_template.format_prompt(content=content)

prompt_input_chain = RunnableLambda(lambda input: customer_service_ai(input))

def emotion_parser(emotion): # 整理情緒
    # print(">>", emotion)
    if "結果：正面" in remove_spaces(emotion):
        return "正面"
    elif "結果：負面" in remove_spaces(emotion):
        return "負面"
    else:
        return "中性"

emotion_parser_chain = RunnableLambda(lambda x: emotion_parser(x))

positive_chain = RunnableLambda(lambda x: "[ACTION] 正面處理 (Positive)")
negative_chain = RunnableLambda(lambda x: "[ACTION] 負面處理 (Negative)")
neutral_chain = RunnableLambda(lambda x: "[ACTION] 中性處理 (Neutral)")

branches = RunnableBranch( # 分支思考
    (lambda x: x == "正面", positive_chain),
    (lambda x: x == "負面", negative_chain),
    neutral_chain,
)

def output(x): # 輸出過程中的數據
    print(">", x)
    return x

output_chain = RunnableLambda(lambda x: output(x))

# 定義 Chain
chain = (
    prompt_input_chain
    | model
    | StrOutputParser()
    | output_chain
    | emotion_parser_chain
    | output_chain
    | branches
    | output_chain
)

result = chain.invoke(client_prompt)
print()

# 打印結束時間
print(evalEndTime(start_time))