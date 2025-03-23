import os, time
from common import *
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

print("=" * 100)

# 定義client_prompt
# client_prompt = "請問桃園結義是幾個人？都是誰？"
# client_prompt = "虎牢關是誰打敗了呂布嗎？"
client_prompt = "孔明是怎麼罵死曹真的？把內封信的原文找出來"
# client_prompt = "劉備三顧茅廬請出的人是誰？"
# client_prompt = "川普是哪一國總統?"

start_time = time.time()  # 獲取開始時間
load_dotenv()

chroma_dbpath = os.path.join(os.path.dirname(__file__), "db/sanguo.db")

if not os.path.exists(chroma_dbpath):
    print(">", f"未找到儲存路徑:{chroma_dbpath}")
    exit(0)

# 定義OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 定義Chroma
db = Chroma(persist_directory=chroma_dbpath, embedding_function=embeddings)

# 定義retriever
# - https://python.langchain.com/v0.2/docs/integrations/vectorstores/chroma/
# retriever_docs = db.similarity_search_with_score(client_prompt)
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

retriever_docs = retriever.invoke(client_prompt)

print(">", "查詢文件:", len(retriever_docs))
# for i, doc in enumerate(retriever_docs):
#     print(f"{i+1}. {doc.page_content}")

if len(retriever_docs) == 0:
    print(">", "未找到相關文件")
    exit(0)

human_prompt = """請根據提供的 `參考文件` 回答下面的問題：

問題: {client_prompt}

參考文件：
\"\"\"
{reference_docs}
\"\"\"

請根據 `參考文件` 回答問題，如果在這個參考文件中沒有找到答案，請回答“不知道”。
""".format(
    client_prompt=client_prompt,
    reference_docs="\n".join([doc.page_content for doc in retriever_docs]),
)
print(human_prompt)

messages = [
    SystemMessage(
        "請嚴格按照提供的參考文件回答使用者的問題，不要引用參考文件之外的內容。"
    ),
    HumanMessage(human_prompt),
]

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

print("==========================================================")
result = model.invoke(messages)
print(result.content)

print(evalEndTime(start_time))