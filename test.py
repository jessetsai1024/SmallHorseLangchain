import os, time
from common import *
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

print("=" * 100)
start_time = time.time()  # 取得開始時間
load_dotenv()

chroma_dbpath = os.path.join(os.path.dirname(__file__), "db/books.db")

if not os.path.exists(chroma_dbpath):
    print(">", f"未找到儲存路徑: {chroma_dbpath}")
    exit(0)

# 定義 OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 定義 Chroma
db = Chroma(persist_directory=chroma_dbpath, embedding_function=embeddings)

# 定義 client_prompt
client_prompt = "西遊記中孫悟空的武器是什麼?"

# 定義 retriever
# - https://python.langchain.com/v0.2/docs/integrations/vectorstores/chroma/
retriever_docs = db.similarity_search(client_prompt, k=2)

print(">", "查詢文件:", len(retriever_docs))
for i, doc in enumerate(retriever_docs):
    print(f"{i+1}: {doc}")

# 列印結束時間
print(evalEndTime(start_time))
