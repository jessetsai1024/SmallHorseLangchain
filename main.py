import os, time
from common import *
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

print("=" * 100)
start_time = time.time() # 記錄開始時間
load_dotenv()

chroma_dbpath = os.path.join(os.path.dirname(__file__), "db/sanguo.db")
if not os.path.exists(chroma_dbpath):
   print("找不到存儲路徑:", chroma_dbpath)
   exit(0)

# 定義 OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 定義 Chroma
db = Chroma(persist_directory=chroma_dbpath, embedding_function=embeddings)

# 定義客戶端提示
client_prompt = "孔明寫信罵死曹真, 那封信裡面有一首很長的詩"

# 定義檢索器
# - https://python.langchain.com/v2.0/docs/integrations/vectorstores/chroma/
retriever_docs = db.similarity_search(client_prompt, k=2)

print("查詢文檔:", len(retriever_docs))
for i, doc in enumerate(retriever_docs):
   print(f"{i+1}. {doc}")

# 打印結束時間
print(f"耗時: {time.time() - start_time}秒")