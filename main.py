import os, time
from common import *
from dotenv import load_dotenv

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

start_time = time.time()  # 我們開始時間
load_dotenv()

sanguo_txt = os.path.join(os.path.dirname(__file__), "data/sanguo.txt")
chroma_dbpath = os.path.join(os.path.dirname(__file__), "db/sanguo.db")

if os.path.exists(chroma_dbpath):
    print(f"Chroma DB 已存在於 {chroma_dbpath}")
    exit(0)

if not os.path.exists(sanguo_txt):
    print(f"Text file not found at {sanguo_txt}")
    exit(0)

# 載入文本
loader = TextLoader(sanguo_txt)
documents = loader.load()

# 將文本分割為字符
print("➡️ 文本分割中...")
# text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
docs = text_splitter.split_documents(documents)

print(f"⚙️ 文本分割為 {len(docs)} 個文件")

# 建立嵌入向量
# https://platform.openai.com/docs/guides/embeddings
print("🧠 建立 OpenAI 嵌入向量...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 建立 Chroma 向量資料庫
print("💾 建立 Chroma 向量資料庫...")
db = Chroma.from_documents(docs, embeddings, persist_directory=chroma_dbpath)

print("✅ Chroma 向量資料庫已建立完成")
print(f"📂 Chroma 向量資料庫儲存路徑: {chroma_dbpath}")
print(f"📄 Chroma 向量資料庫文件數量: {len(docs)}")
print(f"🧠 Chroma 向量資料庫索引數量: {len(db)}")

# 印出總花費時間
print(evalEndTime(start_time))
