import os, time
from common import *
from dotenv import load_dotenv

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

start_time = time.time()  # 獲取開始時間
load_dotenv()

book_files = [
    os.path.join(os.path.dirname(__file__), "data/books/sanguo.txt"),
    os.path.join(os.path.dirname(__file__), "data/books/seeyou.txt"),
    os.path.join(os.path.dirname(__file__), "data/books/watertiger.txt"),
]
print(book_files)

chroma_dbpath = os.path.join(os.path.dirname(__file__), "db/books.db")

if os.path.exists(chroma_dbpath):
    print(f"Chroma DB already exists at {chroma_dbpath}")
    exit(0)

for file in book_files:
    if not os.path.exists(file):
        print(f"Text file not found at {file}")
        exit(0)

titles = ["三國演義", "西遊記", "水滸傳"]
authors = ["羅貫中", "吳承恩", "施耐庵"]
i : int = 0
documents = []
for file_path in book_files:
    loader = TextLoader(file_path)
    book_docs = loader.load()
    print(">", file_path, len(book_docs))
    for doc in book_docs:
        doc.metadata = {
            "Source": file_path,
            "Title": titles[i],
            "Author": authors[i],
        }
        documents.append(doc)
    i += 1

# Split the text into characters
print(">", "文本分割中...")
text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
docs = text_splitter.split_documents(documents)

print(">", f"文本分割為 {len(docs)} 個文檔")

# Create embeddings
# - https://platform.openai.com/docs/guides/embeddings
print(">", "創建 OpenAI Embeddings...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# # 建立 Chroma 向量儲存庫
# print(">", "建立 Chroma 向量儲存中...")
# db = Chroma.from_documents(docs, embeddings, persist_directory=chroma_dbpath)

db = Chroma(
    collection_name="books_collection",
    embedding_function=embeddings,
    persist_directory=chroma_dbpath
)

# 5. 逐批 (batch) 加入到 db
BATCH_SIZE = 300  # 依實際情況調整
for start_idx in range(0, len(docs), BATCH_SIZE):
    end_idx = start_idx + BATCH_SIZE
    print(">", f"Adding batch {start_idx} to {end_idx}")
    batch_docs = docs[start_idx:end_idx]
    
    # 取出該批次的文本及 metadata
    texts = [doc.page_content for doc in batch_docs]
    metadatas = [doc.metadata for doc in batch_docs]
    
    # 加入資料到 Vector Store
    db.add_texts(texts=texts, metadatas=metadatas)
    
    # 如果怕一次打太多 API，或擔心觸發速率限制 (Rate Limit)，可以 sleep 幾秒
    time.sleep(1)  # 依實際需要看要不要加、或加多少

# 6. 最後存檔
db.persist()

print(">", "Chroma 向量儲存建立完成")
print(">", f"Chroma 向量儲存路徑: {chroma_dbpath}")
print(">", f"Chroma 向量儲存文件數量: {len(docs)}")
print(">", f"Chroma 向量儲存文件長度: {len(db)}")

# 印出結束時間
print(evalEndTime(start_time))
