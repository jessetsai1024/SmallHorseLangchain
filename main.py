import os, time
from common import *
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

print("=" * 100)

start_time = time.time()  # 獲取開始時間
load_dotenv()

db = Chroma(
    persist_directory = os.path.join(os.path.dirname(__file__), "db/sanguo.db"), 
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
)
model = ChatOpenAI(model="gpt-4o")
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "請根據「參考文檔」回答問題，如果在這個參考文檔中沒有找到答案，請回答“不知道”。"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    model, retriever, contextualize_q_prompt
)

system_prompt = (
    "請根據「參考文檔」回答問題，如果在這個參考文檔中沒有找到答案，請回答“不知道”。"
    "\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(model, prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

chat_history = []  # 收集聊天歷史（訊息序列）
while True:
    query = input(": ")
    if query.lower() == "exit":
        break
    print("\r> 正在檢索答案...", end="")
    result = rag_chain.invoke({"input": query, "chat_history": chat_history})
    print("\r>", f"{result['answer']}")

    chat_history.append(HumanMessage(content=query))
    chat_history.append(SystemMessage(content=result["answer"]))

print(evalEndTime(start_time))
