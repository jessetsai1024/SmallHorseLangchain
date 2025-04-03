import pprint, time, json
import datetime
import wikipedia
from common import *
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool, StructuredTool
from langchain_openai import ChatOpenAI
from langchain.pydantic_v1 import BaseModel, Field # from pydantic import BaseModel, Field

print("=" * 100)

start_time = time.time()  # 獲取開始時間
load_dotenv()

def get_date(*args, **kwargs):
    now = datetime.datetime.now()
    return "日期是：" + now.strftime("%Y-%m-%d")

def search_wikipedia(query):
    try:
        wikipedia.set_lang("zh-tw")
        return wikipedia.summary(query, sentences=2)
    except:
        return "沒有找到相關信息"
    
# print(search_wikipedia("麻瓜"))

class WikipediaSearchSchema(BaseModel):
    query: str = Field(..., description="要搜尋的維基百科主題")

tools = [
    StructuredTool.from_function(
        get_date,
        name="當前日期",
        description="不需要傳遞任何參數，就可以獲取當前日期",
        return_direct=True,
    ),
    StructuredTool.from_function(
        search_wikipedia,
        name="Wikipedia",
        description="當需要專題資訊時很有用，可以從維基百科中獲取信息",
        args_schema=WikipediaSearchSchema,
        return_direct=True,
    ),
]

prompt = hub.pull("hwchase17/structured-chat-agent")
# prompt.pretty_print()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=1)
memory = ConversationBufferMemory(
    memory_key="chat_history",
    max_buffer_size=5,
    max_message_length=100,
    max_message_count=100,
    return_messages=True,
)
agent = create_structured_chat_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    stop_sequence=True,
)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
)

memory.chat_memory.add_message(SystemMessage(
    content="你是一個AI助手，可以使用可用的工具提供有用的答案。如果你無法回答，可以使用以下工具：當前日期 和 Wikipedia"
))

while True:
    user_input = input(": ")
    if user_input.lower() == "exit":
        break
    elif len(user_input.strip()) == 0:
        continue

    memory.chat_memory.add_message(HumanMessage(content=user_input))
    response = agent_executor.invoke({"input": user_input})
    answer = response["output"]
    print(">", answer)
    memory.chat_memory.add_message(AIMessage(content=answer))

# 打印結束時間
print(evalEndTime(start_time))
