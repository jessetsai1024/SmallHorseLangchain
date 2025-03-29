import pprint, time, json
import datetime
from common import *
from dotenv import load_dotenv

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

print("=" * 100)
start_time = time.time()  # 獲取開始時間
load_dotenv()

def get_date(*args, **kwargs):
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d")

def get_weather(*args, **kwargs):
    return "晴時多雲偶陣雨"

tools = [
    Tool(
        name = "獲取日期",
        description = "可以獲取客戶需要的日期",
        func = get_date,
    ),
    Tool(
        name = "天氣",
        description = "需要指定日期，然後可以獲取天氣",
        func = get_weather,
    ),
]

prompt_react = hub.pull("hwchase17/react")
# print(prompt_react.template)

llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt_react,
    stop_sequence=True,
)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)

response = agent_executor.invoke({"input": "今天的天氣如何？"})

pprint.pprint(response)

# 印出結束時間
print(evalEndTime(start_time))
