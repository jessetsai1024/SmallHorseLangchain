import pprint, time, json
import datetime
from common import *
from dotenv import load_dotenv

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

print("=" * 100)

start_time = time.time()  
load_dotenv()

def get_current_time(*args, **kwargs):
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

tools = [
    Tool(
        name="時間",
        description="可以獲取當前時間",
        func=get_current_time,
    ),
]

prompt_react = hub.pull("hwchase17/react")
print(prompt_react.template)
print("=" * 100)

llm = ChatOpenAI(model="gpt-4o")
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt= prompt_react,
    stop_sequence=True, 
)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)

response = agent_executor.invoke({"input": "現在幾點了？",})

pprint.pprint(response)

print(evalEndTime(start_time))