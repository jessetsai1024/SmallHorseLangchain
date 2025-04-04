import pprint, time
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from common import *

print("=" * 100)

start_time = time.time()  # 取得開始時間
load_dotenv()

@tool()
def say_hello(name: str) -> str:
    """
    Say hello to a person
    """
    return f"親愛的{name}，你好！"

class ReverseStringInput(BaseModel):
    content: str = Field(..., title="The string to reverse")

@tool(args_schema=ReverseStringInput)
def reverse_string(content: str) -> str:
    """
    Reverse a string
    """
    return content[::-1]

tools = [
    say_hello,
    reverse_string,
]

# 建立LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ReAct提示詞
prompt = hub.pull("hwchase17/openai-tools-agent")

# 建立Agent
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

# 建立Agent執行器
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)

##############################################

# 呼叫Agent
response = agent_executor.invoke({"input": "我是小馬"})
pprint.pprint(response)

response = agent_executor.invoke({"input": "我想翻轉一下這個字串：Youtube"})
pprint.pprint(response)

##############################################

# 列印結束時間
print("\n", evalEndTime(start_time))
