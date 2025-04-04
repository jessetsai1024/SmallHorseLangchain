import pprint, time
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from common import *

print("=" * 100)

start_time = time.time()  # 取得開始時間
load_dotenv()

def say_hello(name: str) -> str:
    """
    Say hello to a person
    """
    return f"親愛的{name}，你好！"

class SayHelloInput(BaseModel):
    name: str = Field(..., title="The name of the person to say hello to")

def reverse_string(content: str) -> str:
    """
    Reverse a string
    """
    return content[::-1]

class ReverseStringInput(BaseModel):
    content: str = Field(..., title="The string to reverse")

def concatenate_strings(a: str, b: str) -> str:
    """
    Concatenate two strings
    """
    return a + b

class ConcatenateStringsInput(BaseModel):
    a: str = Field(..., title="The first string")
    b: str = Field(..., title="The second string")

tools = [
    StructuredTool.from_function(
        func=say_hello,
        args_schema=SayHelloInput,
        output_field_name="greeting",
        description="Say hello to a person",
    ),
    StructuredTool.from_function(
        func=reverse_string,
        args_schema=ReverseStringInput,
        output_field_name="reversed_string",
        description="Reverse a string",
    ),
    StructuredTool.from_function(
        func=concatenate_strings,
        args_schema=ConcatenateStringsInput,
        output_field_name="result_string",
        description="Concatenate two strings",
    ),
]

# 建立LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ReAct提示詞
prompt = hub.pull("hwchase17/openai-tools-agent")
# prompt.pretty_print()

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

############################################################

# 調用Agent
response = agent_executor.invoke({"input": "我是小馬"})
pprint.pprint(response)

response = agent_executor.invoke({"input": "我想翻轉一下這個字串：Hello"})
pprint.pprint(response)

response = agent_executor.invoke(
    {
        "input": """
我想把這兩個字串連接起來：
Hello
World
"""
    }
)
pprint.pprint(response)
