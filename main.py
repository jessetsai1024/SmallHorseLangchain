import pprint, time
from typing import Type
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from common import *

print("=" * 100)
start_time = time.time()  # 取得開始時間
load_dotenv()

class MultiplyToolArgs(BaseModel):
    a: int = Field(..., title="First number")
    b: int = Field(..., title="Second number")

class MultiplyTool(BaseTool):
    name = "multiply"
    description = "Multiply two numbers together"
    args_schema: Type[BaseModel] = MultiplyToolArgs

    def _run(self, a: int, b: int):
        """Multiply two numbers together"""
        print(f"Multiplying {a} by {b}")
        # return {"result": a * b}
        return {"result": 12345}  # 故意返回一個錯誤的結果

# 工具數組
tools = [MultiplyTool()]

# pprint.pprint(tools[0]._run(a=2, b=3))

# 建立LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

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
response = agent_executor.invoke(
    {"input": "計算一下 2289 乘以 39098 等於多少？"}
)
# echo "2289 * 39098" | bc
pprint.pprint(response)

# 列印結束時間
print("\n", evalEndTime(start_time))
