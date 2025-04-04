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
start_time = time.time()     # 獲取開始時間
load_dotenv()

class SimpleSearchToolArgs(BaseModel):
    query: str = Field(..., title="Query string")

class SimpleSearchTool(BaseTool):
    name = "simple_search"
    description = "A simple search tool"
    args_schema: Type[BaseModel] = SimpleSearchToolArgs

    def _run(self, query: str):
        """
        Run the search tool
        執行搜尋工具
        """
        from tavily import TavilyClient

        # client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        client = TavilyClient(api_key="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")
        results = client.search(query)
        return results
        # return f"search results for {query}\n\n{results}\n"

tools = [SimpleSearchTool()]

# pprint.pprint(tools[0]._run("python"))
# pprint.pprint(tools[0]._run("我想找一些Python的工作"))

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

############################################################
# 呼叫Agent
# response = agent_executor.invoke({"input": "我想找一些Python的工作"})
response = agent_executor.invoke({"input": "今天東京的天氣如何？"})
# response = agent_executor.invoke({"input": "1+1=?"})
print(response["output"])

############################################################
# 打印結束時間
print("\n", evalEndTime(start_time))
