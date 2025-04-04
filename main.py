import time
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import StructuredTool
from common import *

print("=" * 100)

start_time = time.time()  # 取得開始時間

class SayHelloInput(BaseModel):
    name: str = Field(..., title="The name of the person to say hello to")

def say_hello(name: str) -> str:
    """
    Say hello to a person
    """
    return f"親愛的{name}，你好！"

class ReverseStringInput(BaseModel):
    content: str = Field(..., title="The string to reverse")

def reverse_string(content: str) -> str:
    """
    Reverse a string
    """
    return content[::-1]

class ConcatenateStringsInput(BaseModel):
    a: str = Field(..., title="The first string")
    b: str = Field(..., title="The second string")

def concatenate_strings(a: str, b: str) -> str:
    """
    Concatenate two strings
    """
    return a + b

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

# 直接調用工具
# print(tools[0].func("Koma"))
# print(tools[1].func("Hello"))
# print(tools[2].func("Hello", "World"))

print(tools[0].invoke("Koma"))
print(tools[1].invoke("Hello"))
print(tools[2].invoke({"a": "Hello", "b": "World"}))

############################################################
# 打印結束時間
print("\n", evalEndTime(start_time))
