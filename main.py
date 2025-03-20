from langchain.schema.runnable import RunnableLambda

# 字母大寫
def to_uppercase(inputs):
    return inputs.upper()

# 前後加括號
def append_comma(inputs):
    return f"[[{inputs}]]"

to_uppercase_lambda = RunnableLambda(to_uppercase)
append_comma_lambda = RunnableLambda(append_comma)

chain = to_uppercase_lambda | append_comma_lambda
print(chain.invoke("hello world"))