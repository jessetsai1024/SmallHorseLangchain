import os
import time

def printENV():
    # 取得所有環境變量
    env_vars = os.environ
    # 打印所有環境變量
    for key, value in env_vars.items():
        if key in ["OPENAI_API_KEY", "TAVILY_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"]:
            print(f"{key}: {value}")

def evalEndTime(start_time):
    end_time = time.time()  # 获取结束时间
    execution_time = "(程序运行时间:%.2f 秒)" % (
        end_time - start_time
    )  # 计算程序运行时间
    return execution_time
