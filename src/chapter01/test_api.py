# test_api.py
from config import *
from openai import OpenAI

# 必填：从服务管控页面获取对应服务的APIKey和API Base
api_key = API_KEY
base_url = BASE_URL

client = OpenAI(api_key=api_key, base_url=base_url)
response = client.chat.completions.create(
    model=MODEL_ID,  # 模型名称，必填项
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello, Tell me which model support you?"},
    ],
    stream=False    # stream=False 非流式（一次性返回）、stream=True 流式（实时返回）
)

print(response.choices[0].message.content)