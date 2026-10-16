from dotenv import load_dotenv
import os
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 1. 先列出所有 flash 模型，确认准确名字
print("=== 可用的 flash 模型 ===")
for m in client.models.list():
    if "flash" in m.name.lower():
        print(" ", m.name)

# 2. 试调一次
print("\n=== 试调 ===")
try:
    resp = client.models.generate_content(
        model="gemini-3.5-flash",
        contents="用一句话介绍你自己。",
    )
    print(resp.text)
except Exception as e:
    print("调用失败:", e)
    print("从上面的列表里找准确的模型名，改一下再跑")
