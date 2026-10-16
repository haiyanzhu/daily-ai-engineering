from dotenv import load_dotenv
import os
import time
import random

import chromadb
from chromadb.utils import embedding_functions
from google import genai
from google.genai.errors import ClientError, ServerError

load_dotenv()

# 多模型 fallback（2026-10-07 实测收敛）
# 引自 Haiyan：
# - "2.5 flash 已经不给新用户用了" → 删除
# - "今天 3.8 flash 请求量太高" → 暂不列入
# - "只有 3.5 还好用" → 3.5-flash 打头，lite 作备用
MODELS = [
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]

# 打开已落盘的库 —— 引自 10-06："PersistentClient(path) 本身就是 load，embedding function 每次重传"
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("course", embedding_function=ef)

genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 带 fallback 的单次调用 —— 引自 10-07 实测：
# 503（服务端过载）直接换模型；429（配额）等 60s 重试一次，仍不行再换；
# 模型下线（404/not found）直接跳过。单模型硬编码是 fragile 的。
def generate_with_fallback(prompt):
    last_err = None
    for model in MODELS:
        for attempt in range(2):
            try:
                text = genai_client.models.generate_content(model=model, contents=prompt).text
                return text, model
            except (ClientError, ServerError) as e:
                msg, last_err = str(e), e
                low = msg.lower()
                if "404" in msg or "not found" in low or "not available" in low:
                    print(f"{model} unavailable for this account, skipping...")
                    break
                if "503" in msg:
                    print(f"{model} overloaded, falling back...")
                    break
                if "429" in msg and attempt == 0:
                    wait = 60 + random.uniform(0, 10)
                    print(f"{model} rate limited, waiting {wait:.0f}s then retry...")
                    time.sleep(wait)
                else:
                    print(f"{model} still limited, falling back...")
                    break
    raise last_err

# Map：每个 chunk 独立提炼要点 —— 引自 10-07 context-window 面试题："map-reduce：先逐 chunk 提炼要点再合并"
# 细节要求引自 10-07 复盘"答了但答得浅"："若提到'三个指标'，把三个指标的名字写出来，不要只写'三个指标'"
# 注意这里没接 reranker：map 自带 relevance 过滤，不相关的 chunk 提炼出来是空的
def map_extract(args):
    i, chunk, question = args
    prompt = f"""Extract the key points from the text below that are relevant to the question.
Include names, numbers, and definitions — do not reduce them to topic labels
(e.g., if the text mentions "three metrics", list what the three metrics are).
Return them as a bullet list. Do not add anything not in the text.

Text:
{chunk}

Question: {question}

Relevant points:"""
    text, used_model = generate_with_fallback(prompt)
    return i, text, used_model

# Reduce：综合所有 map 输出 —— 设计来自 Haiyan 10-07：
# "只看提供的资料回答，不做不必要的补充；每句陈述给出资料来源"
# 归类合并要求引自 10-07 复盘："reduce 没有合并归类，只是罗列"
# 用 Chat：引自 10-06 约定"之后默认用 chat"，为后续追问留历史
def reduce_synthesize(mapped, question):
    combined = "\n\n".join(f"[Chunk {i}]\n{m}" for i, m in mapped)
    prompt = f"""Synthesize the final answer from the extracted points below.
Rules:
1. Use ONLY the information in the extracted points. Do not add anything else;
2. Every statement must cite its source as [Chunk N];
3. Organize the points into a coherent structure. Merge overlapping points instead of listing them separately;
4. If the points don't cover some aspect of the question, say so explicitly.

Extracted points:
{combined}

Question: {question}

Final answer (with [Chunk N] citations):"""
    last_err = None
    for model in MODELS:
        try:
            chat = genai_client.chats.create(model=model)
            text = chat.send_message(prompt).text
            print(f"Reduce done via {model}.")
            return text
        except (ClientError, ServerError) as e:
            print(f"Reduce via {model} failed, trying next...")
            last_err = e
    raise last_err

# 主流程
# 免费 tier（5 req/min）下的务实方案：串行 + 节流 —— 引自 10-07 实测：
# "并行重试在紧配额下会 ping-pong 撞墙，退回串行一次跑通"
question = "How do you evaluate a RAG system?"
results = collection.query(query_texts=[question], n_results=10)
chunks = results["documents"][0]
print(f"Retrieved {len(chunks)} chunks.")

mapped = []
for i, c in enumerate(chunks):
    _, text, used_model = map_extract((i, c, question))
    mapped.append((i, text))
    print(f"Chunk {i} done via {used_model}.")
    time.sleep(13)  # 5 req/min → 每次间隔 13s，留 1s 余量

print("\n=== Map outputs (first 2 shown) ===")
for i, m in mapped[:2]:
    print(f"--- Chunk {i} ---\n{m[:300]}...\n")

final = reduce_synthesize(mapped, question)
print("\n=== Reduce: final answer ===")
print(final)
