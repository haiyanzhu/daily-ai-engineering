import os
import glob
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
import os

load_dotenv()  # 把 .env 加载进环境变量
api_key = os.getenv("GEMINI_API_KEY")




    

# ===== 只改这里 =====
DIR = os.path.expanduser("~/Desktop/ai-engineering-from-scratch")  # 课程 markdown 文件所在目录
QUERY = "What is RAG?"                  # 检索问题
# ====================

# 1. 读文件 + 按段落切分（chunking）

chunks, ids, metadatas = [], [], []


EXCLUDE = {"CONTRIBUTING.md", "CODE_OF_CONDUCT.md", "AGENTS.md", "README.md"}
md_files = [
    p for p in sorted(glob.glob(os.path.join(DIR, "phases", "**", "*.md"), recursive=True))
    if os.path.basename(p) not in EXCLUDE
]

#md_files = sorted(glob.glob(os.path.join(DIR, "*.md")))[:5]
print(f"找到 {len(md_files)} 个 markdown 文件：{[os.path.basename(p) for p in md_files]}")

for fi, path in enumerate(md_files):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    for ci, para in enumerate(text.split("\n\n")):  # 按空行切段
        para = para.strip()
        if len(para) < 50:      # 过滤太短的碎片
            continue
        chunks.append(para)
        ids.append(f"f{fi}-c{ci}")
        metadatas.append({"source": os.path.basename(path)})

print(f"共切出 {len(chunks)} 个 chunks")

# 2. 显式指定 embedding 模型，建 collection
ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
#client = chromadb.Client()

# save to disk instead of to memory (which will be disappear after running the program)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("course", embedding_function=ef)

# 3. 存入 Chroma 直接存入数据库存在单次存入的size问题，我们需要分批存入
#collection.add(documents=chunks, ids=ids, metadatas=metadatas)
#print(f"已存入 {collection.count()} 条")

#3.存入， Chroma 底层使用的是sqlite数据库来存储文档和metadata, 但是一次写入量不能超过（5461），但是我们的chunksize 很大 大概有几万个，那么我们就要分批次去写入。
if collection.count() == 0:
    BATCH_SIZE = 1000
    for i in range(0, len(chunks), BATCH_SIZE):
        j = min(i + BATCH_SIZE, len(chunks))
        collection.add(documents=chunks[i:j], ids=ids[i:j], metadatas=metadatas[i:j])
        print(f"已存入 {j}/{len(chunks)}")
else:
    print(f"库中已有 {collection.count()} 条，跳过写入")
print(f"库中共有 {collection.count()} 条")


# 4. 检索
results = collection.query(query_texts=[QUERY], n_results=3)
for rank, (doc, meta, dist) in enumerate(
    zip(results["documents"][0], results["metadatas"][0], results["distances"][0]),
    start=1,
):
    print(f"\n--- Top {rank}（来源：{meta['source']}，距离：{dist:.4f}）---")
    print(doc[:300])

#5 use cross-encode to rerank the result

reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")  # 首次运行会下载模型
docs = results["documents"][0]
pairs = [(QUERY, d) for d in docs]
scores = reranker.predict(pairs)

ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
print("\n=== reranker 精排后 Top-3 ===")
for i, (d, s) in enumerate(ranked[:3]):
    print(f"--- 第{i+1}名 (score={s:.4f}) ---")
    print(d[:200])
    print()

