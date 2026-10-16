from dotenv import load_dotenv
import os
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
from google import genai
import time

load_dotenv()

# ===== 1. 打开已落盘的库（秒开）=====
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("course", embedding_function=ef)
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

genai_client = genai.Client()

PROMPT_TEMPLATE = """Answer the question using ONLY the context below.

Rules:
1. Use only the supplied context. Do not use outside knowledge.
2. If the context only partially covers the question, answer the supported part and explicitly say which part is not covered by the context.
3. If the context is completely irrelevant to the question, say you cannot answer based on the provided context.

Context:
{context}

Question: {question}

Answer:"""


def ask_rag(question: str):
    t0 = time.time()
    results = collection.query(query_texts=[question], n_results = 10)
    t1 = time.time()
    docs = results["documents"][0]
    distances = results["distances"][0]
    pairs = [[question, d] for d in docs]
    scores = reranker.predict(pairs)
    t2 = time.time()
    ranked = sorted(zip(docs, distances,scores), key = lambda x:x[2], reverse=True)[:3]
    context = "\n\n".join(d for d,_,_ in ranked)
    prompt = PROMPT_TEMPLATE.format(question=question, context=context)

    response = genai_client.models.generate_content(
        model="gemini-3.5-flash", contents=prompt
    )
    answer = response.text
    t3 = time.time()
    print(f"retrieve: {t1-t0:.1f}s | rerank: {t2-t1:.1f}s | generate: {t3-t2:.1f}s")
    sources = [{"text": d, "distance": dist} for d, dist, _ in ranked]
    return answer, sources



if __name__ == "__main__":
    ask_rag("What is RAG?")


# # ===== 2. 检索 =====
# question = "What is RAG?"
# results = collection.query(query_texts=[question], n_results=10)
# docs = results["documents"][0]

# # ===== 3. rerank（用你刚才翻盘成功时验证的那个模型）=====
# reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
# scores = reranker.predict([(question, d) for d in docs])
# ranked = sorted(zip(scores, docs), reverse=True)
# top3 = [d for _, d in ranked[:3]]

# print("=== rerank 后 Top-3 ===")
# for i, (s, d) in enumerate(ranked[:3], 1):
#     print(f"--- 第{i}名 (score={s:.4f}) ---")
#     print(d[:200], "...\n")

# # ===== 4. 拼 prompt + 生成 =====
# context = "\n\n---\n\n".join(top3)
# prompt = f"""Answer the question based ONLY on the context below.
# Rules:
# 1. Use only information from the context. Do not supplement with your own knowledge;
# 2. If the context covers only part of the question, answer what you can and explicitly state which aspects are "not mentioned in the context";
# 3. Reply "Cannot answer based on the provided materials" ONLY when the context is completely irrelevant to the question.

# Context:
# {context}

# Question: {question}

# Answer:"""

# genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
# resp = genai_client.models.generate_content(model="gemini-3.8-flash", contents=prompt)

# print("=== Gemini 回答 ===")
# print(resp.text)


