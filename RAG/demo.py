import chromadb

client = chromadb.Client()
collection = client.create_collection("demo")
collection.add(documents=
[
	"The cat sat on the mat",
	"Dogs are wonderful pets",
	"Python is a programming language",
],
ids =["1","2","3"],
)
results = collection.query(query_texts = ["what is Python?"], n_results = 1, include=["embeddings", "documents", "distances"])
print(results)
