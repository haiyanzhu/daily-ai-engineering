from fastapi import FastAPI
from pydantic import BaseModel
from rag_qa import ask_rag
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    
    allow_origins=["http://localhost:8080",
        "http://127.0.0.1:8080",],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    sources: list

@app.post("/ask")
def ask(req: AskRequest) -> AskResponse:
    answer, sources = ask_rag(req.question)
    return AskResponse(answer=answer, sources=sources)
