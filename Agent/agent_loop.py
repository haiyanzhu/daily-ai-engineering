import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()  # finds the shared .env in the parent folder

MODEL = "gemini-3.5-flash"
MAX_STEPS = 5


from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

CHROMA_PATH = Path(__file__).resolve().parent.parent / "RAG" / "chroma_db"
_embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
_collection = chromadb.PersistentClient(path=str(CHROMA_PATH)).get_collection(
    name="course", embedding_function=_embedding_fn
)

get_chunk_stats_declaration = {
    "name": "get_chunk_stats",
    "description": "Returns the number of chunks stored in the local course corpus (Chroma collection 'course'). Use this when the user asks how many chunks or documents the corpus contains.",
    "parameters": {"type": "object", "properties": {}},
}

def get_chunk_stats() -> dict:
    """Returns statistics about the local course corpus."""
    return {"chunk_count": _collection.count()}


# --- Tool implementation (the body: only ever runs locally) ---
def get_weather(city: str) -> dict:
    return {"city": city, "weather": "Fantastic"}

# --- Hand-written declaration (the menu: the only thing the model sees) ---
get_weather_declaration = {
    "name": "get_weather",
    "description": "Gets the current weather for a given city. Use this when the user asks about weather in a specific city.",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "The city name, e.g. London"},
        },
        "required": ["city"],
    },
}

TOOLS = {"get_weather": get_weather, "get_chunk_stats": get_chunk_stats}  # name -> body, YOUR function map

client = genai.Client()
config = types.GenerateContentConfig(
    tools=[types.Tool(function_declarations=[get_weather_declaration, get_chunk_stats_declaration])],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)
question1 = "How many chunks are in my course corpus?"
question = "What's the weather in London and Paris?"
history = [types.Content(role="user", parts=[types.Part(text=question1)])]

for step in range(1, MAX_STEPS + 1):
    response = client.models.generate_content(model=MODEL, contents=history, config=config)
    calls = response.function_calls  # plural: every call in this turn

    if not calls:
        # Natural stop: the model answered in text, we are done
        print(f"[step {step}] final answer: {response.text}")
        break

    print(f"[step {step}] calls: {[(c.name, dict(c.args)) for c in calls]}")
    history.append(response.candidates[0].content)  # model's turn, kept whole

    response_parts = []
    for fc in calls:
        try:
            result = TOOLS[fc.name](**fc.args)
        except Exception as e:
            result = {"error": str(e)}  # errors go back to the model, loop survives
        response_parts.append(
            types.Part(
                function_response=types.FunctionResponse(
                    id=fc.id, name=fc.name, response={"result": result}
                )
            )
        )
    history.append(types.Content(role="user", parts=response_parts))
else:
    print(f"Stopped: hit MAX_STEPS={MAX_STEPS} without a final answer")
