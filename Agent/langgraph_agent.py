import os
from dotenv import load_dotenv
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# L1: fake weather first (swap in real Open-Meteo in L1.5)
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"It's always sunny in {city}!"

def get_population(city: str) -> str:
    """Get the total population for a given city."""
    return f"{city}'s population is 2.1 million."

from pathlib import Path

def get_chunk_stats() -> str:
    """Return how many course chunks are stored in the P1 RAG Chroma collection."""
    import chromadb
    from chromadb.utils import embedding_functions

    db_path = Path(__file__).resolve().parent / ".." / "RAG" / "chroma_db"
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_collection("course", embedding_function=ef)
    return f"collection 'course' has {collection.count()} chunks"


TOOLS = [get_weather, get_chunk_stats, get_population]
tools_by_name = {t.__name__: t for t in TOOLS}   # same role as the TOOLS dict in the hand-written version

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"),
).bind_tools(TOOLS)

# messages is an AgentState field defined as Sequence[BaseMessage]; add_messages is the reducer that updates it
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def call_model(state: AgentState):
    # ---- Checkpoint 1: how many messages this node actually received ----
    msgs = state["messages"]
    print(f"\n>>> [entering call_model] total messages in state: {len(msgs)}")
    for idx, m in enumerate(msgs):
        print(f"    [{idx}] {m.__class__.__name__}: {str(m.content)[:50]}")
    # ------------------------------------------------------------------------
    response = model.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):          # same role as the natural-exit check in the hand-written version
    last = state["messages"][-1]
    if last.tool_calls:
        return "tools"
    return END

from langgraph.types import interrupt, Command
from langchain_core.messages import ToolMessage
def approval_node(state: AgentState):
    last = state["messages"][-1]
    tool_calls = [{"name": tc["name"], "args": tc["args"]} for tc in last.tool_calls]
    decision = interrupt({
        "question": "Approve these tool calls?",
        "tool_calls": tool_calls,
    })
    if decision is False:                      # rejected: skip the tools, feed the rejection back to the model
        msgs = [ToolMessage(
            content="Tool execution was rejected by the user.",
            tool_call_id=tc["id"],
        ) for tc in last.tool_calls]
        return Command(goto="model", update={"messages": msgs})
    return Command(goto="tools")

graph = StateGraph(AgentState)
graph.add_node("approval", approval_node)
graph.add_node("model", call_model)
graph.add_node("tools", ToolNode(TOOLS))         # replaces the hand-written for-loop dispatch
graph.add_edge(START, "model")
graph.add_conditional_edges("model", lambda s: "approval" if s["messages"][-1].tool_calls else END)
# graph.add_conditional_edges("model", should_continue)
graph.add_edge("tools", "model")
checkpointer = InMemorySaver()
app = graph.compile(checkpointer=checkpointer)
# app = graph.compile()


def run_with_approval(question: str, thread_id: str):
    cfg = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"messages": [HumanMessage(question)]}, cfg)

    while "__interrupt__" in result:          # keep asking while the graph is paused at an approval point
        payload = result["__interrupt__"][0].value
        print("llm want to call tool :", payload["tool_calls"])
        ans = input("Do you approve? (y/n): ").strip().lower()
        result = app.invoke(Command(resume=(ans == "y")), cfg)

    print("final answer:", result["messages"][-1].content)


if __name__ == "__main__":
    cfg1 = {"configurable": {"thread_id": "3"}}

    # 1. First invoke: pauses at interrupt, tools run zero times
    r = app.invoke({"messages": [HumanMessage("What's the weather in London?")]}, cfg1)
    print("interrupt payload:", r.get("__interrupt__"))   # expect the interrupt's question and tool_calls here
    # also check: no "executing" in this round's output, so no tool ran

    # 2. Approve: resume on the same thread
    r2 = app.invoke(Command(resume=True), cfg1)
    print("approved answer:", r2["messages"][-1].content)

    # 3. Reject: start a fresh thread
    cfg4 = {"configurable": {"thread_id": "4"}}
    r3 = app.invoke({"messages": [HumanMessage("What's the weather in London?")]}, cfg4)
    print("interrupt payload:", r3.get("__interrupt__"))
    r4 = app.invoke(Command(resume=False), cfg4)
    print("rejected wrap-up:", r4["messages"][-1].content)

    run_with_approval("What's the weather in London?", "5")


# if __name__ == "__main__":
#     cfg1 = {"configurable": {"thread_id": "1"}}

#     # Experiment A: two questions in a row on the same thread
#     # print("\n========== Running A1 (London) ==========")
#     r1 = app.invoke({"messages": [HumanMessage("What's the weather in London?")]}, cfg1)
#     # print("total messages at end of A1:", len(r1["messages"]))
#     print("A1:", r1["messages"][-1].content)

#     # print("\n========== Running A2 (Paris) ==========")
#     r2 = app.invoke({"messages": [HumanMessage("What about Paris?")]}, cfg1)
#     # print("total messages at end of A2:", len(r2["messages"]))
#     print("A2 final answer:", r2["messages"][-1].content)
#     # print("A2:", r2["messages"][-1].content)

#     # Experiment B: fresh thread, from scratch
#     cfg2 = {"configurable": {"thread_id": "2"}}
#     r3 = app.invoke({"messages": [HumanMessage("What about Paris?")]}, cfg2)
#     print("B:", r3["messages"][-1].content)

#     # inspect what the checkpointer actually stored
#     state = app.get_state(cfg1)
#     print("stored messages:", len(state.values["messages"]))



# if __name__ == "__main__":
#     # result = app.invoke({"messages": [HumanMessage(
#     #     "What's the weather in London and Paris?")]})
#     result = app.invoke({"messages": [HumanMessage(
#             "How many chunks are in my RAG collection?")]})
#
#     for m in result["messages"]:
#         print(type(m).__name__, ":", m.content)
