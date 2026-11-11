# Agent/research_agent.py — D2 with episodic memory (E1).
# Flow: recall episodes -> search (ddgs) -> summarize -> write-report.
# Episodes live in a second Chroma collection ("episodes") under Agent/chroma_episodes.
# Only runs that end with a natural exit are stored; runs that stop on errors are
# not stored, because a failed run stored as an example would teach the agent to
# repeat the failure.
# Note: there is no LLM call in this version yet, so recalled episodes surface in
# two places: printed at run start and written into report.md under
# "Recalled episodes". When summarize() later calls a model, they go straight into
# its prompt as few-shot examples.

import sys
import uuid
from pathlib import Path
from typing import TypedDict

from ddgs import DDGS
from langgraph.graph import END, START, StateGraph
import time

MAX_STEPS = 8
MAX_CONSECUTIVE_ERRORS = 3

# Test switches: STUB_FAIL=True simulates consecutive tool errors;
# STUB_NEVER_DONE=True simulates a model that never says done.
STUB_FAIL = False
STUB_NEVER_DONE = False


def real_search(question: str) -> list[dict]:
    """Query ddgs and return title/href/body rows."""
    results = DDGS().text(question, max_results=5)
    return [
        {"title": r.get("title", ""), "href": r.get("href", ""), "body": r.get("body", "")}
        for r in results
    ]


def _episodes_collection():
    """Open (or create) the episodes collection. Same embedding model as P1."""
    import chromadb
    from chromadb.utils import embedding_functions

    db_path = Path(__file__).resolve().parent / "chroma_episodes"
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=str(db_path))
    return client.get_or_create_collection("episodes", embedding_function=ef)


def recall_episodes(question: str, k: int = 2) -> list[dict]:
    """Retrieve the top-k past episodes most similar to the question."""
    collection = _episodes_collection()
    if collection.count() == 0:
        return []
    results = collection.query(
        query_texts=[question], n_results=min(k, collection.count()))
    episodes = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        episodes.append({"document": doc, **meta})
    return episodes


def store_episode(question: str, steps: int, result_count: int) -> None:
    """Store one episode for a run that ended with a natural exit.

    The episode follows the Episode schema from the LangMem conceptual guide:
    observation / thoughts / action / result. The thoughts here are the agent's
    fixed search strategy, not model-generated text; a model-driven agent would
    write its own thoughts per run.
    """
    collection = _episodes_collection()
    document = (
        f"Observation: the question was '{question}'.\n"
        f"Thoughts: search with ddgs text search (max_results=5); recall past episodes first.\n"
        f"Action: ran ddgs search, took {steps} step(s), collected {result_count} result(s).\n"
        f"Result: natural exit, enough results collected."
    )
    collection.add(
        ids=[str(uuid.uuid4())],
        documents=[document],
        metadatas=[{"question": question, "steps": steps, "result_count": result_count}],
    )
    print(f"[memory] episode stored (collection now holds {collection.count()} episode(s))")


class ResearchState(TypedDict):
    question: str
    search_results: list[dict]
    summary: str
    report: str
    steps: int
    consecutive_errors: int
    stopped_reason: str
    recalled_episodes: list[dict]
    episode_stored: bool


def search(state: ResearchState) -> ResearchState:
    """Research mini-agent: recall episodes, then loop real_search until natural
    exit / max_steps / 3 consecutive tool errors."""
    recalled = recall_episodes(state["question"], k=4)
    if recalled:
        print(f"[memory] recalled {len(recalled)} episode(s) from past runs:")
        for ep in recalled:
            print(f"  - {ep['document'].splitlines()[0]}")
    else:
        print("[memory] no past episodes to recall")

    results: list[dict] = []
    steps = 0
    consecutive_errors = 0
    stopped_reason = ""
    episode_stored = False
    while True:
        steps += 1
        if steps > MAX_STEPS:
            stopped_reason = f"stopped: hit max_steps={MAX_STEPS}"
            print(f"[search] stopping: hit max_steps={MAX_STEPS}, wrapping up")
            break
        try:
            batch = real_search(state["question"])
        except Exception as e:  # tool errors go to the counter, don't crash the graph
            consecutive_errors += 1
            print(f"[search] tool error ({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}): {e}")
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                stopped_reason = f"stopped: {MAX_CONSECUTIVE_ERRORS} consecutive tool errors"
                print("[search] stopping: 3 consecutive errors, wrapping up")
                break
            time.sleep(2 ** consecutive_errors)  # back off before retrying a rate-limited backend
            continue
        consecutive_errors = 0
        results.extend(batch)
        if STUB_NEVER_DONE:
            print(f"[search] step {steps}: got {len(batch)} results, model hasn't said done, continuing")
            continue
        stopped_reason = "stopped: natural exit (enough results)"
        print("[search] natural exit: enough results collected")
        break

    # Only successful runs become memories.
    if stopped_reason.startswith("stopped: natural exit"):
        store_episode(state["question"], steps, len(results))
        episode_stored = True
    return {**state, "search_results": results, "steps": steps,
            "consecutive_errors": consecutive_errors, "stopped_reason": stopped_reason,
            "recalled_episodes": recalled, "episode_stored": episode_stored}


def summarize(state: ResearchState) -> ResearchState:
    lines = [
        f"- {r['title']} ({r['href']})\n  {r['body']}"
        for r in state["search_results"]
    ]
    summary = "Question: " + state["question"] + "\n" + "\n".join(lines)
    if state["recalled_episodes"]:
        recalled_lines = [f"- {ep['document']}" for ep in state["recalled_episodes"]]
        summary += ("\n\nRecalled episodes (few-shot examples from past runs):\n"
                    + "\n".join(recalled_lines))
    return {**state, "summary": summary}


def write_report(state: ResearchState) -> ResearchState:
    report = (
        f"# Research report\n\n{state['summary']}\n\n"
        f"## Run notes\n\n- steps: {state['steps']}\n"
        f"- consecutive_errors: {state['consecutive_errors']}\n"
        f"- {state['stopped_reason']}\n"
        f"- recalled episodes: {len(state['recalled_episodes'])}\n"
        f"- episode stored: {state['episode_stored']}\n"
    )
    Path(__file__).with_name("report.md").write_text(report, encoding="utf-8")
    print("[write_report] report.md written to disk")
    return {**state, "report": report}


graph = StateGraph(ResearchState)
graph.add_node("search", search)
graph.add_node("summarize", summarize)
graph.add_node("write_report", write_report)
graph.add_edge(START, "search")
graph.add_edge("search", "summarize")
graph.add_edge("summarize", "write_report")
graph.add_edge("write_report", END)
app = graph.compile()

if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "How does LangGraph checkpointing work?"
    final = app.invoke({"question": question, "search_results": [], "summary": "",
                        "report": "", "steps": 0, "consecutive_errors": 0, "stopped_reason": "",
                        "recalled_episodes": [], "episode_stored": False})
    print("done:", final["stopped_reason"])
