# Agent/research_agent.py — D2 stub version: search -> summarize -> write-report
# The research step is a small agent with termination conditions; search uses stub
# fake data to get the flow running first, real ddgs search comes tomorrow.
# Acceptance: (1) stub runs end to end, Agent/report.md has content;
#   (2) run once with STUB_NEVER_DONE=True and MAX_STEPS=2 and see a clean
#   termination message instead of an infinite loop;
#   (3) run once with STUB_FAIL=True and see it stop after 3 consecutive
#   errors, with the reason recorded in the report.

import sys
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from ddgs import DDGS

MAX_STEPS = 8
MAX_CONSECUTIVE_ERRORS = 3

# Test switches: STUB_FAIL=True simulates consecutive tool errors;
# STUB_NEVER_DONE=True simulates a model that never says done.
STUB_FAIL = False
STUB_NEVER_DONE = False

STUB_RESULTS = [
    {"title": "LangGraph checkpointing overview", "href": "https://example.com/checkpointing", "body": "Checkpoints persist graph state per thread."},
    {"title": "InMemorySaver vs SqliteSaver", "href": "https://example.com/savers", "body": "InMemorySaver is for dev; SqliteSaver persists to disk."},
    {"title": "Time travel with get_state_history", "href": "https://example.com/time-travel", "body": "Rewind to an earlier checkpoint and fork from there."},
]


class ResearchState(TypedDict):
    question: str
    search_results: list[dict]
    summary: str
    report: str
    steps: int
    consecutive_errors: int
    stopped_reason: str


def real_search(question: str) -> list[dict]:
    """Query ddgs and return title/href/body rows."""
    results = DDGS().text(question, max_results=5)
    return [
        {"title": r.get("title", ""), "href": r.get("href", ""), "body": r.get("body", "")}
        for r in results
    ]

def stub_search(question: str) -> list[dict]:
    """Stub search tool: pretends to find 3 results. Raises when STUB_FAIL=True."""
    if STUB_FAIL:
        raise RuntimeError("stub search tool failed (simulated)")
    return STUB_RESULTS


def search(state: ResearchState) -> ResearchState:
    """Research mini-agent: loop stub_search until natural exit / max_steps /
    3 consecutive tool errors."""
    results: list[dict] = []
    steps = 0
    consecutive_errors = 0
    stopped_reason = ""
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
            continue
        consecutive_errors = 0
        results.extend(batch)
        if STUB_NEVER_DONE:
            print(f"[search] step {steps}: got {len(batch)} results, model hasn't said done, continuing")
            continue
        stopped_reason = "stopped: natural exit (enough results)"
        print("[search] natural exit: enough results collected")
        break
    return {**state, "search_results": results, "steps": steps,
            "consecutive_errors": consecutive_errors, "stopped_reason": stopped_reason}


def summarize(state: ResearchState) -> ResearchState:
    lines = [
        f"- {r['title']} ({r['href']})\n  {r['body']}"
        for r in state["search_results"]
    ]
    summary = "Question: " + state["question"] + "\n" + "\n".join(lines)
    return {**state, "summary": summary}
    # lines = [f"- {r['title']} ({r['href']})" for r in state["search_results"]]
    # summary = "Question: " + state["question"] + "\n" + "\n".join(lines)
    # return {**state, "summary": summary}


def write_report(state: ResearchState) -> ResearchState:
    report = (
        f"# Research report\n\n{state['summary']}\n\n"
        f"## Run notes\n\n- steps: {state['steps']}\n"
        f"- consecutive_errors: {state['consecutive_errors']}\n"
        f"- {state['stopped_reason']}\n"
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
                        "report": "", "steps": 0, "consecutive_errors": 0, "stopped_reason": ""})
    print("done:", final["stopped_reason"])
