"""Research agent with a whitelisted terminal tool (T1 stage).

Base: research_agent_d2_ddgs.py (search -> summarize -> write report).
New in T1: a run_shell tool behind five safety gates.

The five gates (from the qwertee.io agent-harness article, Step 4):
  1. Executable allowlist: only ls, pwd, wc, python3 may run.
  2. Trusted scripts only: the tool never runs model-written `python -c`.
     python3 runs a script file; the model cannot inject code as an argument.
  3. No shell: subprocess.run with shell=False and argv passed as a list,
     so pipes, `;`, `&&`, and `$()` are literal strings, never separators.
     They are rejected outright anyway (defense in depth).
  4. Timeout + output cap: 30 seconds, 20,000 characters.
  5. Errors as results: timeout, denial, and nonzero exit all come back as
     plain-text results, so the run never crashes on a failed command.

The approval gate is the lightweight version of the L4 interrupt():
before a command runs, the user is asked in the terminal. Allowlist
rejections happen before approval, so a denied command never prompts.
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import TypedDict

from ddgs import DDGS
from langgraph.graph import END, StateGraph

# --- Terminal tool configuration -------------------------------------------

ALLOWED_COMMANDS = {"ls", "pwd", "wc", "python3"}
COMMAND_TIMEOUT_SECONDS = 30
MAX_OUTPUT_CHARS = 20000
REQUIRE_APPROVAL = True

AGENT_DIR = Path(__file__).resolve().parent
REJECTED_TOKENS = (";", "|", "&&", "$(", "`", ">", "<", "\n")

# --- Search configuration (same as D2 ddgs) ---------------------------------

MAX_STEPS = 8
MAX_CONSECUTIVE_ERRORS = 3


def run_shell(argv):
    """Run one command through the five safety gates.

    Returns a plain-text result in every case: the command output on
    success, or a clear error message on denial, timeout, or failure.
    The caller never sees an exception, so the run never crashes.
    """
    if not argv:
        return "Error: no command given."

    command = argv[0]

    # Gate 1: executable allowlist.
    if command not in ALLOWED_COMMANDS:
        return (
            f"Denied: '{command}' is not on the allowlist. "
            f"Allowed commands: {', '.join(sorted(ALLOWED_COMMANDS))}."
        )

    # Gate 2: trusted scripts only — never run model-written `python -c`.
    if command == "python3" and "-c" in argv:
        return "Denied: 'python3 -c' is not allowed. Run a script file instead."

    # Gate 3 (defense in depth): reject shell metacharacters outright.
    # With shell=False they would be literal strings, never separators,
    # but rejecting them keeps the tool's behavior obvious.
    for arg in argv:
        for token in REJECTED_TOKENS:
            if token in arg:
                return f"Denied: shell metacharacter '{token}' is not allowed in arguments."

    # Approval gate (lightweight L4): ask the human before running.
    if REQUIRE_APPROVAL:
        answer = input(f"[approval] Run: {' '.join(argv)} ? (y/n) ").strip().lower()
        if answer != "y":
            return f"Rejected by user: {' '.join(argv)} was not run."

    # Gates 3 + 4: no shell, timeout, output cap.
    try:
        proc = subprocess.run(
            argv,
            shell=False,
            timeout=COMMAND_TIMEOUT_SECONDS,
            capture_output=True,
            text=True,
            cwd=AGENT_DIR,
        )
    except subprocess.TimeoutExpired:
        return (
            f"Timeout: the command ran longer than "
            f"{COMMAND_TIMEOUT_SECONDS} seconds and was stopped."
        )
    except FileNotFoundError:
        return f"Error: '{command}' was not found on this machine."

    output = (proc.stdout + proc.stderr).strip()
    if len(output) > MAX_OUTPUT_CHARS:
        output = output[:MAX_OUTPUT_CHARS] + "\n... [output truncated]"
    if proc.returncode != 0:
        return f"Command exited with code {proc.returncode}:\n{output}"
    return output if output else "(no output)"


def real_search(question):
    # backend="bing": on 2026-11-10 the duckduckgo backend returned no
    # results for this question from her network; bing worked.
    results = DDGS().text(question, max_results=5, backend="bing")
    return [
        {"title": r["title"], "href": r["href"], "body": r.get("body", "")}
        for r in results
    ]


class AgentState(TypedDict, total=False):
    question: str
    results: list
    terminal_output: list
    stopped_reason: str
    report: str


def search(state):
    """Search loop with the termination triple, as in D2."""
    question = state["question"]
    results = []
    consecutive_errors = 0
    steps = 0
    stopped_reason = ""
    while len(results) < 5 and steps < MAX_STEPS:
        steps += 1
        try:
            new_results = real_search(question)
            results.extend(new_results)
            consecutive_errors = 0  # a success resets the consecutive counter
            print(f"[search] step {steps}: collected {len(results)} result(s)")
        except Exception as e:  # tool errors go to the counter, don't crash the graph
            consecutive_errors += 1
            print(f"[search] tool error ({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}): {e}")
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                stopped_reason = f"stopped: {MAX_CONSECUTIVE_ERRORS} consecutive tool errors"
                print("[search] stopping: 3 consecutive errors, wrapping up")
                break
            time.sleep(2 ** consecutive_errors)  # back off before retrying a rate-limited backend
            continue
    if not stopped_reason:
        if len(results) >= 5:
            stopped_reason = "stopped: natural exit (enough results)"
            print("[search] natural exit: enough results collected")
        else:
            stopped_reason = f"stopped: reached MAX_STEPS ({MAX_STEPS})"
            print(f"[search] stopped: reached MAX_STEPS ({MAX_STEPS})")
    return {"results": results, "stopped_reason": stopped_reason}


# Demo commands for the terminal node. `wc -l report.md` reads the report
# from the PREVIOUS run (this run's report is not written yet); on the
# very first run the file may not exist, which exercises the
# nonzero-exit error-as-result path. `rm` never runs: the allowlist
# rejects it before approval, so it never even prompts.
DEMO_COMMANDS = [
    ["ls"],
    ["wc", "-l", "report.md"],
    ["rm", "report.md"],
]


def run_terminal(state):
    """Run the demo commands through run_shell and record the results."""
    outputs = []
    for argv in DEMO_COMMANDS:
        result = run_shell(argv)
        outputs.append({"command": " ".join(argv), "result": result})
        first_line = (result.splitlines() or [""])[0][:80]
        print(f"[terminal] {' '.join(argv)} -> {first_line}")
    return {"terminal_output": outputs}


def summarize(state):
    """Build the report: search results plus a terminal-output section."""
    lines = [f"# Research report: {state['question']}", "", "## Search results", ""]
    for r in state.get("results", []):
        lines.append(f"- [{r['title']}]({r['href']})")
        if r.get("body"):
            lines.append(f"  {r['body']}")
    lines += ["", "## Terminal output", ""]
    for item in state.get("terminal_output", []):
        lines.append(f"### $ {item['command']}")
        lines.append("```")
        lines.append(item["result"])
        lines.append("```")
        lines.append("")
    return {"report": "\n".join(lines)}


def write_report(state):
    path = Path(__file__).with_name("report.md")
    path.write_text(state["report"], encoding="utf-8")
    print("[write_report] report.md written to disk")
    return {}


def main():
    question = sys.argv[1] if len(sys.argv) > 1 else "How does LangGraph checkpointing work?"
    graph = StateGraph(AgentState)
    graph.add_node("search", search)
    graph.add_node("terminal", run_terminal)
    graph.add_node("summarize", summarize)
    graph.add_node("write_report", write_report)
    graph.set_entry_point("search")
    graph.add_edge("search", "terminal")
    graph.add_edge("terminal", "summarize")
    graph.add_edge("summarize", "write_report")
    graph.add_edge("write_report", END)
    app = graph.compile()
    final = app.invoke({"question": question})
    print(f"done: {final.get('stopped_reason', '')}")


if __name__ == "__main__":
    main()
