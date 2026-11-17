# Agent/research_agent_dual_agent.py — Dual-agent research: researcher + writer + reviewer
# Step 1: researcher (ddgs search -> structured findings) + writer (findings -> report).
# Step 2: reviewer loop (evaluator-optimizer): fixed checklist, PASS/REVISE,
# max 2 revision rounds, then ship the current draft with open items noted.
# The orchestrator is deterministic Python: researcher runs, findings are validated,
# then the writer drafts and the reviewer loop runs. No LLM decides the flow;
# the LLM only does the work inside each agent.
# Test switch: FORCE_REVISE=1 forces REVISE on round 1 only, FORCE_REVISE=2
# forces REVISE on every round (tests the revision limit). Reset to 0 for a real run.
# Acceptance: run with a NEW question, end to end, and report.md holds the writer's
# report with a Sources list whose URLs all come from the researcher's findings.

import json
import sys
import time
from pathlib import Path

from ddgs import DDGS
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client()

MODEL = "gemini-3.5-flash-lite"
MAX_RESULTS = 5
MIN_FINDINGS = 3
MAX_CONSECUTIVE_ERRORS = 3
MAX_REVISION_ROUNDS = 2
FORCE_REVISE = 0
SLOPPY_WRITER = True
DEFAULT_QUESTION = "What is the Model Context Protocol (MCP)?"


# ---------------------------------------------------------------- researcher
def real_search(question: str) -> list[dict]:
    """Search tool (reused from D2/ddgs + E1 backoff): one ddgs call, retried
    only on transient errors with exponential backoff (2s, 4s). Stops after
    3 consecutive errors and returns what it has (possibly empty)."""
    consecutive_errors = 0
    while True:
        try:
            results = DDGS().text(question, max_results=MAX_RESULTS, backend="bing")
            print(f"[researcher] search returned {len(results)} results")
            return list(results)
        except Exception as e:  # transient error: back off, then retry
            consecutive_errors += 1
            print(f"[researcher] search error ({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}): {e}")
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                print("[researcher] stopping: 3 consecutive errors, no results")
                return []
            time.sleep(2 ** consecutive_errors)


def build_findings(question: str, results: list[dict]) -> list[dict]:
    """Researcher step 2: compress raw search results into structured findings.
    One LLM call turns 5 raw results into a JSON list of
    {title, url, summary, key_points}. URLs are then validated in code against
    the ddgs results, so a garbled URL can never reach the writer.
    Falls back to deterministic findings (truncated body) if the LLM output
    does not parse."""
    raw = [
        {"title": r.get("title", ""), "url": r.get("href", ""), "body": r.get("body", "")}
        for r in results
    ]
    known_urls = {item["url"] for item in raw if item["url"]}
    prompt = (
        "You are a research assistant. Turn these web search results into "
        "structured findings for the question below. Return a JSON list; each "
        "item has: title (copy exactly), url (copy exactly), summary (2-3 "
        "sentences in your own words), key_points (2-3 short bullets).\n\n"
        f"Question: {question}\n\n"
        f"Search results:\n{json.dumps(raw, ensure_ascii=False, indent=2)}"
    )
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        text = response.text.strip()
        if text.startswith("```"):  # strip fences if the model adds them anyway
            lines = [line for line in text.splitlines()
                     if not line.strip().startswith("```")]
            text = "\n".join(lines)
        findings = json.loads(text)
        if not isinstance(findings, list):
            raise ValueError("findings JSON is not a list")
    except Exception as e:
        print(f"[researcher] summarize failed ({e}); using deterministic fallback")
        return [
            {"title": item["title"], "url": item["url"],
             "summary": item["body"][:300], "key_points": []}
            for item in raw
        ]
    # Deterministic guard: keep only findings whose URL came from ddgs.
    checked = []
    for item in findings:
        if not isinstance(item, dict):
            continue
        if item.get("url") not in known_urls:
            print(f"[researcher] dropped a finding with an unknown URL: {item.get('title', '?')}")
            continue
        checked.append({
            "title": item.get("title", ""),
            "url": item["url"],
            "summary": item.get("summary", ""),
            "key_points": item.get("key_points", []) or [],
        })
    print(f"[researcher] built {len(checked)} findings")
    return checked


# ---------------------------------------------------------------- writer
def format_findings(findings: list[dict]) -> str:
    """Render findings as the numbered evidence pack the writer sees.
    The writer never sees raw search bodies, only this compressed pack."""
    lines = []
    for i, item in enumerate(findings, start=1):
        lines.append(f"[{i}] {item['title']} ({item['url']})")
        lines.append(f"    Summary: {item['summary']}")
        for point in item["key_points"]:
            lines.append(f"    - {point}")
    return "\n".join(lines)


def write_report(question: str, findings: list[dict]) -> str:
    """Writer agent: pure generation, no tools. Writes a structured report
    from the findings only. Graded honesty (from P1): anything not in the
    findings is left out and the gap is marked explicitly.
    Test switch: SLOPPY_WRITER=True drops the citation rules and the
    findings-only constraint, so the real reviewer has a genuine flaw to
    catch. Reset to False for a real run."""
    if SLOPPY_WRITER:
        print("[writer] test switch: sloppy draft (no citation rules)")
        prompt = (
            "You are a technical writer in a hurry. Write a short structured "
            "report answering the question below. Use the findings as a "
            "starting point, but enrich the report with specific facts, "
            "numbers, dates, and company names from your own knowledge, and "
            "do not include any citations or Sources list.\n\n"
            f"Question: {question}\n\n"
            f"Findings:\n{format_findings(findings)}"
        )
    else:
        prompt = (
            "You are a technical writer. Write a short structured report answering "
            "the question below, using ONLY the findings provided. Rules:\n"
            "- Start with a one-paragraph conclusion (answer first).\n"
            "- Then 2-4 short sections with headings.\n"
            "- Cite every factual claim as [Source N], where N is the finding number.\n"
            "- Never add facts, numbers, or links that are not in the findings.\n"
            "- If the findings do not cover part of the question, say so explicitly "
            "in a short 'Not covered by the findings' section.\n"
            "- End with a 'Sources' list: one line per finding, '[N] Title - URL'.\n\n"
            f"Question: {question}\n\n"
            f"Findings:\n{format_findings(findings)}"
        )
    response = client.models.generate_content(model=MODEL, contents=prompt)
    print("[writer] draft written")
    return response.text


# def write_report(question: str, findings: list[dict]) -> str:
#     """Writer agent: pure generation, no tools. Writes a structured report
#     from the findings only. Graded honesty (from P1): anything not in the
#     findings is left out and the gap is marked explicitly."""
#     prompt = (
#         "You are a technical writer. Write a short structured report answering "
#         "the question below, using ONLY the findings provided. Rules:\n"
#         "- Start with a one-paragraph conclusion (answer first).\n"
#         "- Then 2-4 short sections with headings.\n"
#         "- Cite every factual claim as [Source N], where N is the finding number.\n"
#         "- Never add facts, numbers, or links that are not in the findings.\n"
#         "- If the findings do not cover part of the question, say so explicitly "
#         "in a short 'Not covered by the findings' section.\n"
#         "- End with a 'Sources' list: one line per finding, '[N] Title - URL'.\n\n"
#         f"Question: {question}\n\n"
#         f"Findings:\n{format_findings(findings)}"
#     )
#     response = client.models.generate_content(model=MODEL, contents=prompt)
#     print("[writer] draft written")
#     return response.text


# ---------------------------------------------------------------- reviewer
REVIEW_CHECKLIST = (
    "1. Every [Source N] citation points to a finding that actually exists "
    "and supports the claim.\n"
    "2. No facts, numbers, or links appear that are not in the findings.\n"
    "3. No important point from the findings is missing from the report."
)


def review_report(question: str, findings: list[dict], draft: str, round_no: int) -> tuple[str, str]:
    """Reviewer agent: pure review, no tools. Checks the draft against the
    fixed checklist and returns (verdict, feedback). The verdict is parsed
    from the first line ('VERDICT: PASS' or 'VERDICT: REVISE'); anything
    else is treated as REVISE so an unclear review never ships the draft.
    Test switch (D2 pattern): FORCE_REVISE=1 forces REVISE on round 1 only,
    FORCE_REVISE=2 forces REVISE on every round. Reset to 0 for a real run."""
    if FORCE_REVISE == 1 and round_no == 1:
        print("[reviewer] test switch: forcing REVISE on round 1")
        return "REVISE", (
            "- The conclusion paragraph states a benefit without a [Source N] citation; add one.\n"
            "- Finding 2 describes MCP servers, but the report never mentions servers; add a short section."
        )
    if FORCE_REVISE == 2:
        print(f"[reviewer] test switch: forcing REVISE on round {round_no}")
        return "REVISE", (
            "- The conclusion paragraph states a benefit without a [Source N] citation; add one."
        )
    prompt = (
        "You are a strict reviewer. Check the draft report against the "
        "checklist below, using ONLY the findings as ground truth.\n\n"
        "Checklist:\n"
        f"{REVIEW_CHECKLIST}\n\n"
        "Reply in this exact format:\n"
        "First line: VERDICT: PASS  (if every checklist item holds)\n"
        "or: VERDICT: REVISE (if any item fails)\n"
        "Then, if REVISE, one bullet per issue. Each bullet must name the "
        "paragraph or section AND the finding number involved, e.g. "
        "'Section 2, paragraph 1 cites [Source 4] but finding 4 says nothing "
        "about pricing.' Vague feedback like 'improve the report' is not "
        "acceptable.\n\n"
        f"Question: {question}\n\n"
        f"Findings:\n{format_findings(findings)}\n\n"
        f"Draft report:\n{draft}"
    )
    response = client.models.generate_content(model=MODEL, contents=prompt)
    text = response.text.strip()
    first_line = text.splitlines()[0].strip().upper() if text else ""
    if first_line.startswith("VERDICT: PASS"):
        return "PASS", ""
    if first_line.startswith("VERDICT: REVISE"):
        feedback = "\n".join(text.splitlines()[1:]).strip()
        return "REVISE", feedback or text
    print("[reviewer] no verdict line found; treating as REVISE")
    return "REVISE", text


def revise_report(question: str, findings: list[dict], draft: str, feedback: str) -> str:
    """Writer agent, second mode: revise the draft using ONLY the reviewer's
    feedback. Same graded-honesty rules as the first draft: no new facts
    beyond the findings."""
    prompt = (
        "You are a technical writer revising a draft. Apply ONLY the "
        "reviewer's feedback below. Keep every [Source N] citation accurate, "
        "never add facts, numbers, or links that are not in the findings, "
        "and keep the 'Sources' list at the end.\n\n"
        f"Question: {question}\n\n"
        f"Findings:\n{format_findings(findings)}\n\n"
        f"Reviewer feedback:\n{feedback}\n\n"
        f"Current draft:\n{draft}"
    )
    response = client.models.generate_content(model=MODEL, contents=prompt)
    print("[writer] draft revised")
    return response.text


# ---------------------------------------------------------------- orchestrator
def validate_findings(findings: list[dict]) -> bool:
    """Deterministic gate between researcher and writer: at least MIN_FINDINGS
    findings, each with a non-empty URL. Failing fast here is cheaper than
    letting the writer draft from thin evidence."""
    if len(findings) < MIN_FINDINGS:
        print(f"[orchestrator] only {len(findings)} findings, need at least {MIN_FINDINGS}")
        return False
    if any(not item.get("url") for item in findings):
        print("[orchestrator] a finding is missing its URL")
        return False
    return True


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION
    print(f"[orchestrator] question: {question}")

    results = real_search(question)
    if not results:
        print("[orchestrator] no search results; stopping before the writer")
        return
    findings = build_findings(question, results)
    if not validate_findings(findings):
        print("[orchestrator] findings failed validation; stopping before the writer")
        return
    report = write_report(question, findings)

    # Step 2: reviewer loop (evaluator-optimizer). At most MAX_REVISION_ROUNDS
    # rounds; if the last round still says REVISE, ship the current draft and
    # note the open items on top instead of looping forever.
    open_items = ""
    for round_no in range(1, MAX_REVISION_ROUNDS + 1):
        verdict, feedback = review_report(question, findings, report, round_no)
        print(f"[reviewer] round {round_no}: {verdict}")
        if verdict == "PASS":
            break
        print(f"[reviewer] feedback:\n{feedback}")
        if round_no == MAX_REVISION_ROUNDS:
            open_items = feedback
            print("[orchestrator] revision limit reached; shipping current draft")
        else:
            report = revise_report(question, findings, report, feedback)
    print(f"[orchestrator] review finished after {round_no} round(s)")

    if open_items:
        note = (
            "> **Reviewer open items:** this draft did not pass review within "
            f"{MAX_REVISION_ROUNDS} revision rounds. Unresolved feedback:\n"
            + "\n".join(f"> {line}" for line in open_items.splitlines() if line.strip())
            + "\n\n---\n\n"
        )
        report = note + report
    Path(__file__).with_name("report.md").write_text(report, encoding="utf-8")
    print("[orchestrator] report.md written to disk")


if __name__ == "__main__":
    main()
