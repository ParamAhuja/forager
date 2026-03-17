"""
agent5_qa.py — Agent 5: The QA Engineer
=========================================
Launches a secure cloud microVM via E2B, clones a GitHub repository,
autonomously debugs it using a ReAct (Reason + Act) loop, and streams
all activity live to the frontend via a token_queue.

Architecture
────────────
run_qa_agent(repo_url, token_queue)
  • Initialises an E2B Sandbox (cloud microVM — sandboxed, ephemeral, secure)
  • Drives a ReAct loop: LLM think → tool dispatch → live observation → repeat
  • Puts SSE-ready JSON strings into token_queue for server.py to stream
  • Hard-capped at MAX_ITERATIONS (10) — exits with a final summary on breach

Tools the LLM may invoke:
  run_command(cmd, sandbox)        — sandbox.commands.run with live stdout/stderr callbacks
  read_file(path, sandbox)         — sandbox.files.read
  write_file(path, content, sb)    — sandbox.files.write

SSE event types streamed to the frontend:
  {"type": "thought",  "text":    "..."}   LLM's internal reasoning step
  {"type": "terminal", "text":    "..."}   Real-time stdout / stderr from E2B
  {"type": "app_url",  "url":     "..."}   Live preview URL once the app is running
  {"type": "done",     "summary": "..."}   Final completion/failure summary
  {"type": "error",    "text":    "..."}   Fatal agent errors

Security
────────
  • No API keys or secrets are stored in this file.
  • All LLM calls are routed through api_gateway.groq_execute() — the single
    authoritative call site for the Groq key pool.
  • The E2B_API_KEY is loaded from the environment by the E2B SDK itself —
    agent5_qa.py never reads os.environ directly.
  • On success the sandbox is intentionally kept alive so the user can view
    the live preview URL. E2B automatically reclaims it after SANDBOX_TIMEOUT.
  • On failure or crash the sandbox is always killed in the `finally` block.
"""

from __future__ import annotations

import json
import queue as _queue
import re
import traceback
from typing import Callable

from e2b import Sandbox

import api_gateway

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MAX_ITERATIONS  = 10
QA_MODEL        = "llama-3.3-70b-versatile"

# E2B sandbox timeout (seconds). The sandbox is kept alive for the full
# agent run; individual commands have their own per-command timeout below.
SANDBOX_TIMEOUT = 300   # 5 minutes maximum total session

# Per-command timeout (seconds). Long installs (npm ci, pip install) can
# take a while inside a cold microVM. 90 s is generous but bounded.
COMMAND_TIMEOUT = 90


# ─────────────────────────────────────────────────────────────────────────────
# §1  System Prompt — the LLM's operational constitution
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are Agent 5 — an elite autonomous QA Engineer operating inside a \
secure, disposable cloud microVM (E2B sandbox). Your one mission: clone the \
provided GitHub repository and get its application running so the user can see \
it live in a browser preview.

═══════════════════════════════════════════════════════════
RESPONSE FORMAT — STRICTLY ENFORCED
═══════════════════════════════════════════════════════════
Every response MUST be a single line of valid JSON. No markdown, no preamble,
no explanation outside the JSON object. The schema is:

  {"thought": "<your full reasoning>", "action": {"tool": "<tool_name>", "args": {<args>}}}

Deviation from this format will cause a parse failure that costs an iteration.

═══════════════════════════════════════════════════════════
AVAILABLE TOOLS
═══════════════════════════════════════════════════════════

run_command — Execute any shell command in the sandbox.
  args: { "cmd": "<shell command>" }
  Example: {"thought":"Check what's in the repo","action":{"tool":"run_command","args":{"cmd":"ls -la /home/user/repo"}}}

read_file — Read a file from the sandbox filesystem.
  args: { "path": "<absolute path>" }
  Example: {"thought":"Inspect package.json","action":{"tool":"read_file","args":{"path":"/home/user/repo/package.json"}}}

write_file — Create or overwrite SHORT config files (< 5 lines, no special characters).
  ⚠ DANGER: JSON escaping of newlines (\\n), quotes (\\"), and tabs (\\t) inside a JSON
  string argument is extremely error-prone for multi-line content. A single unescaped
  character causes a JSONDecodeError that wastes an entire iteration.
  args: { "path": "<absolute path>", "content": "<content — short single-line values only>" }
  Example: {"thought":"Set a single env var","action":{"tool":"write_file","args":{"path":"/home/user/repo/.env","content":"PORT=3000"}}}

  ✅ PREFER run_command with a bash heredoc for ANY multi-line content (configs, patches,
  scripts). Heredocs are 100% safe from JSON escaping bugs because the code lives inside
  the cmd string with no inner quoting required:
    {"thought":"Rewrite server entry point","action":{"tool":"run_command","args":{"cmd":"cat << 'HEREDOC' > /home/user/repo/server.js\nconst express = require('express');\nconst app = express();\napp.listen(3000);\nHEREDOC"}}}
  Use the literal token HEREDOC (or EOF) and single-quote it (<< 'HEREDOC') to prevent
  variable expansion inside the heredoc body.

done — Signal completion. Call this as soon as you confirm the app is running,
       or if you exhaust all strategies and cannot proceed.
  args: { "status": "success"|"failure", "summary": "<what happened>", "port": <int or null> }
  Example (success): {"thought":"curl returned 200, app is live","action":{"tool":"done","args":{"status":"success","summary":"React app running on port 3000","port":3000}}}
  Example (failure): {"thought":"pip install fails due to unresolvable conflict","action":{"tool":"done","args":{"status":"failure","summary":"Dependency conflict in requirements.txt: numpy 1.x vs pandas 2.x","port":null}}}

═══════════════════════════════════════════════════════════
OPERATIONAL RULES
═══════════════════════════════════════════════════════════

1. CLONE FIRST — always start with: git clone <url> /home/user/repo
2. INSPECT STRUCTURE — ls, then read package.json / requirements.txt /
   pyproject.toml / Cargo.toml / go.mod to understand the app type.
3. INSTALL DEPS — use the right package manager:
     Node.js  →  cd /home/user/repo && npm install  (or yarn install / pnpm install)
     Python   →  pip install -r requirements.txt  (or pip install -e .)
     Other    →  infer from lock files
4. PORT SELECTION:
     Node.js / frontend   →  port 3000
     Python / FastAPI     →  port 8000
     Both or unclear      →  try 3000 first
5. START THE APP — use the correct start command:
     Node.js: npm start  OR  node index.js  OR  npx vite --host 0.0.0.0
     Python:  uvicorn main:app --host 0.0.0.0 --port 8000  OR  python app.py
   For long-running processes use nohup:
     nohup npm start > /tmp/app.log 2>&1 & echo "PID=$!"
6. VERIFY — after starting, wait 3 s then probe:
     curl -s -o /dev/null -w "%{http_code}" http://localhost:<port>/
   A 200, 301, or 302 response means the app is alive.
7. DEBUG LOOP — if it fails:
     - Read /tmp/app.log (or stderr captured earlier)
     - Diagnose the root cause from the error message
     - Fix it: set an env var, patch a dependency, or rewrite a config.
       For any fix that writes more than one line of content, ALWAYS use
       run_command with a heredoc, NOT write_file. Example:
         cmd: "cat << 'EOF' > /home/user/repo/.env\nPORT=3000\nNODE_ENV=development\nEOF"
     - Restart and re-verify
8. EFFICIENCY — you have at most 10 actions. Do not waste steps:
     - Combine multiple commands with && where possible
     - Do not explore directories that are not relevant
     - Do not read files that will not help you fix the error
9. FINISH — call `done` immediately once curl confirms the app is live.
   Never leave the loop running unnecessarily.

The sandbox ID will be provided in your first user message.
The live preview URL for the user will be constructed as: https://<port>-<sandbox_id>.e2b.dev
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# §2  E2B Tool Implementations
# ─────────────────────────────────────────────────────────────────────────────

def run_command(
    cmd: str,
    sandbox: Sandbox,
    on_output: Callable[[str], None] | None = None,
) -> str:
    """
    Execute a shell command inside the E2B sandbox with real-time output.

    stdout and stderr are streamed to `on_output` as they arrive (for live
    terminal display) and also accumulated into the return string so the LLM
    can read the full output in its next observation.

    Returns a string of the form:
      "[exit_code: N]\\n<combined stdout + stderr>"
    """
    output_lines: list[str] = []

    def _handle_stdout(message) -> None:
        line = message.line if hasattr(message, "line") else str(message)
        output_lines.append(line)
        if on_output:
            on_output(line)

    def _handle_stderr(message) -> None:
        line = message.line if hasattr(message, "line") else str(message)
        tagged = f"[stderr] {line}"
        output_lines.append(tagged)
        if on_output:
            on_output(tagged)

    try:
        result = sandbox.commands.run(
            cmd,
            on_stdout=_handle_stdout,
            on_stderr=_handle_stderr,
            timeout=COMMAND_TIMEOUT,
        )
        exit_code = getattr(result, "exit_code", 0)
        output    = "\n".join(output_lines) if output_lines else "(no output)"

        # ── Fix 2: Context-window guard ───────────────────────────────────────
        # npm install / pip install can produce tens of thousands of tokens.
        # Errors almost always appear at the END of a log, so we keep the tail.
        MAX_OUTPUT_CHARS = 6_000
        if len(output) > MAX_OUTPUT_CHARS:
            dropped = len(output) - MAX_OUTPUT_CHARS
            output  = f"[... {dropped:,} chars truncated — showing last {MAX_OUTPUT_CHARS:,} chars ...]\n" + output[-MAX_OUTPUT_CHARS:]

        return f"[exit_code: {exit_code}]\n{output}"

    except Exception as exc:
        err_msg = f"[command_exception] {type(exc).__name__}: {exc}"
        if on_output:
            on_output(err_msg)
        return err_msg


def read_file(path: str, sandbox: Sandbox) -> str:
    """
    Read a file from the E2B sandbox filesystem.

    Returns the file content as a string (UTF-8), or an error tag on failure.
    Truncated to 8 000 chars to fit the LLM context window comfortably.
    """
    try:
        content = sandbox.files.read(path)
        if content is None:
            return "(file is empty)"
        # Truncate to avoid blowing the LLM context window
        if len(content) > 8_000:
            content = content[:8_000] + f"\n... [truncated — {len(content)} total chars]"
        return content
    except Exception as exc:
        return f"[read_error] {type(exc).__name__}: {exc}"


def write_file(path: str, content: str, sandbox: Sandbox) -> str:
    """
    Write content to a file in the E2B sandbox filesystem.

    Returns a success tag or an error tag.
    """
    try:
        sandbox.files.write(path, content)
        return f"[ok] Wrote {len(content):,} bytes → {path}"
    except Exception as exc:
        return f"[write_error] {type(exc).__name__}: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# §3  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _emit(token_queue: _queue.Queue, event: dict) -> None:
    """Serialise an event dict to JSON and put it on the queue for SSE streaming."""
    token_queue.put(json.dumps(event))


def _emit_terminal(token_queue: _queue.Queue, text: str) -> None:
    _emit(token_queue, {"type": "terminal", "text": text})


def _emit_thought(token_queue: _queue.Queue, text: str) -> None:
    _emit(token_queue, {"type": "thought", "text": text})


def _parse_llm_json(raw: str) -> dict:
    """
    Extract and parse the JSON object from the LLM response.

    Handles the two most common model quirks:
      1. Response wrapped in ```json … ``` markdown fences.
      2. A short preamble text before the opening brace.

    Raises json.JSONDecodeError if no valid JSON object can be found.
    """
    text = raw.strip()

    # Strip markdown fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$",       "", text)
        text = text.strip()

    # Skip any preamble to find the first opening brace
    brace_start = text.find("{")
    if brace_start > 0:
        text = text[brace_start:]

    # Walk to find the matching closing brace (handles nested objects)
    depth = 0
    end   = -1
    for i, ch in enumerate(text):
        if   ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end > 0:
        text = text[:end]

    return json.loads(text)


# ─────────────────────────────────────────────────────────────────────────────
# §4  The ReAct Loop — run_qa_agent
# ─────────────────────────────────────────────────────────────────────────────

def run_qa_agent(repo_url: str, token_queue: _queue.Queue) -> None:
    """
    Main entry point for Agent 5: The QA Engineer.

    Designed to run in a thread-pool worker via:
        asyncio.get_running_loop().run_in_executor(None, run_qa_agent, url, q)

    Algorithm:
        1.  Spin up an E2B sandbox (cloud microVM).
        2.  Seed the LLM conversation with the repo URL + sandbox ID.
        3.  Loop (max MAX_ITERATIONS):
              a. Call Groq LLM → parse JSON {thought, action}.
              b. Emit the thought to the Brain panel.
              c. Dispatch the action to the matching tool function.
              d. Stream all stdout/stderr in real-time to the Terminal panel.
              e. Feed the observation back as the next user message.
              f. If tool == "done" and status == "success": emit app_url + exit.
              g. If tool == "done" and status == "failure": emit summary + exit.
        4.  If MAX_ITERATIONS reached: emit failure summary.
        5.  Always kill the sandbox in the `finally` block.
        6.  Always put None on the queue last (SSE sentinel for server.py).

    All unhandled exceptions are caught, surfaced as {"type":"error"}, and the
    sentinel is still put on the queue so the HTTP stream closes cleanly.
    """
    sandbox: Sandbox | None = None
    # Fix 1: flag set to True when the agent successfully launches the app.
    # The finally block skips sandbox.kill() when True — the microVM must
    # remain alive so the user can interact with the live preview iframe.
    # E2B automatically reclaims the VM after SANDBOX_TIMEOUT seconds.
    keep_sandbox_alive: bool = False

    try:
        # ── §4.1  Sandbox initialisation ─────────────────────────────────────
        _emit_terminal(token_queue, "Requesting E2B cloud microVM…")

        sandbox    = Sandbox.create()
        sandbox_id = sandbox.sandbox_id

        _emit_terminal(token_queue, f"✓  Sandbox ready  |  ID: {sandbox_id}")
        _emit_terminal(token_queue, f"🔗  Repo target   |  {repo_url}")
        _emit_terminal(token_queue, "─" * 60)

        # ── §4.2  Seed conversation ───────────────────────────────────────────
        messages: list[dict] = [
            {
                "role":    "user",
                "content": (
                    f"Repository URL : {repo_url}\n"
                    f"Sandbox ID     : {sandbox_id}\n\n"
                    "The sandbox is a fresh Ubuntu 22.04 microVM. Git, Python 3, "
                    "Node.js, and npm are pre-installed. Begin — clone the repository "
                    "and get the application running."
                ),
            }
        ]

        # ── §4.3  ReAct loop ──────────────────────────────────────────────────
        for iteration in range(1, MAX_ITERATIONS + 1):

            _emit_terminal(
                token_queue,
                f"\n{'─'*20}  Iteration {iteration}/{MAX_ITERATIONS}  {'─'*20}",
            )

            # ── LLM call via api_gateway (key pool + failover) ────────────────
            def _llm_call(client):  # noqa: E306
                resp = client.chat.completions.create(
                    model       = QA_MODEL,
                    max_tokens  = 1_024,
                    temperature = 0.15,         # low temp = deterministic tool dispatch
                    messages    = [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        *messages,
                    ],
                )
                return resp.choices[0].message.content.strip()

            try:
                raw_response: str = api_gateway.groq_execute(
                    _llm_call, context=f"Agent5-QA-iter{iteration}"
                )
            except Exception as exc:
                _emit(token_queue, {
                    "type": "error",
                    "text": f"LLM call failed on iteration {iteration}: {exc}",
                })
                break  # exit loop → final summary below

            # ── Parse LLM JSON ────────────────────────────────────────────────
            try:
                parsed = _parse_llm_json(raw_response)
            except (json.JSONDecodeError, ValueError) as exc:
                _emit(token_queue, {
                    "type": "terminal",
                    "text": (
                        f"⚠  LLM returned non-JSON (iter {iteration}) — self-correcting…\n"
                        f"Raw: {raw_response[:300]}"
                    ),
                })
                # Append the malformed response and tell the LLM to fix it
                messages.append({"role": "assistant", "content": raw_response})
                messages.append({
                    "role":    "user",
                    "content": (
                        f"PARSE ERROR: Your response was not valid JSON.\n"
                        f"Error: {exc}\n\n"
                        "You MUST respond with ONLY a single JSON object. "
                        "No markdown, no preamble. Try again."
                    ),
                })
                continue   # burn an iteration, try to self-correct

            thought = parsed.get("thought", "")
            action  = parsed.get("action",  {})
            tool    = action.get("tool",    "")
            args    = action.get("args",    {})

            # Record the assistant turn in conversation history
            messages.append({"role": "assistant", "content": raw_response})

            # ── Emit thought to Brain panel ───────────────────────────────────
            if thought:
                _emit_thought(token_queue, thought)

            # ── Tool dispatch ─────────────────────────────────────────────────
            observation: str

            if tool == "run_command":
                cmd = args.get("cmd", "").strip()
                if not cmd:
                    observation = "[error] run_command called with empty cmd."
                    _emit_terminal(token_queue, observation)
                else:
                    _emit_terminal(token_queue, f"$ {cmd}")
                    # Build a closure so the lambda captures the current queue ref
                    def _on_output(line: str, _q=token_queue) -> None:
                        _emit_terminal(_q, line)

                    observation = run_command(cmd, sandbox, on_output=_on_output)

            elif tool == "read_file":
                path = args.get("path", "").strip()
                _emit_terminal(token_queue, f"📄  cat {path}")
                observation = read_file(path, sandbox)
                # Show a truncated preview in the terminal (full text goes to LLM)
                preview = observation if len(observation) <= 500 else observation[:500] + "…"
                _emit_terminal(token_queue, preview)

            elif tool == "write_file":
                path    = args.get("path",    "").strip()
                content = args.get("content", "")
                _emit_terminal(token_queue, f"✍   write → {path}")
                observation = write_file(path, content, sandbox)
                _emit_terminal(token_queue, observation)

            elif tool == "done":
                # ── Agent signals completion ──────────────────────────────────
                status  = args.get("status",  "failure")
                summary = args.get("summary", "Agent completed without a summary.")
                port    = args.get("port")

                if status == "success" and port:
                    app_url = f"https://{port}-{sandbox_id}.e2b.dev"
                    _emit_terminal(token_queue, f"\n🚀  App is live!")
                    _emit_terminal(token_queue, f"    URL  → {app_url}")
                    _emit(token_queue, {"type": "app_url", "url": app_url})
                    # Fix 1: keep the microVM alive so the user can browse the
                    # live preview. The finally block checks this flag before
                    # calling sandbox.kill(). E2B reclaims after SANDBOX_TIMEOUT.
                    keep_sandbox_alive = True
                else:
                    _emit_terminal(
                        token_queue,
                        f"\n⚠   Agent finished with status: {status.upper()}",
                    )

                _emit(token_queue, {"type": "done", "summary": summary})
                return   # ← drops into finally; sandbox killed only if not alive

            else:
                # Unknown tool — tell the LLM and let it correct
                observation = (
                    f"[unknown_tool] '{tool}' is not a valid tool. "
                    f"Valid tools: run_command, read_file, write_file, done."
                )
                _emit_terminal(token_queue, observation)

            # ── Feed observation back as next user message ────────────────────
            messages.append({
                "role":    "user",
                "content": f"Observation:\n{observation}",
            })

        # ── Fell through MAX_ITERATIONS without calling `done` ────────────────
        _emit_terminal(
            token_queue,
            f"\n⛔  Maximum iterations ({MAX_ITERATIONS}) reached — "
            "application did not start within the iteration budget.",
        )
        _emit(token_queue, {
            "type":    "done",
            "summary": (
                f"Agent 5 exhausted all {MAX_ITERATIONS} iterations "
                "without successfully launching the application. "
                "Review the terminal output above for the last known error."
            ),
        })

    except Exception as exc:
        tb = traceback.format_exc()
        _emit(token_queue, {
            "type": "error",
            "text": f"Fatal agent error: {type(exc).__name__}: {exc}\n\n{tb}",
        })
        _emit(token_queue, {
            "type":    "done",
            "summary": f"Agent 5 crashed unexpectedly: {exc}",
        })

    finally:
        # ── Conditional sandbox teardown (Fix 1) ─────────────────────────────
        # On success: keep_sandbox_alive is True — the user's preview iframe is
        #   still pointing at this VM. Do NOT kill it. E2B will auto-expire it
        #   after SANDBOX_TIMEOUT seconds with zero action required from us.
        # On failure or crash: kill immediately to avoid billing for idle VMs.
        if sandbox is not None:
            if keep_sandbox_alive:
                _emit_terminal(
                    token_queue,
                    f"\n🟢  Sandbox kept alive for preview (auto-expires in "
                    f"{SANDBOX_TIMEOUT}s).",
                )
            else:
                try:
                    sandbox.kill()
                    _emit_terminal(token_queue, "\n🔒  Sandbox terminated cleanly.")
                except Exception as kill_exc:
                    _emit_terminal(token_queue, f"⚠   Sandbox kill failed: {kill_exc}")

        # ── Sentinel value — tells server.py's SSE generator the stream is over
        token_queue.put(None)