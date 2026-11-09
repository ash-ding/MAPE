"""In-container proposer entrypoint.

Launched by `evolution/proposer.py` via:

    singularity exec --containall --bind ... proposer-base.sif \\
        python /app/evolution/propose_in_container.py \\
            --new-id <int> --workspace /workspace [--mode propose|fix] \\
            --system-prompt-file /workspace/harnesses/<id>/.prompt_system.txt \\
            --task-file          /workspace/harnesses/<id>/.prompt_task.txt   \\
            --model <model> --timeout-s <s>

Inside the Singularity sandbox we have (v10 SELECTIVE binds — see
evolution/proposer.py::_build_singularity_cmd for the authoritative list):
  /app/evolution/{__init__,propose_in_container,memo_class}.py   RO
  /app/core/{__init__,memo_class,recorder,llm,logger}.py   RO
  /app/benchmarks/       full pkg RO (env/workflow/prompts + raw data)
  /workspace           this run's workspace, RW  (cwd for the agent)
  /usr/local/bin/claude         host's claude binary, RO bind
  /usr/local/share/claude/      host's claude install, RO bind
  /root                         per-run scratch HOME, RW (no credentials;
                                claude_code authenticates with
                                ANTHROPIC_API_KEY from the environment)

The agent's filesystem-readable surface is exactly the binds above; nothing
else from the host is visible (`--containall` strips $HOME, /tmp, etc).

# Prompt staging

System prompt + task prompt are rendered HOST-SIDE by
`evolution.proposer._render_and_stage_prompts(...)` and written into the
harness dir as `.prompt_system.txt` / `.prompt_task.txt`. This script does
NOT import `evolution.prompts.*`; it just reads those two files, so each
harness dir captures the exact prompt that produced it.

# Agent CLI

The script shells out to the Claude Code CLI in stream-json mode and parses
its NDJSON events from stdout (`_build_cmd_cc`, `_build_stdin_cc`,
`_handle_event_cc`).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Callable, List
from datetime import datetime
from typing import Optional

#: Kept in step with evolution.paths — this script runs inside the proposer
#: sandbox, which mounts only propose_in_container.py and memo_class.py from
#: evolution, so it cannot import that module.
ENTRY_FILE = "memo.py"


def entry_file(harness_dir: Path):
    """The harness's interface file, or None if it has none."""
    candidate = harness_dir / ENTRY_FILE
    return candidate if candidate.is_file() else None



# Tool restrictions intentionally NOT set: filesystem isolation is enforced
# by the Singularity sandbox in evolution/proposer.py — the agent's reachable
# surface is exactly the bind list (workspace + project root RO + agent
# CLI binary RO + per-run scratch HOME). Bash, WebFetch, WebSearch, mcp__*
# all permitted inside the container; the proposer image (extends
# eval-base.sif) has the same Python env as the evaluator, so the agent can
# `python -c "from harness import ..."` for sub-second import validation.
# See PROPOSER_SYSTEM for the workflow cheat sheet.


def _emit(prefix: str, msg: str) -> None:
    """Print a tagged line to stdout. Host-side launcher line-by-line
    forwards these into the orchestrator log.

    Tag conventions:
      proposer·text    agent assistant text — emitted line-by-line, full
                       content (one log line per source line; rendered at
                       INFO so the reasoning is visible without --verbose)
      proposer·tool    agent tool use (truncated to 140 chars)
      proposer·info    high-level status (model, finish, usage)
      proposer·error   actionable failure
    """
    # Use sys.stdout.write to avoid print's locking subtleties under asyncio
    sys.stdout.write(f"{prefix}: {msg}\n")
    sys.stdout.flush()


# ============================================================================
# Claude Code: build argv + build stdin + handle event
# ============================================================================

# ---- claude_code ----------------------------------------------------------

def _build_cmd_cc(
    *,
    system_prompt_file: str,
    model: str,
    agent_opts: dict,
) -> List[str]:
    cmd = [
        "claude", "-p",
        # --verbose is REQUIRED when --output-format=stream-json in -p mode.
        "--verbose",
        "--output-format", "stream-json",
        "--input-format", "stream-json",
        "--model", model,
        # --system-prompt-file is the file-based variant of --system-prompt;
        # avoids embedding a ~35 KB blob in argv (which `ps` would expose) and
        # makes the prompt visible to forensic diffs at <harness>/.prompt_system.txt.
        #
        # REPLACES the CLI's own system prompt rather than appending to it
        # (--append-system-prompt would append), and that is the point: the
        # file staged beside each harness is then the WHOLE prompt that
        # produced it. Appending would make the effective prompt depend on
        # which `claude` build the image happens to carry, so two harnesses
        # from different weeks would not be comparable and the archived file
        # would no longer say why. The same reasoning rules out moving this
        # text to a CLAUDE.md in the workspace root: /workspace is read-write
        # to the proposer, so a candidate could edit the instructions that
        # constrain the candidates after it.
        "--system-prompt-file", system_prompt_file,
        "--permission-mode", "bypassPermissions",
        # `--disallowed-tools mcp__*` filters the TOOLS; this stops the CLI
        # consulting any MCP config at all. Inert today — --containall gives
        # the agent a scratch /root with no config to read — but it pins the
        # intent so a future bind of a real home cannot quietly reintroduce
        # host MCP servers.
        "--strict-mcp-config",
    ]
    disallowed = agent_opts.get("disallowed_tools") or []
    if disallowed:
        cmd += ["--disallowed-tools", ",".join(disallowed)]
    effort = agent_opts.get("effort")
    if effort:
        # CC --effort: low / medium / high / xhigh / max
        cmd += ["--effort", str(effort)]
    return cmd


def _build_stdin_cc(task: str) -> bytes:
    """One NDJSON user message line. CC -p with --input-format stream-json
    reads events from stdin until EOF."""
    return (
        json.dumps({"type": "user", "message": {"role": "user", "content": task}})
        + "\n"
    ).encode("utf-8")


# CC tool counter (state for _handle_event_cc; module-level since handler
# is called per-line on the event loop's single thread).
_cc_tool_count = 0


def _handle_event_cc(event: dict) -> None:
    """Parse one CC stream-json event → _emit() lines.

    Event types we recognize:
      assistant   — message with content blocks (text / thinking / tool_use)
      result      — final summary (duration, cost, turns, usage)
      system      — startup info; logged briefly
      rate_limit_event / others — silently ignored
    """
    global _cc_tool_count
    etype = event.get("type", "")
    if etype == "assistant":
        for block in event.get("message", {}).get("content", []) or []:
            btype = block.get("type")
            if btype == "text":
                text = (block.get("text") or "").strip()
                if text:
                    # Emit each line separately so orchestrator.log preserves
                    # paragraph structure.
                    for line in text.splitlines():
                        _emit("proposer·text", line)
            elif btype == "tool_use":
                _cc_tool_count += 1
                _emit(
                    "proposer·tool",
                    f"#{_cc_tool_count} {block.get('name','')} "
                    f"{str(block.get('input',''))[:140]}",
                )
            # "thinking" blocks: skipped.
    elif etype == "result":
        cost = event.get("total_cost_usd") or 0
        _emit(
            "proposer·info",
            f"finished turns={event.get('num_turns','?')} "
            f"tools={_cc_tool_count} "
            f"duration={event.get('duration_ms','?')}ms "
            f"cost=${cost:.4f} "
            f"usage={event.get('usage','')}",
        )
        _record_proposer_run(event, tools=_cc_tool_count)
        if event.get("is_error"):
            _emit("proposer·error", f"agent reported is_error=true: {event.get('result','')}")
    # else: system / rate_limit_event / unknown — skip silently.


#: Where the proposer's own cost is written, inside the harness dir so it
#: travels with the harness through the rename to its content hash. A
#: dotfile the hash excludes (evolution.codetree._NON_CODE_FILES): what the
#: proposal cost to write is not part of what it is.
PROPOSER_RECORD = ".proposer.json"
_PROPOSER_RECORD_PATH: Optional[Path] = None
_PROPOSER_RECORD_MODE: str = "propose"

#: The proposer's FULL session, next to the cost record: every NDJSON event
#: the agent CLI streamed (Claude Code `stream-json`), one
#: per line, verbatim — tool inputs and results, assistant text, usage.
#: orchestrator.log keeps only the first ~140 characters of each tool call;
#: this is where "what did the proposer actually read and think" lives.
#: Runs are appended (the propose, then each sanity-fix attempt), each
#: bracketed by a `mape.session_start` / `mape.session_end` marker
#: line. Hash-excluded like the cost record (evolution.codetree._NON_CODE_FILES)
#: and not copied into the seed cache — a transcript of writing the harness
#: is not part of what the harness is.
PROPOSER_SESSION = ".proposer_session.jsonl"
_SESSION_FH = None


def _session_open(path: Path, *, mode: str, agent: str, new_id: str) -> None:
    """Start appending this run's raw agent events to `path`."""
    global _SESSION_FH
    _session_close(rc=None)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _SESSION_FH = path.open("a", encoding="utf-8")
        _session_write(json.dumps({"type": "mape.session_start", "mode": mode, "agent": agent,
                                   "new_id": new_id, "at": datetime.now().isoformat(timespec="seconds")}))
    except Exception as exc:                  # bookkeeping: never fail the proposal for it
        _SESSION_FH = None
        _emit("proposer·info", f"session transcript not recorded: {exc}")


def _session_write(line: str) -> None:
    """One raw line (an NDJSON event, or whatever else the CLI printed)."""
    if _SESSION_FH is None:
        return
    try:
        _SESSION_FH.write(line + "\n")
        _SESSION_FH.flush()
    except Exception:
        pass


def _session_close(*, rc: Optional[int]) -> None:
    global _SESSION_FH
    if _SESSION_FH is None:
        return
    try:
        _session_write(json.dumps({"type": "mape.session_end", "rc": rc,
                                   "at": datetime.now().isoformat(timespec="seconds")}))
        _SESSION_FH.close()
    except Exception:
        pass
    _SESSION_FH = None


def _record_proposer_run(event: dict, *, tools: int) -> None:
    """Append this invocation's cost to <harness>/.proposer.json: one entry
    per proposer run (the propose, then each sanity-fix attempt), so the
    history can say what a harness cost to write, not just what it scored."""
    if _PROPOSER_RECORD_PATH is None:
        return
    path = _PROPOSER_RECORD_PATH
    try:
        record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"runs": []}
    except Exception:
        record = {"runs": []}
    usage = event.get("usage") or {}
    record.setdefault("runs", []).append({
        "mode": _PROPOSER_RECORD_MODE,
        "turns": event.get("num_turns"),
        "scripts": int(tools),
        "duration_ms": event.get("duration_ms"),
        "cost_usd": float(event.get("total_cost_usd") or 0.0),
        "usage": usage if isinstance(usage, dict) else str(usage),
        "is_error": bool(event.get("is_error")),
        "at": datetime.now().isoformat(timespec="seconds"),
    })
    try:
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:  # the record is bookkeeping; never fail the proposal for it
        _emit("proposer·raw", f"could not write {path}: {exc!r}")


# ============================================================================
# Main loop
# ============================================================================


async def _run(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace)
    new_dir = workspace / f"harnesses/{args.new_id}"
    # Where this run's cost and transcript land. In `analyze` mode the
    # harness directory is READ-ONLY — the frontier already points at its
    # content hash — so the host passes paths under diagnostics/ instead.
    record_json = Path(args.record_json) if args.record_json else new_dir / PROPOSER_RECORD
    session_jsonl = (Path(args.session_jsonl) if args.session_jsonl
                     else new_dir / PROPOSER_SESSION)
    global _PROPOSER_RECORD_PATH, _PROPOSER_RECORD_MODE
    _PROPOSER_RECORD_PATH, _PROPOSER_RECORD_MODE = record_json, args.mode
    _session_open(session_jsonl, mode=args.mode, agent="claude_code", new_id=args.new_id)

    # Sanity: in fix mode the harness must already exist. (In propose mode
    # the host launcher created the dir; both .prompt_*.txt files live inside
    # so by definition the dir is there.)
    if args.mode == "fix" and entry_file(new_dir) is None:
        _emit("proposer·error", f"fix mode but {new_dir}/{ENTRY_FILE} missing")
        return 2

    # Prompts are pre-rendered HOST-SIDE — just read the two staged files.
    try:
        # Claude Code reads the system prompt itself (--system-prompt-file);
        # reading it here only checks that the staged file exists.
        Path(args.system_prompt_file).read_text(encoding="utf-8")
        task = Path(args.task_file).read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        _emit("proposer·error", f"missing staged prompt file: {exc}")
        return 7

    # Parse the agent options (JSON dict). Empty / missing → {}. It carries
    # `effort` and `disallowed_tools`; unknown keys are ignored by the argv
    # builder, so adding new keys is forward-compatible.
    try:
        agent_opts: dict = json.loads(args.agent_opts or "{}")
    except json.JSONDecodeError as exc:
        _emit("proposer·error", f"--agent-opts is not valid JSON: {exc}")
        return 6
    if not isinstance(agent_opts, dict):
        _emit("proposer·error", f"--agent-opts must decode to a dict, got {type(agent_opts).__name__}")
        return 6

    cmd = _build_cmd_cc(
        system_prompt_file=args.system_prompt_file,
        model=args.model,
        agent_opts=agent_opts,
    )
    stdin_bytes = _build_stdin_cc(task)

    if agent_opts:
        _emit("proposer·info", f"agent_opts={agent_opts}")
    _emit(
        "proposer·info",
        f"mode={args.mode} new_id={args.new_id} model={args.model}",
    )

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(workspace),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        # Claude Code emits NDJSON events one event per line. A single line can
        # easily exceed asyncio's default 64 KB readline limit — e.g. a
        # `tool_result` block containing `cat` output over a multi-MB dataset
        # file. Without raising this limit, `async for raw in proc.stdout`
        # raises LimitOverrunError and the whole propose silently dies. 16 MB
        # is generous; any tool_result above that means the agent is doing
        # something pathological that we'd want to fail loudly anyway.
        limit=16 * 1024 * 1024,
    )

    assert proc.stdin is not None
    try:
        proc.stdin.write(stdin_bytes)
        await proc.stdin.drain()
    except (BrokenPipeError, ConnectionResetError) as exc:
        _emit("proposer·error", f"failed writing stdin: {exc}")
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass

    handler: Callable[[dict], None] = _handle_event_cc

    async def _pump() -> None:
        assert proc.stdout is not None
        async for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip()
            if not line:
                continue
            _session_write(line)               # the full event, before any truncation
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # Non-JSON output — stderr leaking through, or pre-init
                # banner from the binary. Log a truncated form.
                _emit("proposer·text", line[:500])
                continue
            try:
                handler(event)
            except Exception as exc:
                _emit("proposer·error", f"event handler error: {exc} on {line[:200]}")

    try:
        await asyncio.wait_for(_pump(), timeout=args.timeout_s)
    except asyncio.TimeoutError:
        _emit("proposer·error", f"timed out after {args.timeout_s}s for {args.new_id}")
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()
        _session_close(rc=3)
        return 3

    rc = await proc.wait()
    _session_close(rc=rc)

    if args.mode == "propose":
        sentinel = new_dir / "PROPOSAL_READY"
        if not sentinel.exists():
            _emit(
                "proposer·error",
                f"finished without writing PROPOSAL_READY at {new_dir}",
            )
            return 4

    return rc


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--new-id", required=True)
    p.add_argument("--workspace", required=True,
                   help="Container path of the per-run workspace (typically /workspace).")
    p.add_argument("--record-json", default=None,
                   help="Where to append this run's cost record "
                        "(default: <harness>/.proposer.json).")
    p.add_argument("--session-jsonl", default=None,
                   help="Where to append this run's raw agent transcript "
                        "(default: <harness>/.proposer_session.jsonl).")
    p.add_argument("--mode", choices=["propose", "fix", "analyze"], default="propose",
                   help="propose: enforce PROPOSAL_READY on exit. fix: skip that check.")
    p.add_argument("--system-prompt-file", required=True,
                   help="Path inside the container to the host-rendered system prompt "
                        "(typically /workspace/harnesses/<id>/.prompt_system.txt).")
    p.add_argument("--task-file", required=True,
                   help="Path inside the container to the host-rendered task prompt "
                        "(/workspace/harnesses/<id>/.prompt_task.txt). In fix mode this "
                        "file already has the error trace folded in by proposer_fix_prompt.")
    # Mirrors evolution/orchestrator.py::DEFAULT_CONFIG. This script runs inside the
    # container and imports nothing from evolution, so the value is repeated
    # rather than shared. The orchestrator always passes --model explicitly;
    # this default only covers running the script by hand.
    p.add_argument("--model", default="claude-opus-4-8")
    # Mirrors evolution/orchestrator.py::DEFAULT_CONFIG. This script cannot
    # import from evolution, so the value is repeated; the orchestrator always
    # passes --timeout-s explicitly and this only covers a hand-run.
    p.add_argument("--timeout-s", type=int, default=45 * 60)
    p.add_argument("--agent-opts", default="{}",
                   help="JSON dict of Claude Code options: effort, disallowed_tools. "
                        "Unknown keys are ignored.")
    args = p.parse_args()
    rc = asyncio.run(_run(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
