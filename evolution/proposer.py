"""Host-side launcher for the Singularity-sandboxed proposer.

This module spawns `evolution/propose_in_container.py` inside `proposer-base.sif`
with a SELECTIVE bind list — CC's reachable surface inside the container is
exactly the bind set built by `_build_singularity_cmd` (see that function's
docstring for the per-bind rationale). Nothing else from the host is visible.


In-container runtime (set by propose_in_container.py from CLI args):
  cwd                        /workspace = workspace/<run_id>/
  PYTHONPATH                 /app  (so core.* / datasets.* import cleanly)
  allowed_tools              not set (= SDK default = full toolset)
  disallowed_tools           configurable; default ["mcp__*"]. See
                             cfg["proposer"]["disallowed_tools"] in
                             orchestrator.py / DEFAULT_CONFIG.
  system_prompt              build_proposer_system(sanity_enabled), where
                             sanity_enabled reflects this run's actual
                             sanity-layer behavior.
  timeout_s                  wall-clock cap; enforced via subprocess.wait
                             + SIGTERM/SIGKILL escalation on the singularity
                             exec process group. This is the hard limit.

The proposer ONLY writes files (into /workspace, RW). It never executes the
harness end-to-end — evaluation happens later in a separate Singularity
container (evolution/evaluator.py). It MAY do `python -c "..."` for import-level
self-validation; this is encouraged by PROPOSER_SYSTEM.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from evolution.paths import (
    ENTRY_FILE,
    entry_file,
    PROJECT_ROOT,
    PROPOSER_BASE_SIF,
    paths,
)
from evolution.prompts import (
    build_proposer_system,
    proposer_fix_prompt,
    proposer_task_prompt,
)

try:
    from dotenv import dotenv_values
except ImportError:
    def dotenv_values(path):  # type: ignore
        out = {}
        try:
            with open(path) as f:
                for line in f:
                    s = line.strip()
                    if s and not s.startswith("#") and "=" in s:
                        k, _, v = s.partition("=")
                        out[k.strip()] = v.strip().strip('"').strip("'")
        except FileNotFoundError:
            pass
        return out

log = logging.getLogger("evolution.proposer")


# ---------------------------------------------------------------------------
# Bind / launch helpers
# ---------------------------------------------------------------------------

# Host path to the Claude Code CLI: a native binary plus a versions/ subtree
# under ~/.local/share/claude/. The in-container script invokes
# `claude -p --output-format stream-json …` directly (no SDK dependency).
#
# Claude Code authenticates with ANTHROPIC_API_KEY only (host env, then the
# project .env), injected as SINGULARITYENV_ANTHROPIC_API_KEY. No login file,
# OAuth token or cloud-provider route is supported.
_HOME = Path.home()
_HOST_CLAUDE_BIN = _HOME / ".local" / "bin" / "claude"
_HOST_CLAUDE_SHARE = _HOME / ".local" / "share" / "claude"


class ProposerLaunchError(RuntimeError):
    """Raised when the proposer subprocess cannot be launched or exits abnormally."""


def _resolve_anthropic_api_key() -> str:
    """ANTHROPIC_API_KEY: host env first, then the project .env (same
    convention the evaluator uses for container keys). Returns "" when
    neither source has it — caller decides how to fail."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        return key
    return dotenv_values(str(PROJECT_ROOT / ".env")).get("ANTHROPIC_API_KEY", "") or ""


def _check_environment() -> None:
    """Pre-flight checks. Raise with actionable messages so the user knows
    exactly what to install or configure."""
    if not PROPOSER_BASE_SIF.exists():
        raise ProposerLaunchError(
            f"proposer-base.sif missing at {PROPOSER_BASE_SIF}. "
            f"Build it once with:\n"
            f"  PATH=$HOME/.local/bin:$PATH singularity build "
            f"{PROPOSER_BASE_SIF} containers/proposer-base.def"
        )
    if not _HOST_CLAUDE_BIN.exists():
        raise ProposerLaunchError(
            f"host claude CLI not found at {_HOST_CLAUDE_BIN}. "
            f"Install Claude Code and ensure `which claude` resolves to it."
        )
    if not _resolve_anthropic_api_key():
        raise ProposerLaunchError(
            "ANTHROPIC_API_KEY not found. Claude Code runs through the "
            "Anthropic API: export the key in the host environment or add "
            "it to the project .env."
        )


def _prepare_claude_home() -> Path:
    """Stage a writable per-run HOME for the in-container coding agents.

    `claude` appends to log files and cache state under $HOME/.claude/, so
    `<workspace>/.proposer_home/` is bound as /root inside the container. It
    holds no credential: Claude Code authenticates with the ANTHROPIC_API_KEY
    injected by `_auth_extra_env`, and a login file left by an older
    workspace is removed so it cannot be picked up instead.
    """
    proposer_home = paths.workspace / ".proposer_home"
    claude_dir = proposer_home / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    stale = claude_dir / ".credentials.json"
    if stale.exists():
        try:
            stale.unlink()
        except OSError as exc:
            log.warning(f"could not remove stale credentials file: {exc}")
    return proposer_home


def _auth_extra_env() -> Dict[str, str]:
    """The per-launch secret env for `_stream_subprocess`: ANTHROPIC_API_KEY
    as SINGULARITYENV_ANTHROPIC_API_KEY. Singularity strips the prefix inside
    the container; nothing lands in argv, and the key is re-read on every
    launch so a rotated key is picked up immediately."""
    api_key = _resolve_anthropic_api_key()
    if not api_key:
        # _check_environment already validated this; defensive.
        raise ProposerLaunchError("ANTHROPIC_API_KEY is not set (host env or project .env)")
    return {"SINGULARITYENV_ANTHROPIC_API_KEY": api_key}


def _prepare_auth() -> "tuple[Path, Dict[str, str]]":
    """Auth staging shared by analyze() / propose() / propose_with_fix():
    the per-run scratch HOME and the secret-carrying SINGULARITYENV_* env."""
    return _prepare_claude_home(), _auth_extra_env()


def _build_singularity_cmd(
    *,
    propose_args: List[str],
    proposer_home: Path,
    extra_binds: Optional[List[str]] = None,
    workspace_mode: str = "rw",
) -> List[str]:
    """Build the `singularity exec ...` argv with the selective bind list.

    Bind list rationale:
      - LAYER-1 wrapper script (propose_in_container.py + its evolution.prompts
        import) needs to be reachable via /app/evolution/. Agents never read these.
      - Agent reference materials live under /app/{common,datasets,evolution}/. Only
        the files that contribute to writing a correct harness are bound:
          evolution/memo_class.py    the MemoClass base CC must inherit from
          core/memo_class.py     the underlying ABC (the evolution loop's subclasses it)
          core/recorder.py       Basic_Recorder data envelope
                                   (memo_class imports it)
          core/llm.py            Agent + Embedding helpers (token-tracked)
          core/logger.py         transitive dep of core.llm
          core/qa.py             QAContext — how the HOST sends the QA call;
                                   the prompt tells the proposer to read it to
                                   see the wire format its context lands in
          core/model_policy.py   transitive dep of core.llm: Agent()/Embedding()
                                   consult it at construction (the harness's
                                   import-level self-check builds them)
          core/__init__.py       package marker
          benchmarks/              full directory: env.py, workflow.py,
                                   prompts.py per benchmark + raw data files.
                                   No selective filtering.
      - workspace/ is the agent's cwd (RW). Per-run isolation.
      - The claude binary + ~/.local/share/claude, and the scratch HOME at
        /root (proposer_home; auth via the ANTHROPIC_API_KEY env var)
      - NOT bound (vs the previous /app:ro whole-root bind):
          /app/evolution/{orchestrator,proposer,...}.py    host-side outer-loop
          /app/core/{workflow,judge,tokens}.py        internal infrastructure
          /app/baselines, /app/docs, /app/configs, /app/seeds, /app/venv
          /app/workspace                                cross-run isolation
          /app/.env, /app/.git                          not needed
    """
    binds = [
        # LAYER-1 wrapper script imports. Note: evolution/prompts/ is intentionally
        # NOT bound — prompts are rendered HOST-SIDE by `_render_and_stage_prompts`
        # and staged to the per-harness workspace as `.prompt_system.txt` +
        # `.prompt_task.txt`; the container only reads those files via /workspace.
        f"{PROJECT_ROOT}/evolution/__init__.py:/app/evolution/__init__.py:ro",
        f"{PROJECT_ROOT}/evolution/propose_in_container.py:/app/evolution/propose_in_container.py:ro",

        # Agent reference materials
        f"{PROJECT_ROOT}/evolution/memo_class.py:/app/evolution/memo_class.py:ro",
        f"{PROJECT_ROOT}/core/__init__.py:/app/core/__init__.py:ro",
        f"{PROJECT_ROOT}/core/memo_class.py:/app/core/memo_class.py:ro",
        f"{PROJECT_ROOT}/core/recorder.py:/app/core/recorder.py:ro",
        f"{PROJECT_ROOT}/core/llm.py:/app/core/llm.py:ro",
        f"{PROJECT_ROOT}/core/logger.py:/app/core/logger.py:ro",
        f"{PROJECT_ROOT}/core/model_policy.py:/app/core/model_policy.py:ro",
        f"{PROJECT_ROOT}/core/trace.py:/app/core/trace.py:ro",
        f"{PROJECT_ROOT}/core/qa.py:/app/core/qa.py:ro",
        f"{PROJECT_ROOT}/benchmarks:/app/benchmarks:ro",

        # A read-only CLI over this run's own history. The lineage is already
        # on disk, but spread across history.json, frontier.json and each
        # harness's meta.json — joining those by hand costs the agent tokens
        # and invites mistakes, since harness dirs are 12-hex content hashes
        # that carry no order and look alike.
        f"{PROJECT_ROOT}/scripts/search_history.py:/app/search_history.py:ro",

        # Workspace (cwd) + scratch HOME. `workspace_mode`
        # is "ro" for the analyst, which reads the whole run and writes only
        # the diagnostics/ bind layered on top of it (see analyze()).
        f"{paths.workspace}:/workspace:{workspace_mode}",
        f"{proposer_home}:/root:rw",
        f"{_HOST_CLAUDE_BIN}:/usr/local/bin/claude:ro",
        f"{_HOST_CLAUDE_SHARE}:/usr/local/share/claude:ro",
    ]
    if extra_binds:
        # e.g. the search-split data-isolation overlays.
        binds += list(extra_binds)

    cmd = [
        "singularity", "exec",
        "--containall",
        # Do NOT let the runtime auto-mount the invoking user's home.
        # --containall already replaces it with an empty one, but Apptainer
        # still has to resolve a path for it, and on a ZFS home it resolves to
        # "" — `failed to add  as session directory: path . is not an absolute
        # path`, before a single bind is applied. Nothing here wants that
        # mount: the agent's HOME is the
        # scratch .proposer_home bound at /root below.
        "--no-home",
    ]
    for b in binds:
        cmd += ["--bind", b]
    cmd += [
        # SINGULARITYENV_HOME=/root is implied by --containall; no need to set.
        "--env", "CLAUDE_CODE_ENABLE_SDK_FILE_CHECKPOINTING=true",
        # PYTHONPATH so any `python -c "from core.memo_class import ..."`
        # the agent runs for self-validation finds the bound /app/core package.
        "--env", "PYTHONPATH=/app",
        str(PROPOSER_BASE_SIF),
        "python", "/app/evolution/propose_in_container.py",
    ] + propose_args
    return cmd


async def _stream_subprocess(
    cmd: List[str],
    timeout_s: int,
    label: str,
    *,
    extra_env: Optional[Dict[str, str]] = None,
) -> int:
    """Run `cmd` and forward stdout line-by-line to the host log.

    Returns the subprocess exit code. Raises TimeoutError on wall-clock
    timeout (after escalating SIGTERM → SIGKILL on the singularity exec
    process group).

    `extra_env` is merged onto `os.environ` for the subprocess. Used to
    pass the agent's API key through to the container without exposing it
    in argv (which `ps` would see).
    """
    log.info(f"{label}: launching ({len(cmd)} argv tokens, image={PROPOSER_BASE_SIF.name})")
    env: Optional[Dict[str, str]] = None
    if extra_env:
        env = {**os.environ, **extra_env}
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
        # Put singularity in its own process group so we can kill it cleanly.
        preexec_fn=os.setsid,
        # Match the in-container readline limit. The container only emits
        # truncated `proposer·*: ...` lines so the host pump shouldn't ever
        # see > 64 KB in practice, but if the container crashes with a stack
        # trace (or a stray library prints a giant message) the default limit
        # would silently kill the outer pump too.
        limit=16 * 1024 * 1024,
    )

    async def _pump_stdout() -> None:
        assert proc.stdout is not None
        async for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip()
            if not line:
                continue
            # Lines from propose_in_container.py are pre-tagged
            # ("proposer·tool: ...", "proposer·text: ...", etc).
            # Map error tag → log.error, info / tool / text → log.info.
            # Anything untagged is unstructured output from the agent CLI's
            # own stderr (CC's bun-runtime exceptions, etc.),
            # which we MUST surface — silent rc=1 was traced to these being
            # swallowed by log.debug. Use WARNING so they're visible at the
            # default level without disrupting the agent's structured event
            # stream.
            if line.startswith("proposer·error"):
                log.error(line)
            elif line.startswith("proposer·info"):
                log.info(line)
            elif line.startswith("proposer·tool"):
                log.info(line)
            elif line.startswith("proposer·text"):
                log.info(line)
            else:
                log.warning(f"proposer·raw: {line}")

    pump_task = asyncio.create_task(_pump_stdout())
    try:
        await asyncio.wait_for(proc.wait(), timeout=timeout_s)
    except asyncio.TimeoutError:
        log.error(f"{label}: HUNG past {timeout_s}s — killing process group")
        try:
            import signal
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            await asyncio.wait_for(proc.wait(), timeout=10)
        except (ProcessLookupError, asyncio.TimeoutError):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                await proc.wait()
            except ProcessLookupError:
                pass
        raise TimeoutError(f"proposer subprocess timed out after {timeout_s}s")
    finally:
        # Drain any remaining stdout so log lines aren't lost.
        try:
            await asyncio.wait_for(pump_task, timeout=5)
        except asyncio.TimeoutError:
            pump_task.cancel()

    return proc.returncode if proc.returncode is not None else -1


# ---------------------------------------------------------------------------
# Host-side prompt rendering / staging
# ---------------------------------------------------------------------------

def _render_and_stage_prompts(
    *,
    new_id: str,
    mode: str,  # "propose" | "fix"
    sanity_enabled: bool,
    active_datasets: Optional[List[str]],
    metrics: Optional[List[str]] = None,
    error_trace: Optional[str] = None,
    analyst_enabled: bool = False,
    compose_enabled: bool = True,
) -> Dict[str, str]:
    """Render system + task prompts HOST-SIDE and write them to the harness dir as forensic-friendly text files.

    The container then reads the two files via the /workspace RW bind — it
    never imports `evolution.prompts.*`. Effect: each harness dir
    permanently captures the exact prompt that produced it, available for
    diff between runs without spelunking git.

    Returns the *container-side* absolute paths suitable to pass as argv to
    `propose_in_container.py`.
    """
    new_dir = paths.harnesses_dir / new_id
    new_dir.mkdir(parents=True, exist_ok=True)

    system_text = build_proposer_system(
        sanity_enabled=sanity_enabled,
        active_datasets=active_datasets or None,
        metrics=metrics or None,
        compose_enabled=compose_enabled,
    )
    # An ABSOLUTE container path, not `harnesses/<id>`. The proposer also reads
    # orchestrator.log, which names the HOST workspace, and a relative target
    # invites it to "translate" that: one run wrote a complete harness,
    # sentinel included, under /workspace/<run_name>/harnesses/<id>/ and was
    # scored as "never finished". The placeholder keeps its template name.
    new_dir_rel = f"/workspace/harnesses/{new_id}"
    if mode == "propose":
        task_text = proposer_task_prompt(
            new_dir_rel,
            active_datasets=active_datasets or None,
            analyst_enabled=analyst_enabled,
            )
    elif mode == "fix":
        if error_trace is None:
            raise ValueError("fix mode requires error_trace")
        task_text = proposer_fix_prompt(
            new_dir_rel, error_trace,
            compose_enabled=compose_enabled,
        )
    else:
        raise ValueError(f"unknown mode: {mode!r} (expected propose|fix)")

    system_host = new_dir / ".prompt_system.txt"
    task_host = new_dir / ".prompt_task.txt"
    system_host.write_text(system_text, encoding="utf-8")
    task_host.write_text(task_text, encoding="utf-8")

    return {
        "system_in_container": f"/workspace/harnesses/{new_id}/.prompt_system.txt",
        "task_in_container": f"/workspace/harnesses/{new_id}/.prompt_task.txt",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _stage_analyst_prompts(harness_id: str, *, datasets: List[str],
                           parent_ids: Optional[List[str]]) -> Dict[str, str]:
    """Render the analyst prompts host-side and stage them OUTSIDE the harness
    directory, which the analyst mounts read-only.

    `<workspace>/.analyst/<id>/` rather than the harness dir: `.prompt_*.txt`
    written into a harness IS part of its content hash (that is deliberate —
    a harness captures the prompt that produced it), and the analyst must not
    change the hash the frontier already points at."""
    from evolution.prompts.renderer import analyst_prompt

    stage_dir = paths.workspace / ".analyst" / harness_id
    stage_dir.mkdir(parents=True, exist_ok=True)
    system_text, task_text = analyst_prompt(
        harness_id, datasets=datasets,
        parent_ids=parent_ids,
    )
    (stage_dir / "system.txt").write_text(system_text, encoding="utf-8")
    (stage_dir / "task.txt").write_text(task_text, encoding="utf-8")
    return {
        "system_in_container": f"/workspace/.analyst/{harness_id}/system.txt",
        "task_in_container": f"/workspace/.analyst/{harness_id}/task.txt",
    }


async def analyze(
    harness_id: str,
    *,
    datasets: List[str],
    parent_ids: Optional[List[str]] = None,
    model: str = "claude-opus-4-8",
    timeout_s: int = 30 * 60,
    agent_opts: Optional[Dict[str, Any]] = None,
    extra_binds: Optional[List[str]] = None,
) -> Path:
    """Run the analyst over ONE evaluated harness; return the diagnosis path.

    The workspace is bound READ-ONLY with `diagnostics/` writable on top, so
    the analyst can read every prior candidate and write only its own two
    files. It cannot edit harness code, and it cannot rewrite another
    candidate's diagnosis by accident, because the prompt names the two
    paths it owns and nothing else in the tree accepts a write.

    Raises the same pre-flight / timeout / non-zero-exit errors as `propose`.
    A missing diagnosis is reported by the caller, not raised here: the
    diagnosis is context for the next step, never a gate on this one.
    """
    _check_environment()

    diagnostics_dir = paths.workspace / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    staged = _stage_analyst_prompts(
        harness_id, datasets=datasets, parent_ids=parent_ids,
    )
    proposer_home, extra_env = _prepare_auth()
    args = [
        "--new-id", harness_id,
        "--workspace", "/workspace",
        "--mode", "analyze",
        "--model", model,
        "--timeout-s", str(timeout_s),
        "--system-prompt-file", staged["system_in_container"],
        "--task-file", staged["task_in_container"],
        "--record-json", f"/workspace/diagnostics/{harness_id}.analyst.json",
        "--session-jsonl", f"/workspace/diagnostics/{harness_id}.analyst_session.jsonl",
        "--agent-opts", json.dumps(agent_opts or {}),
    ]
    cmd = _build_singularity_cmd(
        propose_args=args, proposer_home=proposer_home,
        workspace_mode="ro",
        extra_binds=(list(extra_binds or [])
                     + [f"{diagnostics_dir}:/workspace/diagnostics:rw"]),
    )
    rc = await _stream_subprocess(
        cmd, timeout_s=timeout_s + 60, label=f"analyst[{harness_id}]",
        extra_env=extra_env,
    )
    if rc != 0:
        raise RuntimeError(
            f"analyst subprocess exited with rc={rc} for {harness_id}; "
            f"see orchestrator log for the streamed agent output."
        )
    return diagnostics_dir / f"{harness_id}.md"


def _stray_sentinel_note(new_dir: Path) -> str:
    """Name any PROPOSAL_READY that landed outside `new_dir`.

    A proposer that mis-resolves its target directory writes a full harness
    somewhere else in the workspace; without this the failure reads as "forgot
    the sentinel" and the real cause takes a manual find to see. Walked
    without following symlinks: the workspace binds data dirs by link.
    """
    strays = []
    for root, dirs, files in os.walk(paths.workspace, followlinks=False):
        if "PROPOSAL_READY" in files and Path(root) != new_dir:
            strays.append(str(Path(root) / "PROPOSAL_READY"))
        if len(strays) >= 3:
            break
    if not strays:
        return ""
    return (
        " A PROPOSAL_READY was written elsewhere: " + ", ".join(strays)
        + " — the proposer resolved its target directory wrongly."
    )


async def propose(
    new_id: str,
    *,
    # Mirrors evolution/orchestrator.py::DEFAULT_CONFIG; keep in step.
    model: str = "claude-opus-4-8",
    # Mirrors evolution/orchestrator.py::DEFAULT_CONFIG; keep in step.
    timeout_s: int = 45 * 60,
    sanity_enabled: bool = True,
    active_datasets: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    agent_opts: Optional[Dict[str, Any]] = None,
    extra_binds: Optional[List[str]] = None,
    analyst_enabled: bool = False,
    compose_enabled: bool = True,
) -> Path:
    """Run a sandboxed proposer; return the new harness directory on success.

    The in-container script (`propose_in_container.py`) shells out to the
    Claude Code CLI, which authenticates with ANTHROPIC_API_KEY (see
    `_auth_extra_env`).

    Prompts are rendered HOST-SIDE and staged into the harness dir; the
    container does not import any evolution.prompts.* code.

    No parent_id passed — the agent explores `harnesses/` itself and decides
    which prior(s) to draw from. Choices are recorded in `meta.json::parent_ids`.

    Raises:
      ProposerLaunchError  pre-flight (SIF/binary/API key missing).
      TimeoutError         the subprocess exceeds timeout_s.
      RuntimeError         the proposer exits without PROPOSAL_READY, or
                           the subprocess returns a non-zero exit code.
    """
    _check_environment()

    new_dir = paths.harnesses_dir / new_id
    # mkdir on host so the bind sees it; container has it RW via /workspace.
    new_dir.mkdir(parents=True, exist_ok=True)

    staged = _render_and_stage_prompts(
        new_id=new_id,
        mode="propose",
        sanity_enabled=sanity_enabled,
        active_datasets=active_datasets,
        metrics=metrics,
        analyst_enabled=analyst_enabled,
        compose_enabled=compose_enabled,
    )

    proposer_home, extra_env = _prepare_auth()
    propose_args = [
        "--new-id", new_id,
        "--workspace", "/workspace",
        "--mode", "propose",
        "--model", model,
        "--timeout-s", str(timeout_s),
        "--system-prompt-file", staged["system_in_container"],
        "--task-file", staged["task_in_container"],
        "--agent-opts", json.dumps(agent_opts or {}),
    ]
    cmd = _build_singularity_cmd(
        propose_args=propose_args, proposer_home=proposer_home,
        extra_binds=list(extra_binds or []),
    )

    rc = await _stream_subprocess(
        cmd, timeout_s=timeout_s + 60, label=f"proposer[{new_id}]",
        extra_env=extra_env,
    )
    if rc != 0:
        raise RuntimeError(
            f"proposer subprocess exited with rc={rc} for {new_id}; "
            f"see orchestrator log for the streamed agent output."
            + _stray_sentinel_note(new_dir)
        )

    sentinel = new_dir / "PROPOSAL_READY"
    if not sentinel.exists():
        raise RuntimeError(
            f"proposer finished but PROPOSAL_READY missing at {new_dir}."
            + _stray_sentinel_note(new_dir)
        )
    return new_dir


async def propose_with_fix(
    new_id: str,
    error_trace: str,
    *,
    # Mirrors evolution/orchestrator.py::DEFAULT_CONFIG; keep in step.
    model: str = "claude-opus-4-8",
    # Mirrors evolution/orchestrator.py::DEFAULT_CONFIG; keep in step.
    timeout_s: int = 45 * 60,
    sanity_enabled: bool = True,
    active_datasets: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    agent_opts: Optional[Dict[str, Any]] = None,
    extra_binds: Optional[List[str]] = None,
    compose_enabled: bool = True,
) -> Path:
    """Ask the agent to Read + Edit the existing harness to fix a sanity-check
    failure.

    The error trace is folded HOST-SIDE into the rendered task prompt by
    `proposer_fix_prompt`; no separate file is written into the harness dir.
    The staged `.prompt_task.txt` for a fix call therefore CONTAINS the trace
    inline (truncated if >90 lines).
    """
    _check_environment()

    new_dir = paths.harnesses_dir / new_id
    if entry_file(new_dir) is None:
        raise RuntimeError(
            f"propose_with_fix called but {new_dir}/{ENTRY_FILE} does not exist"
        )

    staged = _render_and_stage_prompts(
        new_id=new_id,
        mode="fix",
        sanity_enabled=sanity_enabled,
        active_datasets=active_datasets,
        metrics=metrics,
        error_trace=error_trace,
        compose_enabled=compose_enabled,
    )

    proposer_home, extra_env = _prepare_auth()
    propose_args = [
        "--new-id", new_id,
        "--workspace", "/workspace",
        "--mode", "fix",
        "--model", model,
        "--timeout-s", str(timeout_s),
        "--system-prompt-file", staged["system_in_container"],
        "--task-file", staged["task_in_container"],
        "--agent-opts", json.dumps(agent_opts or {}),
    ]
    cmd = _build_singularity_cmd(
        propose_args=propose_args, proposer_home=proposer_home,
        extra_binds=list(extra_binds or []),
    )
    rc = await _stream_subprocess(
        cmd, timeout_s=timeout_s + 60, label=f"propose_fix[{new_id}]",
        extra_env=extra_env,
    )

    if rc != 0:
        raise RuntimeError(
            f"propose_with_fix subprocess exited with rc={rc} for {new_id}."
            + _stray_sentinel_note(new_dir)
        )
    return new_dir
