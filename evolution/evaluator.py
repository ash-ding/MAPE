"""Singularity-sandboxed evaluator.

Runs `evolution/launch.py` inside the harness's resolved image with a selective
bind list — NOT a whole-project /app:ro bind. Only the directories actually
needed by launch.py + harness imports are exposed:

  singularity exec \\
      --containall \\
      --bind <project_root>/core:/app/core:ro          # full core/ pkg
      --bind <project_root>/datasets:/app/benchmarks:ro      # full datasets/ pkg (incl. raw data)
      --bind <project_root>/evolution/__init__.py:/app/evolution/__init__.py:ro
      --bind <project_root>/evolution/launch.py:/app/evolution/launch.py:ro
      --bind <project_root>/evolution/memo_class.py:/app/evolution/memo_class.py:ro
      --bind <harness_dir>:/harness:ro \\
      --bind <out_dir>:/out:rw \\
      --env OPENAI_API_KEY=... ANTHROPIC_API_KEY=... \\
      --env PYTHONPATH=/app EVALS_LOG_DIR=/out ...
      <image.sif> \\
      python /app/evolution/launch.py --harness-dir /harness --out-dir /out ...

What's NOT bound (vs the previous /app:ro bind):
  - /app/workspace/     (cross-run isolation; this run's workspace is reachable
                         only via /out for the output dir)
  - /app/.env           (keys already injected via --env; load_dotenv() no-ops)
  - /app/.git, /app/baselines, /app/docs, /app/CLAUDE.md, /app/configs, etc.
  - /app/evolution/{orchestrator,proposer,evaluator,...}.py (host-side outer-loop)
  - /app/seeds          (evaluator never reads seeds)

Bound in addition to the source tree: the host's Hugging Face cache, at
/hf-cache. `--containall` gives the container its own HOME, so a local model
would otherwise be downloaded again on every single eval.
  - /app/venv           (container has its own python env)

Uses a dual-signal wait: race process.communicate() against score.json
appearance, kill on wall-clock timeout.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

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

from evolution.paths import PROJECT_ROOT
from core import model_policy

log = logging.getLogger("evolution.evaluator")

# Wall-clock caps per evaluation plan kind: ONE container runs the whole
# evaluation, and the host kills it past the cap.
SUBPROCESS_TIMEOUT = {
    "smoke": 2 * 3600,      # sanity gate / smoke_test — one sanity_check-sized pass
    "single": 12 * 3600,    # the evaluation pass sized by `size`
}

GRACE_AFTER_SCORE = 60
SCORE_POLL_INTERVAL = 5


#: The embedding model the run's policy allows and recommends. Set once per
#: process from cfg.seed.unified_models — the models the packaged seeds were
#: built for — so a harness and the seeds it is compared against embed with
#: the same model. The QA model and the judge arrive per call.
UNIFIED_EMBEDDING = "text-embedding-3-small"


#: cfg.llm.routes — locally served models, by name (see core/llm.py and
#: scripts/vllm/README.md). Empty: every model goes to its provider's API.
MODEL_ROUTES: Dict[str, Any] = {}

#: cfg.trace.max_value_chars — where a recorded prompt / response is clipped
#: (core/trace.py). Every evaluation records the full trace.
TRACE_MAX_VALUE: int = 20000


def configure_trace(*, max_value_chars: Any = None) -> None:
    """Record the run's clip length for every container launched from this
    process. Called by the orchestrator / heldout after the config resolves."""
    global TRACE_MAX_VALUE
    if max_value_chars:
        try:
            TRACE_MAX_VALUE = int(max_value_chars)
        except (TypeError, ValueError):
            pass


def configure_model_policy(*, unified_embedding: Optional[str],
                           routes: Optional[Dict[str, Any]] = None) -> None:
    """Record the run's unified embedding model and its model routes for
    every container launched from this process. Called by the orchestrator /
    heldout after the config is resolved; None keeps the defaults."""
    global UNIFIED_EMBEDDING, MODEL_ROUTES
    if unified_embedding:
        UNIFIED_EMBEDDING = str(unified_embedding)
    if routes is not None:
        MODEL_ROUTES = {str(k): dict(v) for k, v in routes.items()}
        # The host process prices and routes the same way its containers do.
        os.environ["MAPE_MODEL_ROUTES"] = json.dumps(MODEL_ROUTES) if MODEL_ROUTES else ""


async def run_evaluation(
    harness_dir: Path,
    image_path: Path,
    out_dir: Path,
    *,
    dataset: str = "locomo",
    split: str = "search",   # search | test — benchmark split
    plan: Dict[str, Any],
    model: str = "gpt-5-mini",
    judge_model: str = "gpt-5-mini",
    max_sample_concurrent: int = 3,
    gpu: bool = False,
    data_isolation_binds: Optional[list] = None,
    compose_enabled: bool = True,
) -> Path:
    """Evaluate a harness inside Singularity.

    `compose_enabled=False` (cfg.contract.compose) tells launch.py to refuse a
    harness that defines `compose_memory_context`; the refusal is a load
    failure, so under the sanity gate it becomes a fix round.

    claude-* models inside the container (via core.llm) reach the Anthropic
    API with the ANTHROPIC_API_KEY injected below.

    `plan` is the whole-evaluation plan launch.py forwards to evaluate_memo:
    {"smoke": bool, "size": dict|None, "sanity_check": dict|None}.
    ONE container runs the whole pass.

    Returns out_dir on success (or on score.json-observed completion); raises
    RuntimeError if the subprocess hangs past the wall-clock cap. The root
    score.json is written at the END of the pass, so its
    appearance still unambiguously means "this run finished".
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale score.json so its appearance unambiguously means "this run
    # finished". Without this, a leftover from a prior run would fire the
    # dual-signal wait immediately and cause a premature kill.
    for fname in ("score.json", "metrics.json"):
        stale = out_dir / fname
        if stale.exists():
            try:
                stale.unlink()
            except OSError as exc:
                log.warning(f"evaluator: could not remove stale {fname}: {exc}")

    # Where the shared model cache lives on the host: the operator's HF_HOME
    # when they set one, else the standard
    # location. Created if absent so the bind never fails on a fresh machine.
    hf_cache = Path(os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface"))
    hf_cache.mkdir(parents=True, exist_ok=True)

    env_vals = dotenv_values(str(PROJECT_ROOT / ".env"))
    openai_key = env_vals.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    anthropic_key = env_vals.get("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))

    cmd = [
        "singularity", "exec",
        "--containall",
        # Do NOT let the runtime auto-mount the invoking user's home.
        # --containall already replaces it with an empty one, but Apptainer
        # still has to resolve a path for it, and on a ZFS home it resolves to
        # "" — `failed to add  as session directory: path . is not an absolute
        # path`, before a single bind is applied. Nothing here wants that
        # mount: HF_HOME is pointed at the
        # /hf-cache bind, and nothing else writes to HOME.
        "--no-home",
    ]
    if gpu:
        # Bind-mounts host NVIDIA driver libs (libcuda, libnvidia-ml, ...) so
        # the container's torch+cu126 wheel can talk to the driver. Required
        # for torch.cuda.is_available() to return True.
        cmd.append("--nv")
    cmd += [
        # Selective binds — only what launch.py + harness actually need.
        # See module docstring for what's intentionally NOT bound.
        "--bind", f"{PROJECT_ROOT}/core:/app/core:ro",
        "--bind", f"{PROJECT_ROOT}/benchmarks:/app/benchmarks:ro",
        "--bind", f"{PROJECT_ROOT}/evolution/__init__.py:/app/evolution/__init__.py:ro",
        "--bind", f"{PROJECT_ROOT}/evolution/launch.py:/app/evolution/launch.py:ro",
        # Harnesses inherit evolution.memo_class.MemoClass (subclass of the
        # common ABC) — the module must be importable inside the container.
        "--bind", f"{PROJECT_ROOT}/evolution/memo_class.py:/app/evolution/memo_class.py:ro",
        "--bind", f"{harness_dir}:/harness:ro",
        "--bind", f"{out_dir}:/out:rw",
        # One Hugging Face cache, shared by every eval. `--containall` gives
        # the container its own HOME, so without this each run re-downloads
        # whatever local model it uses — zep alone pulls bge-m3 plus its
        # cross-encoder, ~4.5 GB, every single time. Writable on purpose: a
        # read-only cache still makes huggingface_hub try to take its locks.
        "--bind", f"{hf_cache}:/hf-cache:rw",
    ]
    # Search-mode data isolation: overlay binds that shadow the test split
    # inside /app/benchmarks (see evolution/data_isolation.py). Passed for
    # every split=search run.
    for b in (data_isolation_binds or []):
        cmd += ["--bind", b]

    # Model policy for everything that runs in this container: the harness's
    # own calls, the QA agent and the judge. Enforced at the SDK boundary by
    # core.openai_usage and at construction by core.llm — a harness that
    # names any other model fails before the request is sent.
    model_policy.warn_if_unusual_backbone(model, where="cfg.model")
    for key, value in model_policy.env_for_container(
            llm=model, judge=judge_model, embedding=UNIFIED_EMBEDDING).items():
        cmd += ["--env", f"{key}={value}"]
    # Locally served models (scripts/vllm/serve.sh): where a model name lives.
    # Containers share the host network, so 127.0.0.1 routes reach the
    # servers; the judge and embeddings keep going to their API.
    if MODEL_ROUTES:
        cmd += ["--env", f"MAPE_MODEL_ROUTES={json.dumps(MODEL_ROUTES, separators=(',', ':'))}"]
    cmd += ["--env", f"MAPE_TRACE_MAX_VALUE={TRACE_MAX_VALUE}"]

    cmd += [
        "--env", f"OPENAI_API_KEY={openai_key}",
        "--env", f"ANTHROPIC_API_KEY={anthropic_key}",
        "--env", "PYTHONPATH=/app",
        "--env", "PYTHONUNBUFFERED=1",
        # core.logger writes rotating logs to EVALS_LOG_DIR. Send them into
        # the per-run out dir so they're writable and colocated with artifacts.
        "--env", "EVALS_LOG_DIR=/out",
        "--env", "MAPE_LOG_FILE=subprocess.log",
        # Point every HF loader at the shared cache bound above.
        "--env", "HF_HOME=/hf-cache",
        "--env", "HUGGINGFACE_HUB_CACHE=/hf-cache/hub",
        str(image_path),
        "python", "/app/evolution/launch.py",
        "--harness-dir", "/harness",
        "--out-dir", "/out",
        "--dataset", dataset,
        "--split", split,
        "--plan-json", json.dumps(plan),
        "--model", model,
        "--judge-model", judge_model,
        "--max-sample-concurrent", str(max_sample_concurrent),
    ]
    if not compose_enabled:
        cmd.append("--no-compose-hook")

    plan_kind = "smoke" if plan.get("smoke") else "single"
    log.info(
        f"evaluator: launching {harness_dir.name} [{dataset}/{plan_kind}] with {image_path.name} "
        f"(plan={json.dumps(plan)}, gpu={'on' if gpu else 'off'})"
    )

    process = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(PROJECT_ROOT),
        stderr=asyncio.subprocess.PIPE,
    )

    timeout_s = SUBPROCESS_TIMEOUT.get(plan_kind, 14 * 3600)
    score_path = out_dir / "score.json"

    async def _poll_for_score() -> bool:
        while not score_path.exists():
            await asyncio.sleep(SCORE_POLL_INTERVAL)
        return True

    communicate_task = asyncio.create_task(process.communicate())
    score_task = asyncio.create_task(_poll_for_score())

    done, pending = await asyncio.wait(
        {communicate_task, score_task},
        timeout=timeout_s,
        return_when=asyncio.FIRST_COMPLETED,
    )

    stderr_bytes = b""
    forced_kill = False

    if not done:
        for t in pending:
            t.cancel()
        log.error(
            f"evaluator: HUNG past {timeout_s}s for {harness_dir.name} — killing"
        )
        try:
            process.kill()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except asyncio.TimeoutError:
            log.error(f"evaluator: failed to reap {harness_dir.name} within 5s")
        raise RuntimeError(
            f"Subprocess for {harness_dir.name} timed out past {timeout_s}s"
        )

    if communicate_task in done:
        _, stderr_bytes = communicate_task.result()
        score_task.cancel()
        try:
            await score_task
        except asyncio.CancelledError:
            pass
    else:
        log.warning(
            f"evaluator: score.json landed for {harness_dir.name} but "
            f"communicate() still pending; grace {GRACE_AFTER_SCORE}s"
        )
        try:
            _, stderr_bytes = await asyncio.wait_for(
                communicate_task, timeout=GRACE_AFTER_SCORE
            )
        except asyncio.TimeoutError:
            forced_kill = True
            log.warning(
                f"evaluator: asyncio exit notification stuck for "
                f"{harness_dir.name}; force-killing"
            )
            try:
                process.kill()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                log.error(f"evaluator: failed to reap {harness_dir.name} after SIGKILL")
            stderr_bytes = b""

    if not forced_kill and process.returncode not in (0, None):
        stderr_text = stderr_bytes.decode(errors="replace").strip() if stderr_bytes else ""
        log.error(
            f"evaluator: subprocess exited rc={process.returncode} for {harness_dir.name} [{dataset}]"
            + (f"\nstderr:\n{stderr_text[-2000:]}" if stderr_text else "")
        )

    log.info(f"evaluator: done {harness_dir.name} [{dataset}] → {out_dir}")
    return out_dir
