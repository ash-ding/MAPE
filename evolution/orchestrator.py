"""Evolution outer loop.

Per-harness flow (search loop):

    propose → ensure_image
           → sanity_check_harness (small real-data run per dataset, if
                                   sanity.enabled)
               ├─ pass    → evaluate_harness (full run per dataset) → frontier
               └─ fail    → propose_with_fix(error_trace) → retry (max K times)
                             ├─ eventually pass → evaluate_harness
                             └─ all retries exhausted → skip eval,
                                 enter frontier with sanity_failed + score=0

For `smoke_test: true`: skip sanity entirely and run the evaluator once per
benchmark at `sanity_check` sizes (the run IS the sanity-size
verification). steps/k_per_step are forced to 1 unless explicitly given on
the CLI — a smoke run is a single propose→eval pipeline check by default.

Two ways to launch:

    # (a) config-first (recommended for long runs / reproducibility)
    python -m evolution.orchestrator --config configs/evolution/example.yaml

    # (b) CLI-first (quick iteration; uniform params across all datasets)
    python -m evolution.orchestrator --steps 3 --datasets locomo \
        --smoke-test

CLI flags always override the config file when both are given. See
`configs/evolution/example.yaml` (exhaustive, runnable) for the full config schema.

Outputs (per-run, under workspace/<run_id>/):
    harnesses/<hash>/<dataset>/         full-eval score + traces
    harnesses/<hash>/<dataset>/sanity/  small sanity-check score + traces
    harnesses/<hash>/memo.py            the harness's interface
    harnesses/<hash>/meta.json          parent_ids, description, content_hash, created_at
    frontier.json                       population snapshot (with sanity_status)
    history.json                        which harnesses each step produced, in order
    evolution/logs/orchestrator.log         rotating log (shared across runs)

A harness dir is named by the hash of its own code, so the same code is the
same directory: a proposal that reproduces an earlier harness byte-for-byte is
recognised as a duplicate and reuses that harness's evaluation instead of
paying for it twice. Names therefore carry no ordering — `history.json` does.

Per-harness eval over multiple datasets is serial (one Singularity exec per
dataset). `Frontier.entries[i].objectives` stores per-dataset scores
("accuracy_<dataset>") and no cross-dataset mean. Parent selection is
delegated to the proposer agent — the search loop has NO selection rule.
Each orchestrator invocation gets its own per-run dir (`--run-name` or an
auto timestamp); a harness dir is renamed `pending_<n>` → `<hash>` once the
sanity loop settles, before full eval.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import datetime as _dt
import json
import logging
import os
import re
import shutil
import sys
import time
import uuid
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from evolution.env_builder import EnvBuildError, ensure_image
from evolution.evaluator import run_evaluation
from evolution import seed_sources
from evolution.codetree import _compute_content_hash, _copy_harness_code
from evolution.paths import LOGS_DIR, ensure_dirs, entry_file, paths
from evolution.prompts import PromptPartsError, load_prompt_parts
from evolution.proposer import propose, propose_with_fix
from evolution.selection import Entry, Frontier

from core.config import (
    deep_merge as _deep_merge,
    strict_on, missing_schema_paths, raise_completeness,
    REQUIRED, Cond,
)
from core.evaluate import (
    resolve_size_spec, resolve_dataset_sizes, missing_sizing_config,
)

log = logging.getLogger("evolution.orchestrator")


class ProposerInfraError(RuntimeError):
    """Raised when the proposer subprocess itself fails (rc!=0, timeout, auth).

    Distinct from harness-level failures (sanity_failed, env_build, eval crash)
    which the search loop continues past — those are CC's design mistakes that
    the search is supposed to learn from. ProposerInfraError means the proposer
    subprocess never completed (no memo.py written), so there's no signal
    to learn from and continuing would just generate a placeholder entry that
    pollutes the frontier and confuses resume logic.

    Triggers (a) cleanup of the empty harness dir, (b) abort of search_loop,
    (c) main()'s sys.exit(2) so wrapper scripts can detect.
    """


# ---------------------------------------------------------------------------
# Config resolution: defaults ← YAML file ← CLI overrides
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    "steps": 1,
    # Smoke test: run the WHOLE pipeline once per benchmark at
    # sanity_check sizes (search split; no sanity gate — the run IS the
    # sanity-size check). Costs cents;
    # use to verify the pipeline end-to-end. CLI: --smoke-test.
    # Default False = the normal search loop. Held-out evaluation is NOT a
    # mode of this orchestrator — use `python -m evolution.heldout`.
    "smoke_test": False,
    # The QA agent that answers from a harness's retrieved memory. 4.1-mini
    # because gpt-5-mini reasons before answering: measured at 15-23 s per
    # call against 2-3 s, for ~5x the completion tokens (see
    # scripts/package_baseline.py). The judge stays on gpt-5-mini — it scores
    # one short pair at a time, where that latency does not compound.
    "model": "gpt-4.1-mini",
    "judge_model": "gpt-5-mini",
    "max_sample_concurrent": 3,
    # What this search optimizes. Every metric is always MEASURED and written
    # to each harness's score.json — this only decides which axes the frontier
    # (the scoreboard the proposer reads) carries, and what the proposer is
    # told to optimize:
    #   ["accuracy"]                — scores only; no cost axis on the
    #                                 frontier, so a run can be compared
    #                                 against a cost-aware one without the
    #                                 proposer having seen cost at all.
    #   ["accuracy", "efficiency"]  — also `cost_usd_per_1k_queries_<ds>`
    #                                 (dollars per 1K queries, every model
    #                                 priced, embeddings included); the
    #                                 proposer is told to hold accuracy while
    #                                 spending less.
    # There is no weighted sum and no budget: the proposer weighs the axes
    # itself, the same way it already weighs accuracy against robustness.
    # CLI: --metric accuracy,efficiency
    "metrics": ["accuracy"],
    # Locally served models (scripts/vllm/serve.sh writes this block):
    #   routes: {qwen3-8b: {base_urls: [...], api_key: EMPTY,
    #                       chat_template_kwargs: {...}, price_usd_per_1m: [in, out]}}
    # Routing is by model name; the judge and embeddings stay on their API.
    # claude-* models are reached through the Anthropic API (ANTHROPIC_API_KEY).
    "llm": {
        "routes": {},
    },
    "proposer": {
        # Claude Code, run in a sandboxed container. A dateless Claude id from
        # the 4.6 generation on IS a pinned snapshot, not a floating alias.
        # THE default; evolution/proposer.py (signature defaults) and
        # evolution/propose_in_container.py (argparse fallback) mirror it.
        "model": "claude-opus-4-8",
        # low / medium / high / xhigh / max  (--effort). Anthropic's guidance
        # for Opus 4.7/4.8 is xhigh for coding and agentic use; at medium the
        # model scopes work to what was asked instead of exploring, which is
        # the opposite of what a proposer is for.
        "effort": "xhigh",
        # Tools the CLI is told not to consider. Defense-in-depth only — the
        # filesystem boundary is the Singularity bind list.
        "disallowed_tools": ["mcp__*"],
        # The hard limit — wall clock, enforced by SIGTERM/SIGKILL on the
        # singularity exec process group. xhigh effort is meant for runs over
        # 30 minutes; a proposer killed mid-write leaves a half-written harness.
        "timeout_s": 45 * 60,
        # Authentication: ANTHROPIC_API_KEY only (host env or project .env).
    },
    "propose": {
        # k harnesses proposed per outer-loop step. Default 2 follows the
        # Meta-Harness paper. Each is generated by an independent CC-SDK call.
        "k_per_step": 2,
    },
    "sanity": {
        "enabled": True,      # run the pre-eval sanity gate (smoke_test runs always skip)
        "max_retries": 2,     # on sanity failure, call propose_with_fix up to this many times
    },
    "seed": {
        # Seeds are step 0: harnesses the run starts from, so the proposer has
        # known designs (and known scores) to build on. Default is OFF — the
        # agent writes the first harness from scratch with parent_ids=[].
        #
        # Each source names a baseline this repo registers (see
        # baselines.eval_harness.MEMOS: mem0, zep, no_memory, …). It is
        # PACKAGED on demand — evolution shells out to
        # scripts/package_baseline.py in that baseline's own venv and caches the
        # result under packaged/. Anything else is refused before the run
        # starts. Several are allowed: step 0 then holds one directory per
        # seed.
        #
        # `arm` and `unified_models` are how a registered name is packaged:
        # "faithful" runs the method on its published models, "unified" points
        # every baseline at one model so they are comparable.
        # CLI: --seed <a,b> / --no-seed
        "enabled": False,
        "sources": ["no_memory"],
        "arm": "unified",
        "unified_models": {"llm": "gpt-4.1-mini",
                           "embedding": "text-embedding-3-small"},
    },
    # After a candidate is evaluated, hand its evidence to a dedicated agent
    # and let it write the diagnosis: what changed, what the numbers did, and
    # WHERE each failure was lost (build / retrieve / answer / judge / data).
    # It replaces the retro the next proposer writes today — the proposer then
    # spends its turns reading evidence and building, not reconstructing what
    # just happened. Off by default: it adds a serial agent session per step.
    # The analyst runs Claude Code like the proposer; its model and effort
    # are separate, so a cheap analyst can be paired with an expensive
    # proposer.
    "analyst": {
        "enabled": False,
        "model": "claude-opus-4-8",
        "effort": "xhigh",
        "timeout_s": 30 * 60,
        "include_seeds": True,   # the ICL seed's failure modes drove step 1 of every run
    },
    # Every evaluation records the full trace (core/trace.py): per question
    # the QA trajectory, one record per LLM / embedding call with prompt and
    # response, and every call into the harness's own functions. It costs disk
    # and nothing else (the sink only observes).
    "trace": {
        "max_value_chars": 20000,   # per prompt / response, then marked-truncated
    },
    "gpu": {
        # When enabled, evaluator runs `singularity exec --nv ...` so the
        # container can use the host's NVIDIA driver (libcuda.so) and any
        # GPUs visible at /dev/nvidia*. Default OFF for portability — turning
        # on couples runs to a host with a compatible NVIDIA driver. Opt in
        # via `gpu: {enabled: true}` in YAML or `--gpu` CLI flag.
        # Proposer container does NOT honor this (CC just makes API calls).
        "enabled": False,
    },
    # When True, search_loop's resume path scans `harnesses/` for dirs that
    # exist on disk but aren't in frontier (e.g. orchestrator killed mid-eval),
    # classifies them by file presence, and runs only the missing pipeline
    # stages before adding them to frontier. Disable via `--no-adopt-orphans`
    # if you suspect the orphan dirs are corrupt.
    "adopt_orphans": True,
    # Which hooks of the memo contract an evolved harness may implement.
    # `compose: false` closes the third hook: the proposer is told the
    # contract has two hooks (build + retrieve), and a candidate that defines
    # `compose_memory_context` anyway is refused when the container loads it
    # (evolution/launch.py), which fails its sanity check. Required in strict
    # mode like every other knob.
    "contract": {
        "compose": True,
    },
    # Mandatory-complete config enforcement: when a --config file
    # is given, every key in REQUIRED_SCHEMA (below) must be explicitly
    # listed in the YAML (a `null` value counts as provided; an ABSENT key
    # does not) — silent fallback to a DEFAULT_CONFIG value is disallowed.
    # Default on; escape hatch: `strict_config: false` / --no-strict-config.
    # In-process configs (no --config file) are never subject to this check.
    "strict_config": True,
    # `run_name` defaults to a timestamp at resolve time (so the default isn't
    # frozen at import time). `datasets` intentionally omitted from defaults.
}


# ---------------------------------------------------------------------------
# Strict-config completeness schema — see core/config.py for
# the REQUIRED/Cond primitives + the walk/raise machinery. `datasets` is
# validated separately (dynamic per-dataset keys) by _strict_validate's
# custom loop, not by this static schema.
# Meta-exempt (NOT required): run_name,
# strict_config (the escape hatch itself).
# ---------------------------------------------------------------------------

REQUIRED_SCHEMA = {
    "steps": REQUIRED, "smoke_test": REQUIRED, "model": REQUIRED,
    "judge_model": REQUIRED, "metrics": REQUIRED,
    "max_sample_concurrent": REQUIRED,
    "adopt_orphans": REQUIRED,
    "llm": {"routes": REQUIRED},
    "proposer": {"model": REQUIRED, "effort": REQUIRED,
                 "disallowed_tools": REQUIRED, "timeout_s": REQUIRED},
    "propose": {"k_per_step": REQUIRED},
    "sanity": {"enabled": REQUIRED, "max_retries": REQUIRED},
    "analyst": {"enabled": REQUIRED,
                "model": Cond(lambda c: bool(c.get("analyst", {}).get("enabled")), REQUIRED),
                "effort": Cond(lambda c: bool(c.get("analyst", {}).get("enabled")), REQUIRED),
                "timeout_s": Cond(lambda c: bool(c.get("analyst", {}).get("enabled")), REQUIRED),
                "include_seeds": Cond(lambda c: bool(c.get("analyst", {}).get("enabled")), REQUIRED)},
    "seed": {"enabled": REQUIRED,
             "sources": Cond(lambda c: bool(c.get("seed", {}).get("enabled")), REQUIRED),
             # How a registered baseline name is packaged into a seed. Only
             # meaningful when seeding, so only required then.
             "arm": Cond(lambda c: bool(c.get("seed", {}).get("enabled")), REQUIRED),
             "unified_models": Cond(lambda c: bool(c.get("seed", {}).get("enabled")), REQUIRED)},
    "gpu": {"enabled": REQUIRED},
    "trace": {"max_value_chars": REQUIRED},
    "contract": {"compose": REQUIRED},
    # `datasets` validated by _strict_validate's custom loop (dynamic keys).
    # Meta-exempt (NOT required): run_name, strict_config.
}

# evolution.heldout drives the SAME _resolve_config but never reads the
# search-loop-only fields (steps/smoke_test/
# adopt_orphans/agent/proposer.*/propose.*/sanity.*/seed.*/prompts.*) —
# REQUIRED_SCHEMA wrongly demanded them for heldout configs. Heldout's
# real config surface (verified via evolution/heldout.py's cfg[...] reads):
# model, judge_model, max_sample_concurrent, gpu.enabled, llm,
# datasets.
HELDOUT_REQUIRED_SCHEMA = {
    "model": REQUIRED,
    "judge_model": REQUIRED,
    "max_sample_concurrent": REQUIRED,
    "gpu": {"enabled": REQUIRED},
    "trace": {"max_value_chars": REQUIRED},
    "llm": {"routes": REQUIRED},
    # `datasets` validated by _strict_validate's dataset loop (`size`
    # leaves). Meta-exempt: run_name, strict_config.
}


def _strict_validate(provided_tree: Dict[str, Any], resolved_cfg: Dict[str, Any],
                           schema: Dict[str, Any] = REQUIRED_SCHEMA) -> None:
    """Strict completeness for the evolution loop's nested config. Validates `schema` (the
    search-loop schema by default; evolution.heldout passes HELDOUT_REQUIRED_SCHEMA,
    its smaller real config surface) AND the dynamic datasets block, collecting
    ALL missing paths into ONE raise."""
    missing = missing_schema_paths(provided_tree, schema, resolved_cfg)
    ds = provided_tree.get("datasets")
    if not isinstance(ds, dict) or not ds:
        missing.append("datasets (non-empty)")
    else:
        for name, params in ds.items():
            missing += missing_sizing_config(name, params or {}, path_prefix=f"datasets.{name}")
    raise_completeness("evolution config", missing)


def _provided_tree(file_cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Raw YAML ∪ CLI-provided nested paths (deep-copied), for the strict check.

    Only the CLI flags explicitly overlaid below count as "provided" for
    strict-config purposes — steps/model/judge_model/max_sample_concurrent/
    metrics/proposer.timeout_s/proposer.model.
    Other evolution CLI flags that also mutate `cfg` in `_resolve_config` (e.g.
    --datasets, --smoke-test, --gpu,
    --k-per-step, --sanity-max-retries,
    --no-adopt-orphans, --no-seed/--seed-source,
    --proposer-disallowed-tools) are NOT overlaid here. Under
    strict mode (`--config` + strict_config not disabled), using one of those
    flags WITHOUT also listing the corresponding key in the YAML still trips
    the completeness gate (the flag's effect on `cfg` is real, but the
    provided-tree used for the presence check won't reflect it) — set the key
    explicitly in the YAML too, or pass `--no-strict-config`. This is a
    known/accepted gap, not a bug: extending the overlay to every CLI flag is
    future work, tracked separately.
    """
    tree = copy.deepcopy(file_cfg)

    def setpath(path, val):
        if val is None:
            return
        d = tree
        for k in path[:-1]:
            d = d.setdefault(k, {})
        d[path[-1]] = val

    setpath(["steps"], getattr(args, "steps", None))
    setpath(["model"], getattr(args, "model", None))
    setpath(["judge_model"], getattr(args, "judge_model", None))
    setpath(["max_sample_concurrent"], getattr(args, "max_sample_concurrent", None))
    setpath(["metrics"], getattr(args, "metrics", None))
    setpath(["proposer", "timeout_s"], getattr(args, "proposer_timeout_s", None))
    setpath(["proposer", "model"], getattr(args, "proposer_model", None))
    return tree


def _resolve_config(args: argparse.Namespace, *,
                    required_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build the effective run config from defaults, optional YAML, and CLI.

    Precedence (lowest → highest):
      1. DEFAULT_CONFIG (this module)
      2. YAML at --config path (if given)
      3. Explicit CLI args (only when not None)

    `required_schema`: the strict-config schema to validate against (search
    loop callers omit it → REQUIRED_SCHEMA; evolution.heldout passes
    HELDOUT_REQUIRED_SCHEMA, its smaller real config surface).
    """
    cfg: Dict[str, Any] = copy.deepcopy(DEFAULT_CONFIG)
    # Raw YAML dict (empty when no --config given) — kept around so the
    # strict check can tell "explicitly set in the file" apart from "left at
    # its DEFAULT_CONFIG value".
    file_cfg: Dict[str, Any] = {}

    if args.config is not None:
        path = Path(args.config)
        if not path.exists():
            raise FileNotFoundError(f"config file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            file_cfg = yaml.safe_load(f) or {}
        if not isinstance(file_cfg, dict):
            raise ValueError(f"config file must be a mapping at top level: {path}")
        _deep_merge(cfg, file_cfg)

    # Top-level CLI overrides (each applied only if explicitly provided)
    for key in (
        "steps", "model", "judge_model",
        "max_sample_concurrent", "strict_config",
    ):
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val

    if getattr(args, "proposer_timeout_s", None) is not None:
        cfg["proposer"]["timeout_s"] = args.proposer_timeout_s
    if args.proposer_model is not None:
        cfg["proposer"]["model"] = args.proposer_model
    # Comma-separated. Empty string = allow everything.
    if args.proposer_disallowed_tools is not None:
        cfg["proposer"]["disallowed_tools"] = [
            t.strip() for t in args.proposer_disallowed_tools.split(",") if t.strip()
        ]
    if args.k_per_step is not None:
        cfg["propose"]["k_per_step"] = args.k_per_step
    if getattr(args, "smoke_test", False):
        cfg["smoke_test"] = True
    # Smoke test = a single-candidate pipeline check: force
    # steps=1 / k_per_step=1 unless the CLI EXPLICITLY asks for more. YAML
    # values are deliberately overridden — a search config's steps must not
    # silently multiply a smoke run into steps*k proposer calls.
    if cfg["smoke_test"]:
        forced = []
        if getattr(args, "steps", None) is None and cfg["steps"] != 1:
            cfg["steps"] = 1
            forced.append("steps=1")
        if args.k_per_step is None and cfg["propose"]["k_per_step"] != 1:
            cfg["propose"]["k_per_step"] = 1
            forced.append("k_per_step=1")
        if forced:
            log.info(
                "smoke test: forcing %s (pass --steps / --k-per-step to override)",
                ", ".join(forced),
            )
    if args.no_sanity:
        cfg["sanity"]["enabled"] = False
    if args.sanity_max_retries is not None:
        cfg["sanity"]["max_retries"] = args.sanity_max_retries
    if getattr(args, "seed_sources", None):
        cfg["seed"]["sources"] = [s.strip() for s in args.seed_sources.split(",") if s.strip()]
        cfg["seed"]["enabled"] = True
    # --no-seed is the last word: it turns seeding off even if sources were
    # given, so a config's seeds can be skipped without editing the file.
    if args.no_seed:
        cfg["seed"]["enabled"] = False
    if args.gpu:
        cfg["gpu"]["enabled"] = True
    if args.no_adopt_orphans:
        cfg["adopt_orphans"] = False

    # run_name: CLI > YAML > timestamp default. Resolved here (not in
    # DEFAULT_CONFIG) so `from evolution.orchestrator import DEFAULT_CONFIG`
    # doesn't freeze the timestamp at import time.
    if args.run_name is not None:
        cfg["run_name"] = args.run_name
    cfg.setdefault("run_name", _dt.datetime.now().strftime("%Y%m%d_%H%M%S"))

    # Datasets
    cli_datasets = getattr(args, "datasets", None)
    if cli_datasets is not None:
        # CLI --datasets replaces the YAML datasets block entirely; each
        # benchmark is evaluated on its whole search split, with the family's
        # default sanity_check size.
        ds_list = [d.strip() for d in cli_datasets.split(",") if d.strip()]
        cfg["datasets"] = {ds: {"size": {}} for ds in ds_list}

    # metrics: CLI ("a,b") wins over the YAML list; both are normalised (and
    # a typo'd metric raises here, before anything runs).
    if getattr(args, "metrics", None):
        cfg["metrics"] = args.metrics
    cfg["metrics"] = _resolve_metrics(cfg.get("metrics"))


    # Mandatory-complete config gate: only when --config was given AND
    # strict_config (default True) is on. The schema's Cond predicates read
    # the resolved `cfg`; `file_cfg`
    # (raw YAML) ∪ CLI-provided nested paths is the presence source. Callers
    # with a smaller real config surface (evolution.heldout) pass their own
    # `required_schema`; the search loop's own call sites (main() etc.) never
    # pass one, so they get REQUIRED_SCHEMA via the default.
    if strict_on(args.config, cfg):
        _strict_validate(_provided_tree(file_cfg, args), cfg,
                               required_schema or REQUIRED_SCHEMA)

    # Validation + size-schema resolution
    if not cfg.get("datasets"):
        raise ValueError(
            "No datasets specified. Provide `datasets:` in --config YAML, "
            "or pass --datasets on the CLI."
        )
    for ds, params in cfg["datasets"].items():
        if not isinstance(params, dict):
            raise ValueError(
                f"datasets.{ds} must be a mapping, got {type(params).__name__}"
            )
        resolve_dataset_sizes(ds, params)
        # Per-dataset judge override; falls back to the global judge_model.
        params.setdefault("judge_model", cfg["judge_model"])

    return cfg


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

_LOG_FORMATTER = logging.Formatter(
    "%(asctime)s %(name)s [%(levelname)s] %(message)s"
)


def _setup_logging(verbose: bool) -> None:
    """Phase-1 logging: console + global rotating tape (run-agnostic).

    The per-run file handler is attached separately by `_attach_run_log` once
    `paths.set_run_id(...)` resolves the workspace path.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s %(name)s [%(levelname)s] %(message)s", "%H:%M:%S"
    ))
    root.addHandler(ch)
    # Global tape — every run appends here. Rotating to cap disk use.
    fh = RotatingFileHandler(
        LOGS_DIR / "orchestrator.log", maxBytes=5 * 1024 * 1024, backupCount=3,
        encoding="utf-8",
    )
    fh.setFormatter(_LOG_FORMATTER)
    root.addHandler(fh)


def _attach_run_log() -> Path:
    """Phase-2 logging: per-run FileHandler at workspace/<run_id>/orchestrator.log.

    Must be called after `paths.set_run_id(...)`. Returns the log file path.
    """
    paths.workspace.mkdir(parents=True, exist_ok=True)
    log_path = paths.workspace / "orchestrator.log"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(_LOG_FORMATTER)
    logging.getLogger().addHandler(fh)
    return log_path


def _write_resolved_config(cfg: Dict[str, Any]) -> Path:
    """Snapshot the effective run config into workspace/<run_id>/config.yaml.

    Captures the dict AFTER defaults + YAML + CLI have been merged, so the
    file alone is enough to reproduce the run (regardless of whether the
    user passed a config file, plain CLI args, or both).
    """
    paths.workspace.mkdir(parents=True, exist_ok=True)
    cfg_path = paths.workspace / "config.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return cfg_path


# ---------------------------------------------------------------------------
# Resume guard — a resumed run must not quietly change what its numbers mean
# ---------------------------------------------------------------------------

def _eval_inputs(cfg: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    """What one benchmark's evaluation numbers depend on, in readable form."""
    from core.pricing import COST_DEFINITION
    ds_cfg = dict((cfg.get("datasets") or {}).get(dataset) or {})
    return {
        "dataset": dataset,
        "split": cfg.get("split", "search"),
        "sizing": ds_cfg.get("size"),
        "model": cfg.get("model"),
        "judge_model": ds_cfg.get("judge_model") or cfg.get("judge_model"),
        # Local models (zep's reranker, LightMem's compressor) are not
        # guaranteed bit-identical across devices.
        "gpu": bool((cfg.get("gpu") or {}).get("enabled", False)),
        # How cost is defined (core.pricing): a result priced under another
        # table or another set of charged phases is not the same result.
        "cost_definition": COST_DEFINITION,
    }


def _resume_fingerprint(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """What a resumed run may NOT change, flattened to dotted paths.

    The per-dataset half is `_eval_inputs` — what an evaluation's numbers
    depend on. The rest is what the SEARCH depends on rather than the
    evaluation: which axes the frontier carries, what the proposer was told,
    and how many candidates one step means.

    `steps` is deliberately absent. Raising it is the whole point of resuming.
    """
    out: Dict[str, Any] = {}
    for ds in sorted((cfg.get("datasets") or {})):
        inputs = _eval_inputs(cfg, ds)
        for key in sorted(inputs):
            out[f"datasets.{ds}.{key}"] = inputs[key]
    proposer = cfg.get("proposer") or {}
    out.update({
        "metrics": sorted(cfg.get("metrics") or []),
        "propose.k_per_step": (cfg.get("propose") or {}).get("k_per_step"),
        "sanity.enabled": (cfg.get("sanity") or {}).get("enabled"),
        "seed.arm": (cfg.get("seed") or {}).get("arm"),
        "seed.unified_models": (cfg.get("seed") or {}).get("unified_models"),
        "proposer.model": proposer.get("model"),
    })
    return out


def _fingerprint_diff(prior: Dict[str, Any], now: Dict[str, Any]) -> List[str]:
    """One human line per key that changed. Empty list = nothing moved."""
    lines: List[str] = []
    for key in sorted(set(prior) | set(now)):
        was, is_now = prior.get(key, "<absent>"), now.get(key, "<absent>")
        if was != is_now:
            lines.append(f"    {key}:  was {was!r}  →  now {is_now!r}")
    return lines


def _rotate_prior_snapshot(prior_path: Path) -> Path:
    """Move the previous snapshot aside so the record of what the earlier
    steps ran under is not overwritten by the config that replaced it."""
    for n in range(1, 1000):
        candidate = prior_path.with_name(f"config.prior{n:03d}.yaml")
        if not candidate.exists():
            prior_path.rename(candidate)
            return candidate
    raise RuntimeError(f"too many prior config snapshots beside {prior_path}")


def _guard_resume_config(cfg: Dict[str, Any], allow_change: bool) -> None:
    """Refuse to resume a workspace under a config that changes what its
    existing numbers mean.

    A resume is just "same --run-name", and `config.yaml` was overwritten on
    every launch and never read back — so a run could be continued on another
    dataset, another answerer, another prompt version, with no warning and
    with the record of what the earlier steps ran under destroyed. The
    frontier would then hold two incomparable populations and nothing would
    say so.
    """
    prior_path = paths.workspace / "config.yaml"
    if not prior_path.exists():
        return  # fresh workspace
    try:
        prior = yaml.safe_load(prior_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # unreadable snapshot is not a reason to refuse
        log.warning(
            f"resume: {prior_path} could not be read ({exc}); continuing "
            f"without the config check"
        )
        return

    diff = _fingerprint_diff(_resume_fingerprint(prior), _resume_fingerprint(cfg))
    if not diff:
        return

    body = "\n".join(diff)
    if allow_change:
        kept = _rotate_prior_snapshot(prior_path)
        log.warning(
            f"resume: this workspace's config changed, and --allow-config-change "
            f"was given, so the search continues:\n{body}\n"
            f"  the previous snapshot is kept at {kept} — the frontier now holds "
            f"entries produced under BOTH configs, and nothing in it records which"
        )
        return

    log.error(
        f"resume: workspace {paths.workspace.name} already holds results, and "
        f"this config changes what they mean:\n{body}\n"
        f"  Each of these feeds either the evaluation or the proposer, so old "
        f"and new entries would sit on the same frontier without being "
        f"comparable.\n"
        f"  → resume unchanged (only `steps` is meant to move), or start a new "
        f"--run-name, or pass --allow-config-change to accept a mixed frontier."
    )
    sys.exit(2)


# ---------------------------------------------------------------------------
# Proposer config resolution
# ---------------------------------------------------------------------------

def _proposer_opts(cfg: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """(model, agent_opts) from cfg.proposer. `agent_opts` is what the
    in-container script turns into Claude Code flags: `effort` → --effort,
    `disallowed_tools` → --disallowed-tools."""
    proposer_cfg = cfg.get("proposer", {}) or {}
    model = proposer_cfg.get("model")
    if not model:
        raise ValueError("No proposer model configured: set proposer.model in the YAML config.")
    opts: Dict[str, Any] = {"disallowed_tools": list(proposer_cfg.get("disallowed_tools") or [])}
    if proposer_cfg.get("effort") is not None:
        opts["effort"] = str(proposer_cfg["effort"])
    return model, opts


_ISOLATION_BINDS_CACHE: Optional[List[str]] = None


def _isolation_binds(cfg: Dict[str, Any]) -> Optional[List[str]]:
    """Search-split overlay binds: the held-out split (questions and gold
    answers) is invisible inside both containers during search. The
    orchestrator only ever runs the search split (held-out evaluation lives in
    evolution.heldout). Staged once per process; cached."""
    global _ISOLATION_BINDS_CACHE
    if _ISOLATION_BINDS_CACHE is None:
        from evolution.data_isolation import stage_search_data
        # Shared cross-run staging cache (fingerprinted against the source
        # data files) — the LongMemEval file is ~277 MB, so per-run
        # re-filtering would cost a full parse/dump each run.
        _ISOLATION_BINDS_CACHE = stage_search_data()
    return _ISOLATION_BINDS_CACHE


# ---------------------------------------------------------------------------
# Id allocation
# ---------------------------------------------------------------------------

#: The metrics a run may optimize. "accuracy" is always on — a search with no
#: score axis has nothing to climb.
KNOWN_METRICS = ("accuracy", "efficiency")


def _resolve_metrics(value: Any) -> List[str]:
    """Normalise `metrics:` (YAML list or comma-separated CLI string).

    Always includes "accuracy": dropping it would leave the proposer with no
    score to improve. Raises ValueError naming the unknown metric rather than
    silently ignoring a typo — a run that quietly stopped recording cost would
    be indistinguishable from one that never had it.
    """
    if isinstance(value, str):
        items = [v.strip() for v in value.split(",")]
    else:
        items = [str(v).strip() for v in (value or [])]
    items = [v for v in items if v]
    unknown = sorted(set(items) - set(KNOWN_METRICS))
    if unknown:
        raise ValueError(
            f"unknown metric(s) {unknown}; known metrics: {list(KNOWN_METRICS)}"
        )
    return [m for m in KNOWN_METRICS if m in set(items) | {"accuracy"}]


#: A harness is named by the content hash of its code, so the name says WHAT
#: the harness is rather than when it arrived (order lives in history.json).
#: While the proposer is still writing — and while the sanity-fix loop may
#: still rewrite memo.py — the code has no final identity, so the dir gets
#: a pending name and is renamed once the code settles.
HARNESS_ID_LEN = 12
_PENDING_RX = re.compile(r"^pending_[0-9a-f]{8}$")
_HARNESS_ID_RX = re.compile(r"^[0-9a-f]{%d}$" % HARNESS_ID_LEN)


def _new_pending_id() -> str:
    """A unique working name for a harness dir whose code is not settled yet."""
    return f"pending_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# history.json — the order the hash-named harnesses arrived in
# ---------------------------------------------------------------------------

def _read_history() -> Dict[str, Any]:
    """This workspace's history, or an empty skeleton."""
    path = paths.history_path
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("steps"), list):
                return data
            log.warning(f"{path} is malformed; starting a fresh history")
        except Exception as exc:
            log.warning(f"could not parse {path}: {exc} (starting a fresh history)")
    return {"run_id": paths.run_id, "steps": []}


def _record_history(step: int, kind: str, record: Dict[str, Any]) -> None:
    """Append one harness to `step`'s entry in history.json, rewriting the file.

    `step` 0 is the seed; 1..N are search steps. `kind` is "seed" or
    "propose". Written after every candidate rather than once at the end, so
    a run that dies mid-search still explains what it produced.
    """
    history = _read_history()
    for entry in history["steps"]:
        if entry.get("step") == step:
            entry["harnesses"].append(record)
            break
    else:
        history["steps"].append({
            "step": step,
            "kind": kind,
            "started_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "harnesses": [record],
        })
    history["steps"].sort(key=lambda e: e.get("step", 0))
    paths.history_path.parent.mkdir(parents=True, exist_ok=True)
    with paths.history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


#: The axes a run is actually searching on — what the frontier is compared by
#: and what history should therefore record. The rest of `_build_objectives`
#: (`score_max_<ds>`) is there so those two can be READ
#: correctly; it is not a dimension anyone is optimising.
_EVOLVING_AXIS_PREFIXES = ("accuracy_", "cost_usd_per_1k_queries_")


def evolving_axes(objectives: Dict[str, Any]) -> Dict[str, Any]:
    """Just the axes being optimised, from a full objectives dict."""
    return {k: v for k, v in objectives.items()
            if k.startswith(_EVOLVING_AXIS_PREFIXES)}


def _history_record(harness_id: str, sanity_status: str,
                    per_ds: Dict[str, Dict[str, Any]],
                    parent_ids: List[str],
                    metrics: Optional[List[str]] = None) -> Dict[str, Any]:
    """One harness's line in history.json: where it came from, how it did.

    `metrics` carries whatever this run is searching on — accuracy alone, or
    accuracy and cost — and is derived from `_build_objectives`, so the
    chronology and the frontier cannot disagree about a harness's numbers.

    The per-harness directory keeps the detail — every token split, the
    per-user spread, the full traces. This is the index, not the record.
    """
    record = {
        "id": harness_id,
        "sanity": sanity_status,
        "parent_ids": list(parent_ids),
        "metrics": evolving_axes(_build_objectives(per_ds, metrics)),
        "at": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    try:
        harness_dir = paths.harnesses_dir / harness_id
    except Exception:          # no workspace configured (tests build records bare)
        harness_dir = None
    proposer = _proposer_summary(harness_dir) if harness_dir is not None else None
    if proposer:
        record["proposer"] = proposer
    return record


def _analyst_summary(harness_id: str) -> Optional[Dict[str, Any]]:
    """What diagnosing this harness cost, from the analyst's own record.

    Same shape as `_proposer_summary` so one reader serves both. The record
    lives under `diagnostics/`, not in the harness dir: the analyst mounts the
    harness read-only, and a diagnosis is about a harness rather than part of
    it. None when the analyst is off, or crashed before writing anything.

    This is framework-side search cost, like the proposer's, and deliberately
    never reaches `cost_usd_per_1k_queries` — that axis is what the MEMORY
    SYSTEM costs to run, and `COST_DEFINITION` pins it to
    build+retrieve+compose+answer.
    """
    try:
        path = paths.workspace / "diagnostics" / f"{harness_id}.analyst.json"
    except Exception:          # no workspace configured (tests build records bare)
        return None
    if not path.exists():
        return None
    try:
        runs = json.loads(path.read_text(encoding="utf-8")).get("runs") or []
    except Exception as exc:
        log.warning(f"unreadable {path}: {exc}")
        return None
    if not runs:
        return None
    return {
        "cost_usd": round(sum(float(r.get("cost_usd") or 0.0) for r in runs), 4),
        "turns": sum(int(r.get("turns") or 0) for r in runs),
        "scripts": sum(int(r.get("scripts") or 0) for r in runs),
        "seconds": round(sum(float(r.get("duration_ms") or 0.0) for r in runs) / 1000.0, 1),
        "runs": len(runs),
    }


def _amend_history(harness_id: str, key: str, value: Any) -> None:
    """Set `key` on a harness already recorded in history.json.

    The analyst runs after its harness's line is written, so its cost cannot
    be built into `_history_record` the way the proposer's is — it is filled in
    afterwards. Bookkeeping only: a failure here is logged and swallowed,
    because losing a cost figure must not lose the diagnosis it describes.
    """
    try:
        history = _read_history()
        for entry in history["steps"]:
            for record in entry.get("harnesses", []):
                if record.get("id") == harness_id:
                    record[key] = value
                    with paths.history_path.open("w", encoding="utf-8") as f:
                        json.dump(history, f, indent=2, ensure_ascii=False)
                    return
        log.warning(f"history.json has no line for {harness_id}; {key} not recorded")
    except Exception as exc:
        log.warning(f"could not record {key} for {harness_id} in history.json: {exc}")


def _proposer_summary(harness_dir: Path) -> Optional[Dict[str, Any]]:
    """What writing this harness cost, summed over the proposer's runs on it
    (the proposal and every sanity-fix attempt), from the `.proposer.json`
    the in-container wrapper leaves in the harness dir. None for seeds and
    for harnesses written before the record existed."""
    path = harness_dir / ".proposer.json"
    if not path.exists():
        return None
    try:
        runs = json.loads(path.read_text(encoding="utf-8")).get("runs") or []
    except Exception as exc:
        log.warning(f"unreadable {path}: {exc}")
        return None
    if not runs:
        return None
    return {
        "cost_usd": round(sum(float(r.get("cost_usd") or 0.0) for r in runs), 4),
        "turns": sum(int(r.get("turns") or 0) for r in runs),
        "scripts": sum(int(r.get("scripts") or 0) for r in runs),
        "seconds": round(sum(float(r.get("duration_ms") or 0.0) for r in runs) / 1000.0, 1),
        "runs": len(runs),
        "fix_attempts": sum(1 for r in runs if r.get("mode") == "fix"),
    }


# ---------------------------------------------------------------------------
# Content hashing — fingerprint of the harness code that gets evaluated.
# Hash computed AFTER any propose_with_fix retries settle so it reflects the
# code that actually runs in evaluate_harness.
# ---------------------------------------------------------------------------


def _finalize_harness_dir(harness_dir: Path) -> Tuple[str, Path, bool]:
    """Rename `harnesses/pending_<n>` → `harnesses/<content hash>`.

    Returns `(final_id, final_dir, is_duplicate)`. `is_duplicate` is True when
    a harness with this exact code was already evaluated in this workspace:
    the proposer wrote byte-identical code twice, so re-evaluating it would
    buy nothing but tokens. The pending copy is dropped and the caller reuses
    the existing directory and its results.

    Idempotent for dirs that already carry their final name.
    """
    if _HARNESS_ID_RX.match(harness_dir.name):
        return harness_dir.name, harness_dir, False

    full_hash = _compute_content_hash(harness_dir)
    final_id = full_hash[:HARNESS_ID_LEN]
    final_dir = harness_dir.parent / final_id
    if final_dir.exists() and final_dir != harness_dir:
        shutil.rmtree(harness_dir, ignore_errors=True)
        return final_id, final_dir, True
    if final_dir != harness_dir:
        os.rename(harness_dir, final_dir)

    # Stamp meta.json with content_hash + created_at (preserve CC's parent_ids
    # and description; don't overwrite an existing content_hash if it matches).
    meta_path = final_dir / "meta.json"
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as exc:
            log.warning(f"could not parse {meta_path}: {exc} (overwriting)")
            meta = {}
    meta.setdefault("parent_ids", [])
    meta.setdefault("description", "")
    meta["content_hash"] = full_hash
    meta.setdefault("created_at", _dt.datetime.now().isoformat(timespec="seconds"))
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return final_id, final_dir, False


# ---------------------------------------------------------------------------
# Seed harness bootstrap.
# When cfg.seed.enabled, package each registered baseline and copy it into
# this run at startup.
# ---------------------------------------------------------------------------

def _resolve_seed_source(seed_source: str, cfg: Optional[Dict[str, Any]] = None) -> Path:
    """Where a seed's code lives — a packaged registered baseline.

    Raises `seed_sources.SeedSourceError` for anything else. See
    `evolution/seed_sources.py` for what is accepted and why.
    """
    seed_cfg = dict((cfg or {}).get("seed") or {})
    return seed_sources.resolve(
        seed_source,
        arm=seed_cfg.get("arm") or "unified",
        unified_models=seed_cfg.get("unified_models"),
    )


def _bootstrap_seed(seed_source: str, cfg: Optional[Dict[str, Any]] = None) -> str:
    """Copy the seed library entry into a pending dir in this run's harnesses/.

    Returns that pending id. The dir is renamed to the
    seed's content hash once it settles, exactly like a proposed harness — so
    re-running a workspace that already holds this seed is detected as a
    duplicate and costs nothing.
    """
    src = _resolve_seed_source(seed_source, cfg)
    pending_id = _new_pending_id()
    dst = paths.harnesses_dir / pending_id
    paths.harnesses_dir.mkdir(parents=True, exist_ok=True)
    # The code, whatever shape it has: a proposal is usually one file, while a
    # packaged baseline carries a package tree (baselines/<name>/src/…)
    # — copying only top-level files left those behind and the harness failed
    # to import in the container.
    _copy_harness_code(src, dst)
    log.info(f"seed bootstrapped: {src} → {dst}")
    return pending_id


# ---------------------------------------------------------------------------
# Per-dataset output helpers.
# Eval artifacts land at workspace/harnesses/<id>/<dataset>/...
# Sanity artifacts land at workspace/harnesses/<id>/<dataset>/sanity/...
# ---------------------------------------------------------------------------

def _dataset_dir(harness_dir: Path, dataset: str) -> Path:
    return harness_dir / dataset


def _sanity_dir(harness_dir: Path, dataset: str) -> Path:
    return harness_dir / dataset / "sanity"


def _read_dataset_metrics(harness_dir: Path, dataset: str) -> Dict[str, Any]:
    """Read score.json + token_usage.json for one dataset, return a metrics dict.

    Keys:
      raw_score:        float, benchmark_overall_eval_score (native scale)
      score_max:        int, judge's max score (1 for every benchmark)
      per_user_stddev:  Optional[float], population stddev of per_user[*].reward;
                        None when fewer than 2 valid users (stddev undefined)
      tokens:           int, sum of total_tokens across all models in token_usage.json

    Defaults are chosen to make a missing/unreadable file behave like a 0-score
    completed eval (raw_score=0, score_max=1, no stddev, 0 tokens).
    """
    metrics: Dict[str, Any] = {
        "raw_score": 0.0,
        "score_max": 1,
        "per_user_stddev": None,
        "tokens": 0,
    }
    ds_dir = _dataset_dir(harness_dir, dataset)

    # Preferred: metrics.json — evaluate_memo's return dict written by the
    # container.
    metrics_json = ds_dir / "metrics.json"
    if metrics_json.exists():
        try:
            with metrics_json.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "raw_score" in data:
                return {**metrics, **data}
        except Exception as exc:
            log.warning(f"could not parse {dataset} metrics.json for {harness_dir.name}: {exc}")

    # Fallback: score.json (raw_score + score_max + per_user_stddev) — the
    # host-side failure marker written when a container produced no metrics.
    score_json = ds_dir / "score.json"
    if score_json.exists():
        try:
            with score_json.open("r", encoding="utf-8") as f:
                data = json.load(f)
            bes = data.get("benchmark_eval_score", {})
            metrics["raw_score"] = float(bes.get("benchmark_overall_eval_score", 0.0))
            metrics["score_max"] = int(bes.get("score_max", 1))
            rewards = [
                float(v["reward"])
                for v in data.get("per_user", {}).values()
                if v.get("reward") is not None
            ]
            if len(rewards) >= 2:
                # Population stddev (ddof=0) — variance across the evaluated users.
                mean = sum(rewards) / len(rewards)
                metrics["per_user_stddev"] = (
                    sum((r - mean) ** 2 for r in rewards) / len(rewards)
                ) ** 0.5
        except Exception as exc:
            log.warning(f"could not parse {dataset} score.json for {harness_dir.name}: {exc}")

    # token_usage.json: total tokens across all models and phases.
    tokens_json = ds_dir / "token_usage.json"
    if tokens_json.exists():
        try:
            from core.tokens import read_total_tokens
            with tokens_json.open("r", encoding="utf-8") as f:
                data = json.load(f)
            metrics["tokens"] = read_total_tokens(data)
        except Exception as exc:
            log.warning(f"could not parse {dataset} token_usage.json for {harness_dir.name}: {exc}")

    return metrics


def _write_failure(out_dir: Path, msg: str) -> None:
    """Write a stub score.json indicating orchestrator-level failure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark_eval_score": {
            "benchmark_overall_eval_score": 0.0,
            "benchmark_overall_eval_standard_deviation": 0.0,
        },
        "per_user": {},
        "invalid_users": [{"user_id": "orchestrator", "error": msg}],
    }
    with (out_dir / "score.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _publish_run_artifacts(run_dir: Path, dst_dir: Path) -> None:
    """Move evaluator outputs from the transient runs/ staging dir into the
    per-dataset target dir, then remove the staging dir.

    Why move (not copy): runs/ exists only because the container needs a
    writable /out separate from the read-only /harness bind. After the host
    has the artifacts, leaving them in runs/ doubles disk use and adds
    visual clutter. We `shutil.copy2 + rmtree` so the dst_dir always lands
    on the same filesystem regardless of where runs/ was bound.

    Includes `subprocess.log` (container-side Phase 1/2 progress) so users
    don't have to dig into runs/<...>/ to inspect what happened inside the
    container.

    dst_dir is a PERSISTENT location that may hold artifacts from a prior
    attempt (sanity retry N-1, orphan re-eval, resumed run). Those are
    removed FIRST — unconditionally — so that after publishing, dst_dir
    reflects exactly what THIS execution produced. Without this, a crashed
    attempt would inherit the previous attempt's score.json and the caller's
    "did we get a score?" check (and sanity-error collection, and stage
    promotion) would silently operate on stale results.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    # The container writes the whole evaluation layout into /out (score.json,
    # token_usage.json, run_record.json, traces/, metrics.json), so publish is
    # a full tree move: clear every prior eval artifact from dst_dir EXCEPT the
    # separately-published sanity/ dir, then copy everything the run produced.
    _PROTECTED = {"sanity"}
    if run_dir.exists():
        for stale in dst_dir.iterdir():
            if stale.name in _PROTECTED:
                continue
            try:
                if stale.is_dir():
                    shutil.rmtree(stale, ignore_errors=True)
                else:
                    stale.unlink()
            except OSError as exc:
                log.warning(f"publish: could not remove stale {stale}: {exc}")
        for src in run_dir.iterdir():
            dst = dst_dir / src.name
            if src.is_dir():
                shutil.copytree(str(src), str(dst), dirs_exist_ok=True)
            else:
                shutil.copy2(str(src), str(dst))
    # Drop the staging dir entirely. If a copy above failed, rmtree still
    # runs — the score.json that ends up in dst_dir remains the source of
    # truth, and duplicating staging on a partial failure isn't useful.
    shutil.rmtree(run_dir, ignore_errors=True)
    # Also rmdir the parent `runs/` if it's empty — keeps the workspace
    # listing clean when no eval is currently in flight. Silently a no-op
    # if a concurrent eval is using a sibling subdir.
    try:
        run_dir.parent.rmdir()
    except OSError:
        pass


def _fmt_accuracies(objectives: Dict[str, Any]) -> str:
    """`ds=0.42 ds2=0.11` from a frontier objectives dict (per-dataset axes).
    Empty string when no benchmark scored."""
    parts = [
        f"{k[len('accuracy_'):]}={float(v):.3f}"
        for k, v in sorted(objectives.items()) if k.startswith("accuracy_")
    ]
    return " ".join(parts) or "no-score"


def _build_objectives(
    per_dataset_metrics: Dict[str, Dict[str, Any]],
    metrics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """The axes a proposer chooses a prior by — and nothing else.

    Per-benchmark fields (one set per dataset, native scale preserved):
      - `accuracy_<dataset>`     raw judge score
      - `score_max_<dataset>`    judge's max, so the score can be read on its
                                 own scale without a lookup
      - `cost_usd_per_1k_queries_<dataset>`  the efficiency axis (lower = cheaper),
                                 written ONLY when `metrics` contains
                                 "efficiency" — a run that did not ask to
                                 optimise cost keeps a scoreboard with no cost
                                 on it, so the proposer cannot optimise what
                                 the run does not want.

    Deliberately NOT here:

      `tokens_total`, `tokens_build_total`, `tokens_memory_total`,
      `llm_calls_total`  — copies of `tokens`, `tokens_build`, `tokens_memory`
        and `llm_calls` in each dataset's `metrics.json`, summed. Nothing read
        them from the frontier; they are reachable through
        `scripts/search_history.py show`.

      `code_length` — telemetry that is not an objective in either direction
        (a long implementation that generalises and scores well is a good
        harness), and for a packaged baseline it is mostly the vendored method
        anyway. Recomputed on demand from the code tree.

      `robustness_<dataset>` — stddev of per-user reward. It survives in
        `metrics.json` as `per_user_stddev`, but it never belonged on an axis:
        the search split gives LoCoMo TWO conversations (a stddev of
        two numbers is a range), and LongMemEval scores one binary QA per user,
        where stddev is sqrt(p(1-p)) — a function of accuracy itself, carrying
        no information of its own while looking like a third dimension.

    There is NO cross-benchmark mean: each benchmark is recorded independently
    (a mean would reward imbalanced harnesses). The proposer reads frontier.json directly and picks
    priors itself — there is no algorithmic selection (Meta-Harness
    alignment), so no scalar objective is needed.
    """
    want_cost = "efficiency" in set(metrics or ["accuracy"])
    out: Dict[str, Any] = {}
    for ds, m in per_dataset_metrics.items():
        out[f"accuracy_{ds}"] = float(m.get("raw_score", 0.0))
        out[f"score_max_{ds}"] = float(int(m.get("score_max", 10)))
        # Money, not tokens: every call the memory made in build, retrieve
        # and answer, priced per model (core.pricing). Tokens of different
        # models are not additive; dollars are, and embeddings enter at
        # their real weight. The token readings stay in metrics.json.
        if want_cost and m.get("cost_usd_per_1k_queries") is not None:
            out[f"cost_usd_per_1k_queries_{ds}"] = float(m["cost_usd_per_1k_queries"])
    return out


# ---------------------------------------------------------------------------
# Sanity check — small real-data run per dataset, ensures harness runs without crashing
# ---------------------------------------------------------------------------

def _collect_sanity_errors(
    harness_dir: Path, datasets_config: Dict[str, Dict[str, Any]]
) -> Tuple[bool, str]:
    """Inspect sanity score.json for each dataset. Return (passed, error_trace).

    Harness fails sanity if ANY dataset has invalid_users non-empty OR any
    per_user entry has non-empty failure_info.
    """
    errors: List[str] = []
    for ds in datasets_config:
        score_path = _sanity_dir(harness_dir, ds) / "score.json"
        if not score_path.exists():
            errors.append(f"[{ds}] sanity score.json missing (singularity likely didn't launch)")
            continue
        try:
            with score_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            errors.append(f"[{ds}] failed to parse sanity score.json: {exc}")
            continue
        invalid = data.get("invalid_users", [])
        for iu in invalid:
            errors.append(f"[{ds}] invalid user {iu.get('user_id', '?')}: {iu.get('error', '')}")
        for uid, v in data.get("per_user", {}).items():
            fi = v.get("failure_info")
            if fi:
                errors.append(f"[{ds}] user {uid} partial failure: {fi}")
        # The container records every model the harness was refused. A
        # harness that caught the exception scored normally above; it still
        # fails here, with the trace the proposer needs to fix it.
        metrics_path = _sanity_dir(harness_dir, ds) / "metrics.json"
        if metrics_path.exists():
            try:
                violations = json.loads(metrics_path.read_text(encoding="utf-8")).get(
                    "model_policy_violations") or []
            except Exception:
                violations = []
            if violations:
                errors.append(f"[{ds}] {_model_policy_message(violations)}")
    if not errors:
        return True, ""
    return False, "\n".join(errors)


def _model_policy_message(violations: List[Dict[str, Any]]) -> str:
    """One line naming what was refused and the way out."""
    seen: Dict[str, int] = {}
    for v in violations:
        key = f"{v.get('model')} at {v.get('where')}"
        seen[key] = seen.get(key, 0) + 1
    named = "; ".join(f"{k} ×{n}" for k, n in seen.items())
    return (f"model policy violated — {len(violations)} call(s) named a model outside "
            f"the run's allowlist ({named}). Every LLM/embedding call must use the "
            f"run's unified model: leave `model` unset on core.llm.Agent / "
            f"Embedding, or read MAPE_UNIFIED_LLM / MAPE_UNIFIED_EMBEDDING.")


def _apply_model_policy(m: Dict[str, Any], harness_id: str, ds: str) -> Dict[str, Any]:
    """A full evaluation that recorded a model-policy violation scores zero
    and is marked failed, keeping the record; the score it produced was made
    with a model the run does not allow, so it is not comparable to any
    other entry on the frontier."""
    violations = m.get("model_policy_violations") or []
    if not violations:
        return m
    log.error(f"{harness_id} [{ds}] {_model_policy_message(violations)}")
    return {"raw_score": 0.0, "score_max": m.get("score_max", 10), "per_user_stddev": None,
            "tokens": m.get("tokens", 0), "failed": True,
            "model_policy_violations": violations}



def _compose_enabled(cfg: Dict[str, Any]) -> bool:
    """cfg.contract.compose — whether evolved harnesses may implement the
    third hook. Absent means the full three-hook contract."""
    return bool(((cfg.get("contract") or {}).get("compose", True)))


def _eval_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """The run-level settings every search-split evaluation of a harness takes."""
    return dict(
        model=cfg["model"], judge_model=cfg["judge_model"],
        max_sample_concurrent=cfg["max_sample_concurrent"],
        gpu=cfg["gpu"]["enabled"],
        data_isolation_binds=_isolation_binds(cfg),
        compose_enabled=_compose_enabled(cfg),
    )


def _proposer_kwargs(cfg: Dict[str, Any], *, sanity_enabled: bool) -> Dict[str, Any]:
    """What both proposer calls (first write and sanity fix) take from the run."""
    agent_model, agent_opts = _proposer_opts(cfg)
    return dict(
        model=agent_model,
        timeout_s=cfg["proposer"]["timeout_s"],
        sanity_enabled=sanity_enabled,
        active_datasets=list(cfg["datasets"].keys()),
        metrics=cfg.get("metrics"),
        compose_enabled=_compose_enabled(cfg),
        agent_opts=agent_opts,
        extra_binds=_isolation_binds(cfg),
    )


def _zero_scores(dataset_names: List[str]) -> Dict[str, Dict[str, Any]]:
    """Per-dataset metrics of a candidate that never became evaluable."""
    return {ds: {"raw_score": 0.0, "score_max": 1, "per_user_stddev": None, "tokens": 0}
            for ds in dataset_names}


def _make_entry(cfg: Dict[str, Any], harness_dir: Path,
                per_ds: Dict[str, Dict[str, Any]],
                parent_ids: Optional[List[str]] = None) -> Entry:
    """A frontier Entry from per-dataset metrics and the dir's meta.json."""
    meta = _read_meta(harness_dir)
    return Entry(
        id=harness_dir.name,
        objectives=_build_objectives(per_ds, cfg.get("metrics")),
        parent_ids=_read_parent_ids(harness_dir) if parent_ids is None else parent_ids,
        content_hash=meta.get("content_hash"),
        created_at=meta.get("created_at"),
    )


def _record_candidate(cfg: Dict[str, Any], frontier: Frontier, harness_id: str,
                      per_ds: Dict[str, Dict[str, Any]], *, sanity_status: str,
                      parent_ids: List[str], step: int, kind: str) -> None:
    """Write one evaluated candidate into the run's record: frontier entry,
    sanity status, history line."""
    entry = _make_entry(cfg, paths.harnesses_dir / harness_id, per_ds, parent_ids)
    frontier.add(entry)
    _persist_sanity_status(entry.id, sanity_status)
    frontier.save(paths.frontier_path)
    _record_history(step, kind, _history_record(harness_id, sanity_status, per_ds,
                                                parent_ids, cfg["metrics"]))

async def sanity_check_harness(
    harness_id: str,
    image_path: Path,
    *,
    datasets_config: Dict[str, Dict[str, Any]],
    model: str,
    judge_model: str,
    max_sample_concurrent: int,
    gpu: bool = False,
    data_isolation_binds: Optional[List[str]] = None,
    compose_enabled: bool = True,
) -> Tuple[bool, str]:
    """Run a sanity_check-sized evaluation on every dataset (a smoke plan in ONE container per dataset).

    Returns (passed, error_trace). Artifacts go to `<harness>/<dataset>/sanity/`.
    """
    harness_dir = paths.harnesses_dir / harness_id

    for ds, params in datasets_config.items():
        sanity_dst = _sanity_dir(harness_dir, ds)
        run_dir = paths.runs_dir / f"{harness_id}_{int(time.time())}_{ds}_sanity"
        crashed: Optional[Exception] = None
        try:
            await run_evaluation(
                harness_dir=harness_dir,
                image_path=image_path,
                out_dir=run_dir,
                dataset=ds,
                # Sanity exists only inside the search loop — always the
                # search split (held-out flows never run sanity).
                split="search",
                # smoke plan → evaluate_memo(smoke=True) in-container: ONE
                # sanity_check-sized pass, artifacts at the /out root.
                plan={"smoke": True, "size": None,
                      "sanity_check": params.get("sanity_check")},
                model=model,
                judge_model=params.get("judge_model", judge_model),
                max_sample_concurrent=max_sample_concurrent,
                gpu=gpu,
                data_isolation_binds=data_isolation_binds,
                compose_enabled=compose_enabled,
            )
        except Exception as exc:
            log.error(f"sanity crashed for {harness_id} [{ds}]: {exc}")
            crashed = exc
        # Always publish + cleanup, even on crash — the container may have
        # written subprocess.log / partial score.json that's useful for retry.
        _publish_run_artifacts(run_dir, sanity_dst)
        if not (sanity_dst / "score.json").exists():
            _write_failure(sanity_dst, f"sanity_crashed: {crashed}" if crashed else "no score.json produced")

    return _collect_sanity_errors(harness_dir, datasets_config)


# ---------------------------------------------------------------------------
# Full eval — one harness across every dataset (serial).
# ---------------------------------------------------------------------------

async def evaluate_harness(
    harness_id: str,
    image_path: Path,
    *,
    datasets_config: Dict[str, Dict[str, Any]],
    split: str = "search",        # search | test — which benchmark split
    smoke: bool = False,          # True = one sanity_check-sized pass
    model: str,
    judge_model: str,
    max_sample_concurrent: int,
    gpu: bool = False,
    data_isolation_binds: Optional[List[str]] = None,
    compose_enabled: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Run the evaluation on every dataset (serial): ONE pass per benchmark,
    sized by the dataset's REQUIRED `size` config block (raises ValueError if
    absent — no automatic whole-split fallback), published to
    `harnesses/<id>/<ds>/`.

    `smoke=True` runs the `sanity_check` sizes instead, on the search split.

    Returns per-dataset metrics dicts (raw_score, score_max, per_user_stddev,
    tokens, failed, ...) — consumed by `_build_objectives`.
    """
    harness_dir = paths.harnesses_dir / harness_id

    # ONE container per (harness, dataset) runs the whole evaluation via the
    # shared evaluate_memo. The host only launches the container, publishes
    # /out into harnesses/<id>/<ds>/, and reads back metrics.json.
    per_ds: Dict[str, Dict[str, Any]] = {}
    for ds, params in datasets_config.items():
        # Pre-flight sizing validation (cheap, pure) — a config error like a
        # missing `size` block must fail fast HERE, not surface as an opaque
        # in-container crash after a Singularity launch.
        if not smoke:
            resolve_size_spec(ds, params.get("size"))
        ds_judge = params.get("judge_model", judge_model)
        dst_root = _dataset_dir(harness_dir, ds)
        # smoke runs live on the search split (they exist only inside the
        # search loop); everything else uses the caller's split.
        stage_split = "search" if smoke else split
        plan = {
            "smoke": smoke,
            "size": params.get("size"),
            "sanity_check": params.get("sanity_check"),
        }
        run_dir = paths.runs_dir / f"{harness_id}_{int(time.time())}_{ds}"
        crashed: Optional[Exception] = None
        try:
            await run_evaluation(
                harness_dir=harness_dir,
                image_path=image_path,
                out_dir=run_dir,
                dataset=ds,
                split=stage_split,
                plan=plan,
                model=model,
                judge_model=ds_judge,
                max_sample_concurrent=max_sample_concurrent,
                gpu=gpu,
                data_isolation_binds=data_isolation_binds,
                compose_enabled=compose_enabled,
            )
        except Exception as exc:
            log.error(f"evaluation crashed for {harness_id} [{ds}]: {exc}")
            crashed = exc
        # Always publish + cleanup. If the container died mid-write,
        # subprocess.log + any partial artifacts are still useful for triage.
        _publish_run_artifacts(run_dir, dst_root)
        if not (dst_root / "score.json").exists():
            _write_failure(dst_root, f"eval_crashed: {crashed}" if crashed else "no score.json produced")
        # Metrics: the container writes metrics.json (evaluate_memo's return
        # dict). A missing/unreadable one (container crash before any score)
        # degrades to a 0-score failed entry — never indistinguishable from a
        # genuine 0-score run (score.json got the failure marker above).
        m: Dict[str, Any] = {}
        metrics_path = dst_root / "metrics.json"
        if metrics_path.exists():
            try:
                m = json.loads(metrics_path.read_text(encoding="utf-8"))
            except Exception as exc:
                log.error(f"{harness_id} [{ds}] unreadable metrics.json: {exc}")
        if not m:
            m = {"raw_score": 0.0, "score_max": 1, "per_user_stddev": None,
                 "tokens": 0, "failed": True}
        per_ds[ds] = _apply_model_policy(m, harness_id, ds)
    return per_ds


# ---------------------------------------------------------------------------
# One harness from propose through eval, in two halves.
#
# `_propose_one` writes the code and settles it; `_evaluate_prepared` names it
# and runs it. The split exists for k_per_step > 1: every candidate in a step
# must be proposed from the SAME state, so no evaluation, frontier write,
# history line or diagnosis may land between them. During a step's propose
# phase a sibling candidate's directory is therefore still `pending_*`, which
# the proposer prompt describes as a directory being written.
# ---------------------------------------------------------------------------

@dataclass
class _Prepared:
    """A candidate whose code has settled, before it is named and evaluated.

    `terminal_per_ds` is set when the candidate never got as far as being
    evaluable (its env would not build, or sanity failed past its retries): the
    scores are already known to be zero and `_evaluate_prepared` passes them
    straight through. `step_index` is the search step it belongs to (0 for a
    seed).
    """
    new_id: str
    harness_dir: Path
    image_path: Optional[Path]
    sanity_status: str
    step_index: int
    terminal_per_ds: Optional[Dict[str, Dict[str, Any]]] = None


async def _propose_one(
    cfg: Dict[str, Any],
    new_id: str,
    *,
    is_seed: bool = False,
    step_index: int = 0,
) -> _Prepared:
    """Write one harness and settle its code; do NOT name or evaluate it.

    Proposes (unless it is a seed), builds its environment image, and runs the
    sanity retry loop. A candidate that cannot become evaluable (env build
    failure, sanity failed past its retries) is named by its content hash here
    and returned with `terminal_per_ds` set; every other candidate keeps its
    pending name until `_evaluate_prepared`.

    sanity_status: "passed" / "passed_on_retry_N" / "failed" / "skipped" /
    "env_build_failed".
    """
    dataset_names = list(cfg["datasets"].keys())

    # Sanity is skipped entirely for smoke-test runs (the run IS a sanity-size
    # quick check by definition).
    sanity_enabled = (not cfg["smoke_test"]) and cfg["sanity"]["enabled"]
    harness_dir = paths.harnesses_dir / new_id

    def _give_up(status: str, reason: str, image_path) -> _Prepared:
        """Name the dir by its code, write the failure, and stop here."""
        nonlocal harness_dir
        if entry_file(harness_dir) is not None and not _HARNESS_ID_RX.match(harness_dir.name):
            _, harness_dir, _ = _finalize_harness_dir(harness_dir)
        for ds in dataset_names:
            _write_failure(_dataset_dir(harness_dir, ds), reason)
        return _Prepared(new_id=new_id, harness_dir=harness_dir,
                         image_path=image_path, sanity_status=status,
                         step_index=step_index,
                         terminal_per_ds=_zero_scores(dataset_names))

    # Propose (unless seed). No parent_id passed — the agent browses the
    # workspace itself and records its chosen prior(s) in meta.json::parent_ids.
    if not is_seed:
        try:
            await propose(
                new_id=new_id,
                analyst_enabled=bool((cfg.get("analyst") or {}).get("enabled")),
                **_proposer_kwargs(cfg, sanity_enabled=sanity_enabled),
            )
        except (TimeoutError, RuntimeError) as exc:
            # The proposer subprocess failed (auth, timeout, other infra
            # error); memo.py was almost certainly never written. Remove the
            # empty dir so resume is not polluted, then abort the search
            # loop: main() catches ProposerInfraError and exits 2.
            log.error(f"proposer failed for {new_id}: {exc}")
            if harness_dir.exists() and entry_file(harness_dir) is None:
                shutil.rmtree(harness_dir, ignore_errors=True)
                log.info(f"cleaned up empty harness dir {harness_dir}")
            raise ProposerInfraError(
                f"propose() failed for harness {new_id}: {exc}"
            ) from exc

    # Build (or look up the cached) env image. The harness is validated inside
    # the container, at sanity time or at the evaluation itself
    # (launch.py::_load_harness_class).
    try:
        image_path = await ensure_image(harness_dir)
    except EnvBuildError as exc:
        log.error(f"env build failed for {new_id}: {exc}")
        return _give_up("env_build_failed", f"env_build_failed: {exc}", None)

    # Sanity retry loop (seeds are trusted and skip it).
    sanity_status = "skipped"
    if sanity_enabled and not is_seed:
        max_retries = int(cfg["sanity"]["max_retries"])
        for attempt in range(max_retries + 1):
            passed, error_trace = await sanity_check_harness(
                new_id, image_path, datasets_config=cfg["datasets"],
                **_eval_kwargs(cfg),
            )
            if passed:
                sanity_status = "passed" if attempt == 0 else f"passed_on_retry_{attempt}"
                log.info(f"sanity OK for {new_id} (attempt {attempt}/{max_retries})")
                break
            log.warning(
                f"sanity FAILED for {new_id} (attempt {attempt}/{max_retries}):\n{error_trace}"
            )
            if attempt >= max_retries:
                sanity_status = "failed"
                break
            # Retry: ask the agent to fix its own harness from the error trace.
            try:
                await propose_with_fix(
                    new_id=new_id, error_trace=error_trace,
                    **_proposer_kwargs(cfg, sanity_enabled=sanity_enabled),
                )
            except (TimeoutError, RuntimeError) as exc:
                # Same infra failure as propose() above. The original memo.py
                # still exists, so the dir is kept; resume picks up cleanly.
                log.error(f"propose_with_fix failed for {new_id}: {exc}")
                raise ProposerInfraError(
                    f"propose_with_fix() failed for harness {new_id}: {exc}"
                ) from exc
            # The fix may have edited requirements.txt → re-resolve the image
            # (cached unless requirements.txt actually changed).
            try:
                image_path = await ensure_image(harness_dir)
            except EnvBuildError as exc:
                log.error(f"env rebuild after fix failed for {new_id}: {exc}")
                sanity_status = "failed"
                break

        if sanity_status == "failed":
            log.error(f"sanity_failed for {new_id}; skipping full eval")
            return _give_up("failed", "sanity_failed_after_retries", image_path)

    # Sanity settled. The code is final; naming and evaluating it belong to the
    # other half, so that a step can settle every candidate before any of them
    # produces a result the next one could read.
    return _Prepared(new_id=new_id, harness_dir=harness_dir,
                     image_path=image_path, sanity_status=sanity_status,
                     step_index=step_index)


async def _evaluate_prepared(
    cfg: Dict[str, Any],
    prepared: _Prepared,
) -> Tuple[str, Dict[str, Dict[str, Any]], str, List[str]]:
    """Name a settled candidate by its content hash and evaluate it.

    Returns (final_id, per_dataset_metrics, sanity_status, parent_ids).
    """
    dataset_names = list(cfg["datasets"].keys())
    harness_dir = prepared.harness_dir

    if prepared.terminal_per_ds is not None:
        # Never became evaluable. Its directory was already settled and its
        # failure already written by the half that gave up on it.
        return (harness_dir.name, prepared.terminal_per_ds,
                prepared.sanity_status, _read_parent_ids(harness_dir))

    # Name the dir by the hash of the code that will actually run, then
    # evaluate it — unless this exact code was already evaluated in this
    # workspace, in which case its results stand.
    final_id, is_dup = harness_dir.name, False
    if entry_file(harness_dir) is not None and not _HARNESS_ID_RX.match(harness_dir.name):
        final_id, harness_dir, is_dup = _finalize_harness_dir(harness_dir)
    if is_dup:
        log.info(
            f"{prepared.new_id}: byte-identical to harnesses/{final_id} — reusing "
            f"its evaluation instead of re-running it"
        )
        return (final_id,
                {ds: _read_dataset_metrics(harness_dir, ds) for ds in dataset_names},
                "duplicate", _read_parent_ids(harness_dir))

    per_ds = await evaluate_harness(
        final_id, prepared.image_path,
        datasets_config=cfg["datasets"], split="search", smoke=cfg["smoke_test"],
        **_eval_kwargs(cfg),
    )
    return final_id, per_ds, prepared.sanity_status, _read_parent_ids(harness_dir)


async def _seed_one(cfg: Dict[str, Any], source: str, frontier: Frontier,
                    dataset_names: List[str]) -> Optional[str]:
    """Put one seed into this run's step 0. Returns its harness id, or None.

    A seed this workspace already evaluated (a resumed run) is left as is;
    otherwise it is evaluated like any other harness.
    """
    log.info(f"seed {source!r}: {seed_sources.describe(source)}")
    src = _resolve_seed_source(source, cfg)
    try:
        seed_id = _compute_content_hash(src)[:HARNESS_ID_LEN]
    except Exception as exc:                 # noqa: BLE001
        log.error(f"cannot read seed {source!r}: {exc}")
        return None

    if frontier.get(seed_id) is not None:
        log.info(f"seed {source!r} ({seed_id}) already evaluated in this workspace")
        return seed_id

    pending_id = _bootstrap_seed(source, cfg)
    if pending_id is None:
        return None
    log.info(f"evaluating seed {source!r} on {dataset_names} ...")
    prepared = await _propose_one(cfg, new_id=pending_id, is_seed=True)
    final_id, per_ds, sanity_status, parent_ids = await _evaluate_prepared(cfg, prepared)
    _record_candidate(cfg, frontier, final_id, per_ds, sanity_status=sanity_status,
                      parent_ids=parent_ids, step=0, kind="seed")
    log.info(
        f"seed {final_id}: "
        + ", ".join(f"{ds}={per_ds.get(ds, {}).get('raw_score', 0):.3f}" for ds in per_ds)
        + f"  [sanity={sanity_status}]"
    )
    await _run_analyst(cfg, final_id, dataset_names=dataset_names,
                       parent_ids=parent_ids, sanity_status=sanity_status,
                       is_seed=True)
    return final_id


async def _run_analyst(cfg: Dict[str, Any], harness_id: str, *,
                       dataset_names: List[str], parent_ids: List[str],
                       sanity_status: str, is_seed: bool = False) -> None:
    """Have the analyst read this candidate's evidence and write its diagnosis.

    This is CONTEXT FOR THE NEXT STEP, never a gate on this one: a failed or
    timed-out analyst is logged and the loop continues, because a missing
    diagnosis costs the next proposer some evidence and nothing else.

    The harness directory goes in read-only (evolution/proposer.py::analyze), and
    its content hash is checked either side of the session. That hash is the
    directory's name and the frontier points at it, so a changed hash would
    mean the recorded population no longer matches the code that produced it.
    """
    an = cfg.get("analyst") or {}
    if not an.get("enabled"):
        return
    if is_seed and not an.get("include_seeds", True):
        return
    if sanity_status not in ("passed", "skipped"):
        return      # nothing was evaluated, so there is nothing to explain

    from evolution.codetree import _compute_content_hash
    from evolution.proposer import analyze

    harness_dir = paths.harnesses_dir / harness_id
    before = _compute_content_hash(harness_dir)
    _, agent_opts = _proposer_opts(cfg)
    agent_opts = dict(agent_opts or {})
    if an.get("effort"):
        agent_opts["effort"] = an["effort"]

    t0 = time.time()
    try:
        await analyze(
            harness_id,
            datasets=dataset_names,
            parent_ids=parent_ids,
            model=an.get("model") or "claude-opus-4-8",
            timeout_s=int(an.get("timeout_s", 30 * 60)),
            agent_opts=agent_opts,
            extra_binds=_isolation_binds(cfg),
        )
    except Exception as exc:
        log.warning(
            f"analyst[{harness_id}] failed ({type(exc).__name__}: {str(exc)[:200]}); "
            f"the next proposer runs without a diagnosis for it"
        )
    finally:
        after = _compute_content_hash(harness_dir)
        if after != before:
            log.error(
                f"analyst[{harness_id}] CHANGED the harness directory "
                f"({before} → {after}). The frontier entry no longer matches the "
                f"code that produced it. The workspace is bound read-only for the "
                f"analyst, so this means the bind did not apply — treat this run's "
                f"population as suspect."
            )

    md = paths.workspace / "diagnostics" / f"{harness_id}.md"
    js = paths.workspace / "diagnostics" / f"{harness_id}.json"
    # The analyst is ~a third of a run's total spend (measured: $17-18 against
    # $28-36 of proposer on a 10-step run), so it belongs in the record next to
    # the proposer's, not only in a log line that scrolls away.
    summary = _analyst_summary(harness_id)
    cost = float((summary or {}).get("cost_usd") or 0.0)
    if summary:
        _amend_history(harness_id, "analyst", summary)
    if md.exists():
        log.info(
            f"analyst[{harness_id}]: diagnosis written "
            f"({md.stat().st_size} B md, {'json ok' if js.exists() else 'JSON MISSING'}, "
            f"${cost:.2f}, {time.time() - t0:.0f}s)"
        )
    else:
        log.warning(
            f"analyst[{harness_id}] wrote no diagnosis at {md} "
            f"(${cost:.2f} spent); the next proposer runs without one"
        )


# ---------------------------------------------------------------------------
# Outer loop
# ---------------------------------------------------------------------------

async def search_loop(cfg: Dict[str, Any]) -> None:
    ensure_dirs()
    frontier = Frontier.load(paths.frontier_path)

    # Resume cleanup: drop entries that never got code (the proposer returned
    # without writing memo.py), so they are proposed again instead of eating
    # the budget. Persisted immediately so the cleanup survives an abort.
    if _cleanup_polluting_entries(frontier) > 0:
        frontier.save(paths.frontier_path)

    # Resume orphan adoption: re-attach harness dirs that exist on disk but
    # aren't in frontier (e.g. orchestrator killed mid-eval). Runs BEFORE
    # seed-bootstrap so an orphaned seed dir is auto-detected and the
    # `seed_already_in_frontier` check below sees it.
    if cfg.get("adopt_orphans", True):
        n_adopted = await _adopt_orphan_dirs(cfg, frontier)
        if n_adopted:
            log.info(f"adoption: re-attached {n_adopted} orphan harness(es) from disk")

    datasets_config: Dict[str, Dict[str, Any]] = cfg["datasets"]
    dataset_names = list(datasets_config.keys())

    # Step 0: the seeds. Each source is copied in, identified by the hash of
    # its code like every other harness, and evaluated — unless this workspace
    # already holds it.
    seed_ids: List[str] = []
    if cfg["seed"]["enabled"]:
        for source in cfg["seed"]["sources"]:
            try:
                seed_id = await _seed_one(cfg, source, frontier, dataset_names)
            except seed_sources.SeedSourceError as exc:
                # A misnamed seed is a config error, not a failure to recover
                # from: the run would otherwise start from something nobody
                # chose. Reported as text, not a traceback — the message names
                # what IS accepted.
                log.error(f"seed {source!r} cannot be used:\n{exc}")
                return
            if seed_id is None:
                log.error(f"seed {source!r} failed; aborting search")
                return
            seed_ids.append(seed_id)
    else:
        log.info("no seeds — the first proposed harness has no prior to reference")

    # Search loop. No parent selection — for each step, propose k harnesses
    # back-to-back; each call lets the proposer browse the workspace and pick
    # its own priors. The dir is renamed to its content hash after sanity settles.
    #
    # Resume semantics: cfg["steps"] is the TOTAL number of search
    # steps across all invocations of this --run-name. We compute "proposals
    # already done" from the frontier (post-cleanup, every entry is real) and
    # only run the remaining count. Empty workspace → done=0 → fresh run with
    # full budget. Same workspace re-launched → done=N → only `total - N` more.
    k = int(cfg["propose"]["k_per_step"])
    total = cfg["steps"] * k
    done = len(frontier.all_entries())
    # Seeds are baseline references, not "search proposals" — subtract them
    # from `done` so they do not eat into the proposal budget.
    done = max(0, done - sum(1 for sid in seed_ids if frontier.get(sid) is not None))

    if done >= total:
        log.info(
            f"all {total} proposals already complete in this workspace; nothing to do"
        )
        return

    log.info(
        f"{'resuming' if done > 0 else 'starting'} search: {done}/{total} "
        f"proposals already done; running {total - done} more"
    )

    # One step = propose k candidates, THEN evaluate and record them. With k>1
    # that ordering is the whole point: a candidate must not be able to read a
    # sibling's score, frontier entry, history line or diagnosis, because they
    # are meant to be k draws from one state. With k=1 the phases collapse and
    # the sequence is identical to when this was a flat loop over candidates.
    #
    # Resume is therefore at candidate granularity still, not step granularity:
    # `done` counts candidates, so a run interrupted after 2 of 3 in a step
    # picks up at the third.
    start_step = done // k + 1
    skip_in_start_step = done % k
    for step in range(start_step, cfg["steps"] + 1):
        first_j = (skip_in_start_step if step == start_step else 0) + 1
        if first_j > k:
            continue

        # ---- propose the step's candidates, from one shared state ----------
        prepared: List[Tuple[int, _Prepared]] = []
        for j in range(first_j, k + 1):
            pending_id = _new_pending_id()
            log.info(f"[step {step}/{cfg['steps']}] candidate {j}/{k} → {pending_id}")
            try:
                prepared.append((j, await _propose_one(
                    cfg, new_id=pending_id, is_seed=False, step_index=step,
                )))
            except ProposerInfraError as exc:
                log.error(
                    f"[step {step}/{cfg['steps']}] propose subprocess failed for "
                    f"{pending_id}: {exc}\n"
                    f"  → search aborted to avoid generating placeholder entries.\n"
                    f"  → Resume with the SAME --run-name (total {cfg['steps']} "
                    f"steps; {done + len(prepared)} of {total} proposals done "
                    f"before this attempt's failure)."
                )
                # Candidates already settled in this step are evaluated and
                # recorded before the abort: their code exists and was paid for,
                # and leaving them unrecorded is what the orphan scan then has
                # to guess about.
                for jj, prep in prepared:
                    await _settle_candidate(cfg, prep, step=step, j=jj, k=k,
                                            frontier=frontier,
                                            dataset_names=dataset_names)
                raise   # propagate to main() for sys.exit(2)

        # ---- evaluate and record them, in proposal order -------------------
        for j, prep in prepared:
            await _settle_candidate(cfg, prep, step=step, j=j, k=k,
                                    frontier=frontier,
                                    dataset_names=dataset_names)
        done += len(prepared)



async def _settle_candidate(
    cfg: Dict[str, Any],
    prepared: _Prepared,
    *,
    step: int,
    j: int,
    k: int,
    frontier: Frontier,
    dataset_names: List[str],
) -> None:
    """Evaluate one prepared candidate and write it into the run's record.

    Lifted out of the search loop so the abort path records the candidates a
    failed step had already settled, instead of leaving them for the orphan scan
    to classify.
    """
    final_id, per_ds, sanity_status, parent_ids = await _evaluate_prepared(cfg, prepared)
    _record_candidate(cfg, frontier, final_id, per_ds, sanity_status=sanity_status,
                      parent_ids=parent_ids, step=step, kind="propose")
    log.info(
        f"[step {step}/{int(cfg['steps'])}] candidate {j}/{k} {final_id} "
        + ", ".join(
            f"{ds}={per_ds.get(ds, {}).get('raw_score', 0):.3f}"
            for ds in dataset_names
        )
        + f"  [sanity={sanity_status}]  parent_ids={parent_ids}"
    )
    await _run_analyst(cfg, final_id, dataset_names=list(dataset_names),
                       parent_ids=parent_ids, sanity_status=sanity_status)

# ---------------------------------------------------------------------------
# Resume orphan adoption: re-attach harness dirs that exist on disk but
# aren't in frontier (e.g. orchestrator killed mid-eval). Classify each by
# file presence and run ONLY the missing pipeline stages.
#
# Cases handled (a dir whose sanity never finished is discarded, not
# re-run):
#
#   complete       hash-suffix dir + every <ds>/score.json present  →
#                  build Entry from on-disk metrics, no eval
#   sanity_passed  hash-suffix dir + every <ds>/sanity/score.json   →
#                  call evaluate_harness for the missing full eval
#   sanity_failed  hash-suffix dir + every <ds>/score.json contains
#                  the "sanity_failed_after_retries" sentinel       →
#                  build all-zero Entry, no eval
#   incomplete     anything else (no memo.py, no hash suffix,
#                  partial sanity, etc.)                            →
#                  rm -rf the dir + warn
# ---------------------------------------------------------------------------

# A valid harness dir is named by its content hash; `pending_*` is one whose
# code never settled. Anything else (e.g. user-created `notes/`, transient
# scratch files) is filtered out by _scan_orphan_dirs.


def _scan_orphan_dirs(frontier: Frontier) -> List[Path]:
    """Harness dirs on disk that `frontier` does not know about.

    A `pending_*` dir is a harness whose code never settled (the run died
    while the proposer was writing it), so it has no identity to adopt: it is
    returned too, and `_classify_orphan` discards it. Ordered by the dir's own
    creation time so adoption replays roughly the order they arrived in.
    """
    if not paths.harnesses_dir.exists():
        return []
    known = {e.id for e in frontier.all_entries()}
    orphans = [
        child for child in paths.harnesses_dir.iterdir()
        if child.is_dir()
        and (_HARNESS_ID_RX.match(child.name) or _PENDING_RX.match(child.name))
        and child.name not in known
    ]
    orphans.sort(key=lambda p: (p.stat().st_mtime, p.name))
    return orphans


def _classify_orphan(harness_dir: Path, dataset_names: List[str]) -> str:
    """Return 'complete' | 'sanity_passed' | 'sanity_failed' | 'incomplete'.

    Decision tree (in order):
      - no memo.py                                     → 'incomplete'
      - dir still carries a pending name               → 'incomplete'
      - meta.json missing or content_hash missing      → 'incomplete'
      - every <ds>/sanity/score.json present:
          if every <ds>/score.json present             → 'complete'
          else                                         → 'sanity_passed'
      - no <ds>/sanity/ exists for any ds AND every
        <ds>/score.json contains the sanity-failed
        sentinel                                       → 'sanity_failed'
      - else (partial / mixed-state)                   → 'incomplete'
    """
    if entry_file(harness_dir) is None:
        return "incomplete"
    if not _HARNESS_ID_RX.match(harness_dir.name):
        return "incomplete"
    meta = _read_meta(harness_dir)
    if not meta.get("content_hash"):
        return "incomplete"

    sanity_score_paths = [
        _sanity_dir(harness_dir, ds) / "score.json" for ds in dataset_names
    ]
    full_score_paths = [
        _dataset_dir(harness_dir, ds) / "score.json" for ds in dataset_names
    ]
    has_all_sanity = all(p.exists() for p in sanity_score_paths)
    has_all_full = all(p.exists() for p in full_score_paths)
    has_no_sanity_dirs = not any(
        _sanity_dir(harness_dir, ds).exists() for ds in dataset_names
    )

    if has_all_sanity:
        return "complete" if has_all_full else "sanity_passed"

    # No sanity dirs at all + every full-eval score.json carries the
    # "sanity_failed_after_retries" marker that _propose_one writes when the
    # sanity retry loop is exhausted.
    if has_no_sanity_dirs and has_all_full:
        for p in full_score_paths:
            try:
                with p.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                return "incomplete"
            invalid = data.get("invalid_users", [])
            if not any(
                isinstance(u, dict) and u.get("error") == "sanity_failed_after_retries"
                for u in invalid
            ):
                return "incomplete"
        return "sanity_failed"

    return "incomplete"


async def _adopt_orphan(
    cfg: Dict[str, Any], harness_dir: Path, frontier: Frontier
) -> Optional[Entry]:
    """Re-enter the pipeline at the right stage for one orphan.

    Returns the new Entry (already added to frontier — caller batches the
    save) or None if the orphan was discarded.
    """
    dataset_names = list(cfg["datasets"].keys())
    kind = _classify_orphan(harness_dir, dataset_names)
    name = harness_dir.name

    if kind == "incomplete":
        log.warning(
            f"adoption: discarded incomplete orphan harnesses/{name} "
            f"(missing memo.py / unsettled name / content_hash, or partial sanity/eval state)"
        )
        shutil.rmtree(harness_dir, ignore_errors=True)
        return None

    # Defensive content_hash check: detects post-finalize edits to memo.py,
    # which would mean the eval runs different code than the dir name claims.
    expected_hash = name
    try:
        actual_hash = _compute_content_hash(harness_dir)[:HARNESS_ID_LEN]
    except Exception as exc:
        log.warning(
            f"adoption: discarded harnesses/{name} — could not "
            f"compute content_hash: {exc}"
        )
        shutil.rmtree(harness_dir, ignore_errors=True)
        return None
    if actual_hash != expected_hash:
        log.warning(
            f"adoption: discarded harnesses/{name} — content_hash mismatch "
            f"(computed={actual_hash}, dir name={expected_hash}); "
            f"memo.py was modified post-finalize"
        )
        shutil.rmtree(harness_dir, ignore_errors=True)
        return None

    if kind in ("sanity_failed", "complete"):
        # Scores already on disk (synthetic zeros for sanity_failed).
        log.info(f"adoption: harnesses/{name} → {kind} (no eval needed)")
        per_ds = {ds: _read_dataset_metrics(harness_dir, ds) for ds in dataset_names}
        status = "adopted_failed" if kind == "sanity_failed" else "adopted_complete"
    else:
        # Sanity passed but the full eval is missing/partial: run it only.
        log.info(
            f"adoption: harnesses/{name} → sanity_passed; running evaluate_harness "
            f"on {dataset_names}"
        )
        try:
            image_path = await ensure_image(harness_dir)
        except EnvBuildError as exc:
            log.error(
                f"adoption: env build failed for harnesses/{name}: {exc}; "
                f"adopting as failed Entry with all-zero objectives"
            )
            for ds in dataset_names:
                _write_failure(_dataset_dir(harness_dir, ds), f"env_build_failed: {exc}")
            per_ds = {ds: _read_dataset_metrics(harness_dir, ds) for ds in dataset_names}
            status = "adopted_env_build_failed"
        else:
            per_ds = await evaluate_harness(
                name, image_path, datasets_config=cfg["datasets"], split="search",
                smoke=cfg["smoke_test"], **_eval_kwargs(cfg),
            )
            status = "adopted_passed"

    entry = _make_entry(cfg, harness_dir, per_ds)
    frontier.add(entry)
    _persist_sanity_status(name, status)
    return entry


async def _adopt_orphan_dirs(cfg: Dict[str, Any], frontier: Frontier) -> int:
    """Top-level: scan + classify + adopt orphan harness dirs in id order.

    Saves the frontier once at the end if any adoption succeeded. Returns
    the count successfully adopted (incomplete/discarded don't count).
    """
    orphans = _scan_orphan_dirs(frontier)
    if not orphans:
        return 0
    log.info(
        f"adoption: scanning harnesses/ for orphans not in frontier — "
        f"found {len(orphans)} candidate(s): {[p.name for p in orphans]}"
    )
    adopted = 0
    for harness_dir in orphans:
        try:
            entry = await _adopt_orphan(cfg, harness_dir, frontier)
        except Exception as exc:
            log.error(
                f"adoption: unexpected error adopting harnesses/{harness_dir.name}: "
                f"{exc}; leaving dir as-is"
            )
            continue
        if entry is not None:
            adopted += 1
            log.info(
                f"adoption: re-attached {entry.id} "
                f"[{_fmt_accuracies(entry.objectives)}]"
            )
    if adopted:
        frontier.save(paths.frontier_path)
    return adopted


# ---------------------------------------------------------------------------
# Resume cleanup. Sanity-failed harnesses (hash + memo.py) are kept as
# legitimate score-0 data points; only entries without code are dropped.
# ---------------------------------------------------------------------------

def _cleanup_polluting_entries(frontier: Frontier) -> int:
    """Drop entries with no content_hash (no code was ever written) and remove
    their directories when they hold no memo.py. Returns the number removed;
    the caller saves the frontier when it is nonzero."""
    ids = {e.id for e in frontier.all_entries() if e.content_hash is None}
    if not ids:
        return 0
    for entry_id in ids:
        harness_dir = paths.harnesses_dir / entry_id
        if harness_dir.exists() and entry_file(harness_dir) is None:
            shutil.rmtree(harness_dir, ignore_errors=True)
    n_removed = frontier.remove_by_ids(ids)
    log.info(f"cleanup: removed {n_removed} frontier entries that never got code")
    return n_removed


# ---------------------------------------------------------------------------
# sanity_status is orthogonal to selection; we keep it as a sidecar file
# so we don't have to plumb it through the Frontier dataclass yet.
# ---------------------------------------------------------------------------

def _persist_sanity_status(harness_id: str, status: str) -> None:
    ds_dir = paths.harnesses_dir / harness_id
    ds_dir.mkdir(parents=True, exist_ok=True)
    try:
        with (ds_dir / "sanity_status.txt").open("w", encoding="utf-8") as f:
            f.write(status + "\n")
    except OSError as exc:
        log.warning(f"could not write sanity_status for {harness_id}: {exc}")


def _read_meta(harness_dir: Path) -> Dict[str, Any]:
    """Read meta.json (returns {} if missing/invalid)."""
    meta_path = harness_dir / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        log.warning(f"could not parse meta.json for {harness_dir.name}: {exc}")
        return {}


def _read_parent_ids(harness_dir: Path) -> List[str]:
    """Read `meta.json::parent_ids`.

    Returns:
        List of harness ids that the proposer says it drew from. Empty list
        if meta.json is missing or doesn't list any (e.g. seed harness).
    """
    meta = _read_meta(harness_dir)
    if not meta:
        return []
    raw = meta.get("parent_ids")
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw] if raw else []
    if isinstance(raw, list):
        return [str(x) for x in raw if x]
    log.warning(
        f"meta.json::parent_ids in {harness_dir.name} has unexpected type "
        f"{type(raw).__name__}; ignoring"
    )
    return []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    # allow_abbrev=False: avoid surprising prefix matching. Without this,
    # `--mode check` would be parsed as `--model check` (since "mode" is a
    # prefix of "model"), silently using "check" as the OpenAI model name
    # rather than erroring as "unrecognized argument".
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a YAML config file. CLI flags override its values.",
    )
    parser.add_argument(
        "--strict-config", dest="strict_config",
        action=argparse.BooleanOptionalAction, default=None,
        help="Require --config to list every evolution parameter explicitly "
             "(default on when --config is given; a null value still counts "
             "as provided). --no-strict-config to disable.",
    )
    # Each override defaults to None so we can detect "explicitly set" vs "use config/default"
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument(
        "--datasets", default=None,
        help="Comma-separated dataset names. If given, replaces any YAML "
             "`datasets:` block; each benchmark is evaluated on its whole "
             "search split (set `size` in the YAML for a subset).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run the whole pipeline once per benchmark at "
             "sanity_check sizes (search split; no sanity gate) — a cheap "
             "end-to-end pipeline check. Without "
             "this flag the orchestrator runs the normal search loop. "
             "Held-out evaluation: use `python -m evolution.heldout`.",
    )

    parser.add_argument("--model", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--max-sample-concurrent", type=int, default=None)

    parser.add_argument("--proposer-model", default=None)
    parser.add_argument("--proposer-timeout-s", type=int, default=None)
    parser.add_argument("--metric", dest="metrics", type=str, default=None,
                        help="What this search optimizes: 'accuracy' (default) "
                             "or 'accuracy,efficiency'. Everything is measured "
                             "either way — this picks the axes the frontier "
                             "carries and what the proposer is told to trade off.")
    parser.add_argument(
        "--proposer-disallowed-tools", default=None,
        help='Comma-separated list of tools the proposer cannot use (default ["mcp__*"]). '
             'Pass "" to allow all scripts. Examples: "mcp__*", "Bash,WebFetch,mcp__*".',
    )

    parser.add_argument("--k-per-step", type=int, default=None,
                        help="Harnesses proposed per outer-loop step (default 2)")

    parser.add_argument("--no-sanity", action="store_true",
                        help="Disable the pre-eval sanity gate (mode=search/test; dev always skips it)")
    parser.add_argument("--sanity-max-retries", type=int, default=None,
                        help="Max propose_with_fix retries when sanity fails (default 2)")

    parser.add_argument("--run-name", default=None,
                        help="Per-run workspace dir under workspace/. "
                             "Defaults to a timestamp (YYYYMMDD_HHMMSS).")
    parser.add_argument("--no-seed", action="store_true",
                        help="Skip copying the seed harness — first proposed "
                             "harness has no prior to reference.")
    parser.add_argument("--seed", dest="seed_sources", default=None,
                        help="Comma-separated seed sources for step 0: names of "
                             "registered baselines (e.g. no_memory, mem0). "
                             "Enables seeding.")

    parser.add_argument("--gpu", action="store_true",
                        help="Pass --nv to evaluator's singularity exec so the "
                             "container can use the host's NVIDIA driver. "
                             "Default off (CPU-only).")
    parser.add_argument("--allow-config-change", action="store_true",
                        help="Resume a workspace whose config changed in a way "
                             "that makes its existing results incomparable. Off "
                             "by default: such a resume aborts and names what "
                             "moved. The previous config.yaml is kept as "
                             "config.priorNNN.yaml when this is used.")
    parser.add_argument("--no-adopt-orphans", action="store_true",
                        help="Skip orphan-harness-dir adoption pass on resume. "
                             "Default ON: dirs on disk but not in frontier are "
                             "re-attached (only the missing pipeline stages re-run). "
                             "Pass this to force a clean restart instead.")

    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    _setup_logging(args.verbose)
    cfg = _resolve_config(args)
    paths.set_run_id(cfg["run_name"])
    run_log = _attach_run_log()
    try:
        # Load now so a missing or malformed part surfaces at startup, not
        # halfway through the search loop. The result is cached.
        load_prompt_parts()
    except PromptPartsError as exc:
        log.error(f"prompts: {exc}")
        sys.exit(2)

    log.info(f"proposer: Claude Code, model={cfg['proposer']['model']}, auth=ANTHROPIC_API_KEY")

    _guard_resume_config(cfg, allow_change=args.allow_config_change)
    # The eval container's model policy: QA model and judge go per call; the
    # embedding is the one the seeds were packaged for.
    from evolution.evaluator import configure_model_policy, configure_trace
    configure_model_policy(
        unified_embedding=((cfg.get("seed") or {}).get("unified_models") or {}).get("embedding"),
        routes=(cfg.get("llm") or {}).get("routes") or {},
    )
    configure_trace(max_value_chars=(cfg.get("trace") or {}).get("max_value_chars"))
    cfg_snapshot = _write_resolved_config(cfg)
    log.info(
        f"run config: {json.dumps(cfg, default=str)}"
    )
    log.info(f"workspace: {paths.workspace}")
    log.info(f"run log: {run_log}")
    log.info(f"config snapshot: {cfg_snapshot}")
    try:
        asyncio.run(search_loop(cfg))
    except ProposerInfraError as exc:
        # search_loop already logged the actionable message. Exit with a
        # distinct code so wrapper scripts can detect "abort due to propose
        # infra failure" (rc=2) versus other errors (rc=1, generic crashes).
        log.error(f"search aborted: {exc}")
        sys.exit(2)


if __name__ == "__main__":
    main()
