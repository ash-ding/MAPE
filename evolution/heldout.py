"""Dedicated held-out test entry point — evaluate specific harness(es) on the
test split, WITHOUT the search loop's proposer / sanity gate / frontier.

    # Config-first (see configs/evaluation/example.yaml for the annotated schema —
    # `harnesses:` lists the target dirs right in the YAML):
    uv run python -m evolution.heldout --config configs/evaluation/example.yaml

    # CLI overrides: --harness (repeatable) REPLACES the YAML `harnesses:`
    # list; every orchestrator flag (--datasets, ...) works too.
    uv run python -m evolution.heldout --config configs/evaluation/example.yaml \\
        --harness workspace/<run>/harnesses/9f00aa11b2c3

Flow per harness:
  1. Copy the harness dir into this run's workspace (source stays untouched;
     per-run isolation, same principle as the search loop).
  2. ensure_image (per-harness delta resolved from its requirements.txt).
  3. evaluate_harness(split="test") — ONE pass per benchmark, sized by each
     dataset's `size` block (null fields = the whole test split).

Outputs under workspace/<run_name>/:
  heldout_results.json   {harness_id: {objectives (accuracy_<ds>,
                          score_max_<ds>, and cost_usd_per_1k_queries_<ds>
                          when the run records it), per_ds}}
  harnesses/<id>/<ds>/...  the pass's artifacts, written straight into the
                          dataset directory
  config.yaml, orchestrator.log — same bookkeeping as any run.

The datasets / model / judge_model / llm-transport / gpu configuration is
inherited from --config (plus the usual CLI overrides); search-loop-only
settings (steps, k_per_step, sanity, seed, proposer.*) are ignored.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import json
import logging
import shutil
import sys

import yaml
from pathlib import Path
from typing import Any, Dict, List

from evolution.env_builder import EnvBuildError, ensure_image
from evolution.orchestrator import (
    HELDOUT_REQUIRED_SCHEMA,
    KNOWN_METRICS,
    _attach_run_log,
    _build_objectives,
    _resolve_config,
    _setup_logging,
    _write_resolved_config,
    build_arg_parser,
    evaluate_harness,
)
from evolution.paths import paths, ENTRY_FILE, entry_file

log = logging.getLogger("evolution.heldout")


def _heldout_arg_parser() -> argparse.ArgumentParser:
    """The orchestrator's parser (so every config override keeps working)
    plus the heldout-specific --harness flag and a single-stage default."""
    parser = build_arg_parser()
    parser.add_argument(
        "--harness", action="append", dest="harnesses", metavar="DIR",
        help="Path to a harness directory (must contain memo.py). Repeat "
             "for several. Overrides (replaces) the config's `harnesses:` list.",
    )
    return parser


def _stage_harness(src: Path) -> str:
    """Copy a source harness dir into this run's workspace; return its id."""
    src = src.resolve()
    if entry_file(src) is None:
        raise SystemExit(
            f"--harness {src}: no {ENTRY_FILE} found — pass a harness directory "
            f"(e.g. workspace/<run>/harnesses/<hash>)"
        )
    dst = paths.harnesses_dir / src.name
    if dst.exists():
        raise SystemExit(
            f"duplicate harness id {src.name!r} in this heldout run — "
            f"pass directories with distinct names"
        )
    shutil.copytree(src, dst)
    return src.name


async def _run(cfg: Dict[str, Any], harness_paths: List[str]) -> None:
    paths.harnesses_dir.mkdir(parents=True, exist_ok=True)
    ids = [_stage_harness(Path(p)) for p in harness_paths]

    results: Dict[str, Any] = {}
    for hid in ids:
        harness_dir = paths.harnesses_dir / hid
        try:
            image_path = await ensure_image(harness_dir)
        except EnvBuildError as exc:
            log.error(f"heldout: image build failed for {hid}: {exc}")
            results[hid] = {"error": f"image build failed: {exc}"}
            continue

        log.info(f"heldout: evaluating {hid} (datasets={list(cfg['datasets'])})")
        per_ds = await evaluate_harness(
            hid, image_path,
            datasets_config=cfg["datasets"],
            split="test",
            model=cfg["model"], judge_model=cfg["judge_model"],
            max_sample_concurrent=cfg["max_sample_concurrent"],
            gpu=cfg["gpu"]["enabled"],
            compose_enabled=bool(((cfg.get("contract") or {}).get("compose", True))),
        )
        results[hid] = {
            # Held-out numbers are a REPORT, not a search scoreboard: record
            # every axis regardless of what the search was told to optimize.
            "objectives": _build_objectives(per_ds, list(KNOWN_METRICS)),
            "per_ds": per_ds,
        }

    out_path = paths.workspace / "heldout_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    log.info(f"heldout: results written to {out_path}")
    for hid, r in results.items():
        if "error" in r:
            log.info(f"  {hid}: ERROR — {r['error']}")
            continue
        obj = r["objectives"]
        per_ds_str = "  ".join(
            f"{k[len('accuracy_'):]}={v:.3f}" for k, v in sorted(obj.items())
            if k.startswith("accuracy_")
        )
        # Token totals left the frontier (see _build_objectives); they live
        # in each dataset's metrics.json, which this result already carries.
        tokens = sum(int((m or {}).get("tokens", 0) or 0)
                     for m in (r.get("per_ds") or {}).values())
        log.info(f"  {hid}: {per_ds_str}  tokens={tokens}")


def _yaml_raw(args) -> Dict[str, Any]:
    """The --config YAML as written (pre-merge) — for heldout-specific keys
    (`harnesses:`)."""
    if not getattr(args, "config", None):
        return {}
    with open(args.config) as f:
        return yaml.safe_load(f) or {}






def _resolve_harnesses(args, raw_yaml: Dict[str, Any]) -> List[str]:
    """CLI --harness (repeatable) REPLACES the YAML `harnesses:` list."""
    if getattr(args, "harnesses", None):
        return list(args.harnesses)
    yaml_harnesses = raw_yaml.get("harnesses") or []
    if not isinstance(yaml_harnesses, list) or not yaml_harnesses:
        raise SystemExit(
            "no harnesses to evaluate — pass --harness <dir> (repeatable) or "
            "set a `harnesses:` list in the config "
            "(see configs/evaluation/example.yaml)"
        )
    return [str(h) for h in yaml_harnesses]


def _heldout_run_name(cfg: Dict[str, Any]) -> str:
    """The workspace name for a held-out run: always prefixed `heldout`.

    This prefix is what keeps test-split results out of a search workspace, and
    it is load-bearing rather than cosmetic. A search run binds
    `workspace/<its run_id>` into the proposer container as `/workspace`, RW, and
    the proposer may read anything under it — including every harness's
    `<dataset>/traces/*.json`, which carry `reference` (the gold answer),
    `relevant_turns` (the gold evidence text) and `qa_metadata.evidence`. In a
    search run those traces cover the SEARCH split, which the proposer is meant
    to learn from. A held-out run writes the same filenames with the TEST
    split's questions and answers in them. Landing those in a search workspace
    would hand the proposer the held-out answers on the next resume, silently.

    So `--run-name locomo_main_pareto_r1` becomes
    `heldout_locomo_main_pareto_r1`: a different directory, which no proposer
    container binds.
    """
    run_name = str(cfg.get("run_name") or _dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    return run_name if run_name.startswith("heldout") else f"heldout_{run_name}"


def _reject_search_workspace() -> None:
    """Refuse to write held-out results into a search run's workspace.

    `_heldout_run_name`'s prefix already makes the collision hard to reach — it
    takes a SEARCH run that was itself named `heldout...`. This checks the
    invariant the prefix is there to produce, rather than trusting the naming
    convention to keep producing it: `history.json` and `frontier.json` are
    written only by the search loop, so either one means this directory belongs
    to a search run and test-split traces must not go into it.

    A resumed held-out run is fine and must stay fine: its workspace holds
    `heldout_results.json` and harness directories, neither of which is checked.

    Must be called after `paths.set_run_id` and before anything is written.
    """
    for marker in ("history.json", "frontier.json"):
        if (paths.workspace / marker).is_file():
            log.warning(
                f"evolution.heldout: refusing to run — {paths.workspace} is a SEARCH "
                f"workspace (it has {marker}), and held-out evaluation writes "
                f"test-split traces.\n"
                f"A search run binds its own workspace into the proposer "
                f"container, so test-split questions and gold answers written "
                f"here would reach the proposer on the next resume of that run — "
                f"silently, with no failure and no warning.\n"
                f"Fix: give this held-out run a name of its own "
                f"(--run-name p1_heldout). One held-out run can take many "
                f"--harness flags and keys them all in one heldout_results.json, "
                f"so a separate workspace per harness is not needed."
            )
            sys.exit(2)


def main() -> None:
    args = _heldout_arg_parser().parse_args()
    _setup_logging(getattr(args, "verbose", False))

    cfg = _resolve_config(args, required_schema=HELDOUT_REQUIRED_SCHEMA)
    from evolution.evaluator import configure_model_policy, configure_trace
    configure_model_policy(
        unified_embedding=((cfg.get("seed") or {}).get("unified_models") or {}).get("embedding"),
        routes=(cfg.get("llm") or {}).get("routes") or {},
    )
    configure_trace(max_value_chars=(cfg.get("trace") or {}).get("max_value_chars"))
    raw_yaml = _yaml_raw(args)
    harnesses = _resolve_harnesses(args, raw_yaml)

    run_name = _heldout_run_name(cfg)
    paths.set_run_id(run_name)
    _reject_search_workspace()
    paths.workspace.mkdir(parents=True, exist_ok=True)
    _attach_run_log()
    _write_resolved_config(cfg)
    log.info(f"heldout run: {paths.workspace} (test split)")

    asyncio.run(_run(cfg, harnesses))


if __name__ == "__main__":
    main()
