"""Evolution in-container runner.

Invoked by evolution/evaluator.py inside Singularity — ONE container per
(harness, dataset) runs the WHOLE evaluation plan:

    python /app/evolution/launch.py --harness-dir /harness --out-dir /out \
        --dataset locomo --split search --plan-json '{"smoke": false, "size": {...}, ...}'

Steps:
  1. Dynamically load the MemoClass subclass from /harness/memo.py.
  2. One call into the shared, execution-independent
     core.evaluate.evaluate_memo — the pass sized by `size`, or the
     sanity-size smoke run, inside THIS container.
  3. evaluate_memo writes /out/{score.json,token_usage.json,run_record.json,traces/};
     this runner additionally writes /out/metrics.json (evaluate_memo's
     return dict) for the host to read back.

The harness_dir is bind-mounted read-only; /out is bind-mounted read-write.
Binds are SELECTIVE (v10): only core/, datasets/, evolution/{__init__,launch,
memo_class}.py are mounted under /app — the rest of evolution/ (host outer
loop) and baselines/ are deliberately NOT visible in-container. See
evolution/evaluator.py's module docstring for the authoritative bind list.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import inspect
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Type

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from core.memo_class import MemoClass
from benchmarks.registry import DATASETS


#: Kept in step with evolution.paths — the container gets only launch.py,
#: memo_class.py and __init__.py from evolution, so it cannot import that module.
ENTRY_FILE = "memo.py"


def _load_harness_class(harness_dir: Path) -> Type[MemoClass]:
    """Import the harness's interface file and return its MemoClass subclass.

    On import failure, raise ImportError with an actionable message. The
    error string is propagated up to score.json::invalid_users[0].error and
    becomes the trace shown to CC during sanity-retry — so write it for an
    LLM reader.
    """
    harness_py = harness_dir / ENTRY_FILE
    if not harness_py.exists():
        raise ImportError(
            f"{ENTRY_FILE} missing at {harness_dir / ENTRY_FILE}. The proposer "
            f"must write a {ENTRY_FILE} defining its MemoClass subclass; any "
            f"further source goes under {harness_dir.name}/src/."
        )
    if str(harness_dir) not in sys.path:
        sys.path.insert(0, str(harness_dir))
    spec = importlib.util.spec_from_file_location("harness_mod", str(harness_py))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for {harness_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["harness_mod"] = module
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"{harness_py.name} imports a package not in the container: {exc.name!r}. "
            f"Either (a) declare it in `requirements.txt` (will trigger a delta "
            f"image build), or (b) switch to a package already in the base image "
            f"(see PROPOSER_SYSTEM for the list). Original: {exc}"
        ) from exc
    except Exception as exc:
        raise ImportError(
            f"{harness_py.name} raised at import time: [{type(exc).__name__}] {exc}"
        ) from exc
    from core.memo_select import select_memo_class
    # Only classes DEFINED in this harness file (not imported bases), concrete first.
    defined = [obj for _, obj in inspect.getmembers(module, inspect.isclass)
               if obj.__module__ == module.__name__]
    try:
        return select_memo_class(defined, str(harness_py))
    except TypeError as exc:
        raise ImportError(
            f"{exc} Define a class that inherits from `evolution.memo_class.MemoClass`."
        ) from exc


def _reject_compose_override(memo_class) -> None:
    """cfg.contract.compose=false: the contract this run evolves against has
    two hooks. A class that defines the third anyway fails here, before any
    evaluation is paid for; the message is what the fix prompt shows."""
    # The contract modules define the hook's DEFAULT (core.memo_class, and
    # evolution.memo_class re-exports it); only a definition in the harness's own
    # classes is an override. The packaged seeds inherit from
    # core.memo_class.MemoClass directly, so stopping at the evolution loop's base alone
    # would refuse `no_memory` itself.
    _CONTRACT_MODULES = ("core.memo_class", "evolution.memo_class")
    for cls in memo_class.__mro__:
        if cls.__module__ in _CONTRACT_MODULES or cls is object:
            continue
        if "compose_memory_context" in cls.__dict__:
            raise ImportError(
                f"{cls.__name__} defines `compose_memory_context`, but this run's "
                f"contract has two hooks only (contract.compose: false): "
                f"`build_memory_from_data` and `retrieve_memory_for_query`. The "
                f"host renders the benchmark's own prompt around what retrieve "
                f"returns. Delete the method (and any code that only served it); "
                f"put the design into what retrieve stores and returns."
            )


def _violations_so_far() -> list:
    try:
        from core import model_policy
        return model_policy.violations()
    except Exception:
        return []


def _write_error(out_dir: Path, err: str) -> None:
    try:                                   # flush whatever the trace captured
        from core import trace
        trace.close()
    except Exception:
        pass
    """Load-failure artifacts: an error score.json + a matching metrics.json so
    the host always has both to read back."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark_eval_score": {
            "benchmark_overall_eval_score": 0.0,
            "benchmark_overall_eval_standard_deviation": 0.0,
        },
        "per_user": {},
        "invalid_users": [{"user_id": "load_failed", "error": err}],
    }
    with (out_dir / "score.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({
            "raw_score": 0.0, "score_max": 1, "per_user_stddev": None,
            "tokens": 0, "failed": True, "error": err,
            # a crash after a refusal still carries the record
            "model_policy_violations": _violations_so_far(),
        }, f, indent=2, ensure_ascii=False)


async def _async_main(args: argparse.Namespace) -> None:
    from core.evaluate import evaluate_memo
    from core.openai_usage import install as install_usage_tracking

    # Token accounting BEFORE the harness is imported — a vendored module that
    # does `from openai import OpenAI` at import time still lands on the
    # patched resource classes, and, more importantly, a harness that never
    # asks for accounting still gets it.
    #
    # Installed here rather than left to each harness: one that calls the SDK
    # directly and never installs it would report ZERO tokens for every
    # phase, and so top the efficiency axis while scoring normally on
    # accuracy.
    install_usage_tracking()
    from core import model_policy
    from core.logger import get_logger
    get_logger("main").info(model_policy.describe())

    harness_dir = Path(args.harness_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Level 2+ call recording, before the harness is imported so its own
    # import-time LLM calls (if any) are covered. A no-op at level 1.
    from core import trace
    trace.install(out_dir)

    try:
        memo_class = _load_harness_class(harness_dir)
        if args.no_compose_hook:
            _reject_compose_override(memo_class)
    except Exception as exc:
        err = f"[{type(exc).__name__}] {exc}\n{traceback.format_exc()}"
        _write_error(out_dir, err)
        return

    # Wrap the harness's own functions, now that its modules are imported and
    # its classes exist. It never raises — a harness that cannot be
    # instrumented still gets evaluated.
    try:
        trace.instrument(harness_dir)
    except Exception as exc:
        get_logger("main").warning(f"[trace] instrumentation skipped: {exc!r}")

    # The evaluation plan, serialized by the host (evolution/evaluator.py):
    #   smoke: bool              — ONE sanity_check-sized pass (sanity gate / smoke_test)
    #   size: dict|null          — the pass's sizes (benchmark-native fields)
    #   sanity_check: dict|null  — the smoke pass's sizes
    plan = json.loads(args.plan_json)

    try:
        metrics = await evaluate_memo(
            memo_class=memo_class,
            dataset=args.dataset,
            split=args.split,
            out_dir=out_dir,
            qa_model=args.model,
            judge_model=args.judge_model,
            size=plan.get("size"),
            sanity_check=plan.get("sanity_check"),
            max_sample_concurrent=args.max_sample_concurrent,
            smoke=bool(plan.get("smoke", False)),
            max_logs=args.max_logs,
            memo_sha=harness_dir.name,
        )
    except Exception as exc:
        # Config/plan errors (e.g. a missing size block that slipped
        # past the host) or unexpected evaluator failures: leave readable
        # artifacts rather than a bare traceback + empty /out.
        err = f"[{type(exc).__name__}] {exc}\n{traceback.format_exc()}"
        _write_error(out_dir, err)
        return

    # Every model-policy refusal this container made. Non-empty means the
    # harness tried a model outside the run's allowlist; whether or not it
    # caught the exception, the host fails it on this record.
    metrics["model_policy_violations"] = model_policy.violations()
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    trace.close()

    # chroma / httpx background threads can delay normal interpreter
    # shutdown. Artifacts are flushed; exit abruptly.
    os._exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset", default="locomo", choices=sorted(DATASETS))
    parser.add_argument("--split", default="search", choices=["search", "test"])
    parser.add_argument("--plan-json", required=True,
                        help='JSON evaluation plan, e.g. \'{"smoke": false, '
                             '"size": {...}, "sanity_check": null}\'. '
                             'The pass runs inside THIS container via evaluate_memo.')
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--judge-model", default="gpt-5-mini")
    parser.add_argument("--max-logs", type=int, default=None)
    parser.add_argument("--max-sample-concurrent", type=int, default=3)
    parser.add_argument("--no-compose-hook", action="store_true",
                        help="cfg.contract.compose=false: refuse a harness that defines "
                             "compose_memory_context (two-hook contract).")
    args = parser.parse_args()
    asyncio.run(_async_main(args))
    os._exit(0)
