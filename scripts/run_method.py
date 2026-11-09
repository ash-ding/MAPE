"""Evaluate a memory program in this process, without containers.

Examples (from the repository root, after `uv sync` and `scripts/fetch_data.py`):

    # MAPE-A_LME on 20 held-out LongMemEval questions with gpt-4.1-mini
    uv run python scripts/run_method.py --benchmark longmemeval --n 20

    # the same program transferred to LoCoMo: 1 held-out conversation, 40 questions
    uv run python scripts/run_method.py --benchmark locomo --conversations 1 --n 40

    # a locally served backbone (any OpenAI-compatible endpoint)
    uv run python scripts/run_method.py --benchmark longmemeval --n 20 \
        --model qwen3-30b-a3b-instruct-2507 --base-url http://127.0.0.1:8000/v1

    # your own program: a directory holding memo.py (and optionally src/)
    uv run python scripts/run_method.py --harness path/to/my_memory --benchmark longmemeval --n 20

`--model` is the serving backbone: the QA model and the model behind every LLM
call the memory program makes. The judge (`--judge-model`, default gpt-5-mini)
and the embedding model (`--embedding-model`, default text-embedding-3-small)
use the OpenAI API, so OPENAI_API_KEY is required even with a local backbone.

This is the same evaluation the containerised runners perform
(`core.evaluate.evaluate_memo`), run in the current Python environment: the
program's dependencies must be importable here, and nothing isolates its code
from the host. Use `evolution.heldout` for full-split, isolated evaluation.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BENCHMARKS = {"longmemeval": "longmemeval_s", "locomo": "locomo"}
DEFAULT_HARNESS = ROOT / "programs" / "mape_a"


def _assemble_harness(src: Path, benchmark: str, dest: Path) -> Path:
    """Copy the program into `dest` the way the paper experiments stage it.

    programs/mape_a reads LongMemEval histories directly; on LoCoMo it runs
    through the fixed input adapter (experiments/transfer/locomo.py), as in
    the paper's transfer evaluation. Any other directory is copied as is.
    """
    if dest.exists():
        shutil.rmtree(dest)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    if src.resolve() == DEFAULT_HARNESS.resolve():
        dest.mkdir(parents=True)
        shutil.copytree(src / "src", dest / "src", ignore=ignore)
        if benchmark == "locomo":
            shutil.copyfile(src / "memo.py", dest / "memo_lme.py")
            shutil.copyfile(ROOT / "experiments/transfer/locomo.py", dest / "memo.py")
            shutil.copyfile(src / "adapters/locomo.py", dest / "locomo_adapter.py")
        else:
            shutil.copyfile(src / "memo.py", dest / "memo.py")
    else:
        if not (src / "memo.py").exists():
            raise SystemExit(f"{src} has no memo.py (the file that defines the MemoClass subclass)")
        shutil.copytree(src, dest, ignore=ignore)
    return dest


def _size(args: argparse.Namespace) -> dict:
    if args.benchmark == "longmemeval":
        return {"n_questions": args.n}
    return {"n_conversations": args.conversations, "n_qa": args.n}


async def _run(args: argparse.Namespace) -> dict:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    harness_dir = _assemble_harness(Path(args.harness).resolve(), args.benchmark, out_dir / "harness")

    # The run's model policy, as the containerised runners set it.
    from core import model_policy
    os.environ.update(model_policy.env_for_container(
        llm=args.model, judge=args.judge_model, embedding=args.embedding_model))
    if args.base_url:
        os.environ["MAPE_MODEL_ROUTES"] = json.dumps(
            {args.model: {"base_urls": [args.base_url], "api_key": args.api_key}})

    from core.openai_usage import install as install_usage_tracking
    install_usage_tracking()
    from core import trace
    trace.install(out_dir)
    from core.evaluate import evaluate_memo
    from evolution.launch import _load_harness_class

    memo_class = _load_harness_class(harness_dir)
    trace.instrument(harness_dir)   # record the harness's own function calls, as the container does
    t0 = time.time()
    metrics = await evaluate_memo(
        memo_class=memo_class,
        dataset=BENCHMARKS[args.benchmark],
        split=args.split,
        out_dir=out_dir,
        qa_model=args.model,
        judge_model=args.judge_model,
        size=_size(args),
        max_sample_concurrent=args.max_concurrent,
        sample_seed=args.seed,
        memo_sha=harness_dir.parent.name,
    )
    metrics["model_policy_violations"] = model_policy.violations()
    metrics["wall_seconds"] = round(time.time() - t0, 1)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    trace.close()
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--harness", default=str(DEFAULT_HARNESS),
                   help="directory holding memo.py (default: programs/mape_a, the released program)")
    p.add_argument("--benchmark", choices=sorted(BENCHMARKS), required=True)
    p.add_argument("--split", choices=["test", "search"], default="test",
                   help="held-out test split (default) or the search split")
    p.add_argument("--n", type=int, default=20,
                   help="LongMemEval: number of questions; LoCoMo: questions per conversation")
    p.add_argument("--conversations", type=int, default=1, help="LoCoMo: number of conversations")
    p.add_argument("--model", default="gpt-4.1-mini", help="serving backbone (QA model and memory-side calls)")
    p.add_argument("--judge-model", default="gpt-5-mini")
    p.add_argument("--embedding-model", default="text-embedding-3-small")
    p.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint serving --model")
    p.add_argument("--api-key", default="EMPTY", help="API key for --base-url")
    p.add_argument("--max-concurrent", type=int, default=4, help="histories evaluated concurrently")
    p.add_argument("--seed", default=None, help="sampling seed for the question subset")
    p.add_argument("--out", default=None, help="output directory (default: workspace/runs/<benchmark>_<timestamp>)")
    args = p.parse_args()
    if args.out is None:
        args.out = str(ROOT / "workspace" / "runs" / f"{args.benchmark}_{time.strftime('%Y%m%d_%H%M%S')}")

    m = asyncio.run(_run(args))
    n_q = m.get("cost_n_queries") or m.get("context_tokens_n_queries")
    print(f"\naccuracy        {100 * float(m.get('raw_score', 0.0)):.2f}%   ({n_q} questions, {args.benchmark} {args.split} split)")
    print(f"serving cost    ${float(m.get('cost_usd_per_1k_queries', 0.0)):.2f} per 1,000 queries"
          f"  (build {m.get('cost_usd_build_per_1k_queries', 0):.2f}, retrieve {m.get('cost_usd_retrieve_per_1k_queries', 0):.2f},"
          f" compose {m.get('cost_usd_compose_per_1k_queries', 0):.2f}, answer {m.get('cost_usd_answer_per_1k_queries', 0):.2f})")
    print(f"serving calls   {m.get('llm_calls')}   wall time {m.get('wall_seconds')} s")
    if m.get("model_policy_violations"):
        print(f"model policy violations: {m['model_policy_violations']}")
    print(f"artifacts       {args.out}/  (score.json, traces/, metrics.json)")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # skip interpreter teardown of lingering client threads


if __name__ == "__main__":
    main()
