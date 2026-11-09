# Reproduction guide

Run all commands from the repository root. The default suite is offline after benchmark download; full evaluation and search use model APIs or a local model server.

## Data and splits

```bash
uv run python scripts/fetch_data.py
uv run python scripts/fetch_data.py --check
```

The downloader verifies LoCoMo `locomo10.json` and LongMemEval-S `longmemeval_s_cleaned.json` against pinned checksums. LongMemEval uses the upstream **cleaned** release, not the similarly named original release. The files remain local and are excluded from Git.

`benchmarks/locomo/env.py` defines the 2/8 conversation search/evaluation split. `benchmarks/longmemeval/env.py` defines the stratified 100/400 question split with seed `longmemeval.search.v1`. `core/sampling.py` controls sampling within splits. Search data isolation is implemented in `evolution/data_isolation.py`.

## Model and container setup

Copy `.env.example` to `.env` and fill the required credentials. Configure `llm.routes` in the selected YAML for local OpenAI-compatible endpoints. The supplied case-study configs expect `qwen3-30b-a3b-instruct-2507` at `http://127.0.0.1:8000/v1`, with `gpt-5-mini` as judge and `text-embedding-3-small` as embedder. API credentials are still needed for the judge and embeddings.

Optional vLLM serving instructions are in `scripts/vllm/README.md`. A local serving URL is machine-specific; adjust it before running. Token cost estimates use `core/pricing.py`, not measured GPU rental costs.

The evolution and held-out runners use Linux and a `singularity` executable. Build the images from the repository root:

```bash
mkdir -p .cache/images
singularity build .cache/images/eval-base.sif containers/eval-base.def
singularity build .cache/images/proposer-base.sif containers/proposer-base.def
```

The second definition inherits `.cache/images/eval-base.sif`. If using a custom image directory, adjust that `From:` path and set `MAPE_IMAGES_DIR` to its absolute location. Image builds install substantial dependencies and require a working unprivileged container-build setup. GPU-backed baselines also require compatible NVIDIA drivers.

Search additionally requires the Claude Code CLI. The proposer and the diagnosis agent call it with `ANTHROPIC_API_KEY` from the environment or `.env`; no other authentication route is supported. Their settings are under `proposer` and `analyst`. The shipped search configs use `xhigh`, reflecting the saved configurations. Do not interpret archived model names as guarantees that those models remain available to every account.

## Quick in-process evaluation

`scripts/run_method.py` evaluates one memory program in the current Python environment without Singularity, on a subset of a split, and prints accuracy and serving cost. It is the fastest way to check a setup or a new program; the paper's numbers come from the container-based runs described below.

```bash
uv run python scripts/run_method.py --benchmark longmemeval --n 20
uv run python scripts/run_method.py --benchmark locomo --conversations 1 --n 40
```

## Case-study evaluation and experiments

```bash
uv run python scripts/prepare_experiments.py
```

This creates the following under `workspace/paper/`:

| Directories | Purpose | Additional inputs |
|---|---|---|
| `longmemeval_method`, `locomo_method` | Full case-study program and benchmark transfer | Benchmark data and model access |
| `<benchmark>_full` | Cache-producing full ablation arm | Run before the other raw-round arms |
| `<benchmark>_noraw`, `<benchmark>_noraw_nocompose` | Raw-round removal, with/without RQS | The full arm's memory cache |
| `heldout_*_nocompose` | RQS removal with fixed retrieval | `replay.json` from full-program traces |
| `heldout_lme_qwen_control_<method>` | Fixed baseline retrieval, no RQS | Baseline `replay.json` |
| `heldout_lme_qwen_transplant_<method>` | Same retrieval with transplanted RQS | The same baseline `replay.json` |

Each directory has `config.yaml` and `harness/`. Run a prepared configuration with:

```bash
uv run python -m evolution.heldout --config workspace/paper/longmemeval_full/config.yaml
uv run python -m evolution.heldout --config workspace/paper/longmemeval_noraw/config.yaml
```

The original raw-round wrappers use `/hf-cache/mape_ablation_7723a2` for LongMemEval and `/hf-cache/mape_ablation_7723a2_locomo` for LoCoMo. The evaluator binds host `HF_HOME`, or `~/.cache/huggingface`, to `/hf-cache`. Keep this cache consistent across arms. The LongMemEval wrapper also recognizes `MAPE_ABL_CACHE` inside the container.

For replay experiments, build a retrieval table from your locally generated traces:

```bash
uv run python scripts/build_replay.py \
  --traces /path/to/full-run/traces \
  --benchmark longmemeval \
  --output workspace/paper/heldout_lme_qwen_acc_7723a2_nocompose/harness/replay.json
```

For LoCoMo, use `--benchmark locomo --locomo-data benchmarks/locomo/locomo10.json`. For transplantation, also supply `--baseline mem0` or the appropriate method, and put identical replay files in the paired control and transplantation harness directories. Replay keys preserve query dates, speaker identity and repeated-query order. Missing retrievals raise an error.

No results, retrieval traces or caches are bundled: every number is produced by running the experiments, and model non-determinism means a rerun lands near the paper's figures rather than on them. Replay runs perform no memory construction or retrieval model calls, so their reported costs alone cannot represent the full lifecycle: include the source run's construction and retrieval costs when comparing total serving cost.

## Baselines

The runner is `baselines/eval_harness.py`. `icl` is full-context prompting; `rag` retrieves raw messages. The five memory systems are `mem0`, `amem`, `lightmem`, `simplemem` and `zep`. `no_memory` supplies the minimal seed/control.

RAG and the five memory systems each have their own `pyproject.toml`, lockfile and README. Their shared configuration example is `baselines/config.example.yaml`. Full-context and no-memory harnesses use the root environment. Install each memory baseline environment separately to avoid incompatible model-library versions. For example:

```bash
uv sync --project baselines/rag
```

The adapters implement the common interface around preserved upstream source. Consult each method's README for the exact upstream revision and integration choices.

## Source identity and prompts

The released final program is MAPE-A_LME, originally identified as `7723a2cc870a`. Its code keeps the class name `AtomicLedgerMemo` from the search; only package imports were relocated and comments renamed. `experiments/provenance.json` records source paths and hashes.

`evolution/prompts/parts/` holds the single prompt set used by the search configurations: the proposer, task, fix and diagnosis prompts of the paper's runs, with module paths updated to this repository's layout. Beyond those path changes, the proposer system prompt differs from the one the paper's runs used in three passages, each of which named a concrete technique that the released program happens to use. They were rewritten to be design-neutral for this release:

1. The `meta.json` example of a good `design_choices.how` entry read: *"Hybrid BM25 + dense fusion via RRF in `_retrieve_pool`. BM25 alone misses paraphrased queries; dense alone misses low-frequency exact facts. Fusion catches both. Chosen after harness 3's traces showed synthesis queries scoring 0 with dense-only retrieval."* (bad example: *"BM25 + dense fusion."*). The example now describes a session-level chunking choice.
2. The Phase 1 description continued: *"Read-side code SHOULD still: maintain precomputed indices (BM25 corpus, dense embeddings, inverted indices) — built incrementally during ingest, or lazily on first retrieve, then cached on `self` and reused; look up write-time-built abstractions for generalization queries. Anti-pattern to avoid: don't defer ALL abstraction to first retrieve, computing it over the retrieved subset and discarding afterward. It puts the cost on the read path, which has a tighter budget than the write path. Build abstractions incrementally during your build calls; store them persistently on `self`; reuse them across queries."* The Phase 2 read-only rule also listed *"(BM25 corpus, embeddings)"* as examples of lazily built indices. Both now state only that build-time work runs once per user and retrieve-time work once per query.
3. The compose-hook description listed *"(re-ranking, compressing, a second retrieval pass)"* as examples of model calls made inside the hook, and ended: *"A design that wants question-type-specific behaviour has to infer the type from the query (and may be wrong): that is the honest version of the idea, and the one that transfers."* Both sentences were removed; the hook description now ends at the statement that benchmark labels are not readable at answer time.

The task prompt's search-strategy guidance (prefer an exploratory redesign when the frontier looks saturated) and the list of packages preinstalled in the evaluation image are unchanged.

## Validation scope

```bash
uv sync --group dev
uv run python scripts/fetch_data.py
uv run pytest -q
```

The tests check release provenance hashes, strict configuration, search/test data isolation, benchmark label hiding, token and cost accounting, prompt assembly, the memory interface and baseline packaging. They make no model calls and run no containers. Passing them does not establish that a fresh API/GPU run reproduces the historical scores. Full model experiments and image builds are separate steps.
