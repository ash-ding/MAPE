# Third-party provenance

The release retains upstream baseline implementations and implements the lifecycle adapters separately. Each baseline README records its source revision and any integration-specific changes. Moving directories does not relicense the bundled source.

| Integration | Upstream | Recorded revision | License |
|---|---|---|---|
| A-Mem | https://github.com/WujiangXu/A-mem | `0c8039f28fdcc08189a23c07a3437d9d2482f9c2` | MIT |
| LightMem | https://github.com/zjunlp/LightMem | `34410f41173f9107fbe8daf273092f398bfaf85b` | MIT |
| SimpleMem | https://github.com/aiming-lab/SimpleMem | `db80b6a7c591e0ea730a058e9f5fc4eb06572299` | MIT |
| Mem0 | https://github.com/mem0ai/mem0 | `12c47f524935692e27ad48d829f35fa1e4417181` | Apache-2.0 |
| Zep / Graphiti | https://github.com/getzep/graphiti | `4f62cfe7a2d519e55bfdf2dc4a2fd06649dc00b3` | Apache-2.0 |

Baseline dependency versions are pinned in their individual lockfiles. Full-context and RAG implementations are in `baselines/icl/memo.py` and `baselines/rag/`; the latter has its own README.

Data is downloaded separately from [LoCoMo](https://github.com/snap-research/locomo) and [LongMemEval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned). The download script includes exact URLs and content hashes. Dataset terms remain those of their original releases.

Project-owned code is released under the Apache License 2.0 (`LICENSE`). Each bundled upstream copy keeps its own license text in `baselines/<name>/src/LICENSE`, and `NOTICE` lists them; the root license does not replace those terms.

The preserved LightMem revision contains an unfinished optional `memory/graph.py` stub. Its graph-memory branch is not used by the supplied text-memory integration. The upstream file is retained unchanged; enabling that optional branch is unsupported in this release.
