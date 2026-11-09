# mem0 — LLM-extracted fact memory

[Mem0](https://github.com/mem0ai/mem0) as a `MemoClass`, scored through the
shared registry/workflow/judge path like every other baseline. The baseline
vendors and drives `mem0` directly.

**Provenance**: `src/mem0/` is vendored VERBATIM (byte-identical) from
<https://github.com/mem0ai/mem0> @ `12c47f524935692e27ad48d829f35fa1e4417181`
(tag `v2.0.17`, i.e. `mem0ai==2.0.17`), pruned to the
exercised slice — provenance lives here, not in per-file headers, to preserve
byte-identity:

    git -C <mem0-clone> archive 12c47f5 mem0 | tar -x -C /tmp/m0
    diff -r /tmp/m0/mem0 src/mem0        # only src/mem0/__init__.py differs (see below)

**Excluded** (upstream carries a very large provider surface; the baseline
exercises one LLM, one embedder, one vector store):

| excluded | why |
|---|---|
| `client/`, `proxy/`, `reranker/` | hosted-platform HTTP client, OpenAI-proxy shim, rerankers — none used |
| `llms/` — 17 of 18 providers | only `llms/openai.py` is configured (`provider: openai`) |
| `embeddings/` — 10 of 11 providers | only `embeddings/openai.py`; `mock.py` is kept because `utils/factory.py` imports it eagerly |
| `vector_stores/` + `configs/vector_stores/` — 24 of 25 each | only `qdrant.py` (embedded, on-disk, per user) |
| `utils/gcp_auth.py` | Vertex/GCP auth, unreachable from the exercised path |

Everything kept is on the eager import closure of `from mem0 import Memory` plus
those three factory-selected providers; the factory loads providers by dotted
string, so an excluded provider is a config that raises rather than a silent
fallback. `mem0/utils/` has no `__init__.py` upstream (namespace package) and is
vendored that way.

**The one file that is NOT byte-identical** is `src/mem0/__init__.py`: upstream
reads its version from installed `mem0ai` metadata
(`importlib.metadata.version("mem0ai")`), which raises `PackageNotFoundError` for
a vendored copy, and eagerly imports the un-vendored `client/` subtree. It is
replaced with a minimal initializer that inlines `__version__ = "2.0.17"` (read
by `memory/telemetry.py`) and re-exports only `Memory`/`AsyncMemory`. The reason
is restated in the file itself. All other integration code lives in `memo.py` /
`eval_harness.py`, never in `src/`.

## What the method is

Mem0's contribution is at WRITE time. `Memory.add(messages)` sends a batch of
messages to an LLM that (a) extracts standalone facts from them and (b) decides,
per fact and against what is already stored, whether to **ADD** it, **UPDATE** an
existing fact, or **DELETE** one it contradicts. Memory therefore holds distilled,
de-conflicted facts — "User adopted a rescue dog named Pico, a beagle mix" — not
raw turns. Read time is a plain vector search over those facts.

That write-side consolidation is the whole point, so `infer: true` is the method;
`infer: false` degrades Mem0 to a vector store and is only meaningful as an
ablation.

## Integration

| hook | what runs |
|---|---|
| `build_memory_from_data` | ingestion units → chat messages → `Memory.add(..., user_id)` in batches of `add_batch_size` |
| `retrieve_memory_for_query` | `Memory.search(query, filters={user_id}, top_k, threshold)` → `{"passages": [...]}` |

Answering is not a hook: the SHARED QA agent answers from the
retrieved facts, so the comparison is about
memory rather than about each method's own answerer.

Each user gets its own Mem0 instance with its own on-disk Qdrant collection and
history DB under `outputs/<instance>/`, so nothing leaks across conversations.

**Ingestion units.** locomo: one message per turn, with speaker and session date
folded INTO the content — Mem0's extractor only sees `role`/`content`, so
dropping them would make every "who said it / when" question unanswerable.
longmemeval: one message per message, roles preserved.

## Model configuration (two arms)

Every model this baseline touches is a config parameter, so it runs in two arms:

| | faithful arm (`arm: faithful` — `CONFIG_DEFAULTS`) | unified arm (`arm: unified` — example `unified_models`) |
|---|---|---|
| internal LLM (`mem0_llm_model`) | `gpt-4o-mini` — Mem0's own default | `gpt-4.1-mini` |
| embedder (`embedding_model`) | `text-embedding-3-small`, API, 1536-dim | **unchanged** |

Mem0 is the one baseline already on an API embedder,
so its embedder does not change between arms and no adapter is involved. **The
faithful arm is the default**, and `config.paper.yaml` additionally pins the
paper's gpt-4o-mini answerer. The unified arm puts all seven baselines on one
LLM and one embedder so the comparison against the main method is like-for-like
— a deliberate deviation from the paper, whose numbers must not be quoted as
Mem0's published result.

Mem0's OpenAI provider sends `temperature` and `max_tokens`, which the gpt-5
family rejects; the shim in [`../model_config.py`](../model_config.py) drops the
rejected params and renames `max_tokens` → `max_completion_tokens` at the
OpenAI-SDK boundary. It acts at the SDK boundary precisely so the
vendored `src/mem0/` tree stays byte-identical — wrap, don't rewrite.

## Dependencies

`mem0ai` is NOT a dependency — the source is vendored under `src/` (see
Provenance), so `uv sync` alone is the whole setup. `pyproject.toml` carries only
what the vendored slice imports: qdrant-client, pydantic, posthog (eagerly
imported by `memory/telemetry.py` even with telemetry off, which `memo.py` sets
explicitly on the module — not through the `MEM0_TELEMETRY` env var), plus openai/httpx from the shared-core block. Upstream's
sqlalchemy / pytz / protobuf / spacy are not imported by the slice (protobuf
arrives transitively via qdrant-client; spaCy only via the optional BM25
lemmatization path, which falls back to the raw text when it is absent).

The exercised path is Mem0's default stack — OpenAI LLM + OpenAI embedder +
embedded Qdrant, no server.

## Faithfulness boundary

| Category | Items |
|---|---|
| Verbatim | the whole vendored slice (@ v2.0.17) except `__init__.py`: the ADD/UPDATE/DELETE fact-consolidation pipeline, extraction + update prompts (`configs/prompts.py`), scoring/BM25 path, Qdrant store, OpenAI LLM + embedder clients |
| Integration adaptations (not algorithm) | `src/mem0/__init__.py` replaced (metadata version lookup + un-vendored `client/` import — see Provenance); embedded on-disk Qdrant + per-user instance for isolation rather than a shared server; `MEM0_TELEMETRY=False` set before import (upstream's telemetry store takes a process-global Qdrant lock that breaks concurrent users, and a benchmark should not phone home); batched `add()` (`add_batch_size`, Mem0's intended usage) instead of one call per turn; per-dataset ingestion-unit mappings (locomo folds speaker+date into content, longmemeval preserves roles); answering via the shared QA agent; embedding inputs truncated to 7,800 tokens in `memo.py` (after `embed()`'s own `\n`→space step) — `Memory.add` embeds each joined `add_batch_size` batch as its existing-memory search query, and 65 batches in the LongMemEval-S test split exceed OpenAI's 8,192-token limit (fact extraction still reads the full batch; only that search query is shortened) |
| Upstream quirks preserved | all extraction/update prompts, thresholds and the `infer=True` default path untouched; excluded providers fail loudly (factory loads by dotted string) rather than falling back |

## Run

```bash
# build mem0's isolated env
cd baselines/mem0 && uv sync

# run it (from the repo root)
uv run --project baselines/mem0 python -m baselines.eval_harness \
        --config baselines/config.example.yaml      # harness: mem0
```
