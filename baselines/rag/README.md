# rag — RAG over raw messages

Retrieval-augmented generation in its plainest form, as a `MemoClass`: every
message is embedded as it is and kept; a query retrieves the `top_k` messages
with the highest cosine similarity and the shared QA agent answers from them.
No LLM in build or retrieve, no memory construction, no query rewriting, no
reranking. It is the control every memory system on this repo has to beat:
does building a memory buy anything over retrieving raw turns?

**What it follows.** The method is "Naive RAG" as defined by Gao et al. 2023
(*Retrieval-Augmented Generation for Large Language Models: A Survey*, arXiv
2312.10997), on the retrieve-then-read idea of Lewis et al. 2020 (arXiv
2005.11401). The concrete recipe — ingestion unit, embedder, k — is the
**NaiveRAG baseline of LightMem** (ICLR 2026, arXiv 2510.18866), the
comparison point it reports on LoCoMo and LongMemEval-S with gpt-4o-mini and
Qwen3-30B-A3B-Instruct-2507:

    github.com/zjunlp/LightMem @ 8449d574df6bae1bdf3314a1564da65e2f37e046
    src/lightmem/memory_toolkits/memories/layers/naive_rag.py        the layer
    src/lightmem/memory_toolkits/memory_construction.py              one add per message
    src/lightmem/memory_toolkits/memory_search.py                    --top-k (default 10)
    src/lightmem/memory_toolkits/memories/datasets/{locomo,longmemeval}.py

## Written here, not vendored

Unlike the other harness baselines there is **no `src/`**. Upstream's
`NaiveRAGLayer` is a thin wrapper over mem0 1.0.2 (`envs/rag_requirements.txt`
pins `mem0ai==1.0.2`) — `Memory.add(..., infer=False)` per message, then
`Memory.search(query, limit=k)` — and on that path mem0 does only this:

| upstream (mem0 1.0.2) | what it does | here |
|---|---|---|
| `Memory._add_to_vector_store`, `infer=False` branch | embed each message's content; insert into a Qdrant collection with `Distance.COSINE` | `RAGMemo._build`: embed, keep |
| `HuggingFaceEmbedding.embed` | `SentenceTransformer.encode(text)`, no normalisation | `model_config.get_embedder(...).encode` |
| `OpenAIEmbedding.embed` | `text.replace("\n", " ")`, then the embeddings API | same replacement, then `core.llm.Embedding` |
| `Memory._search_vector_store` | embed the query; Qdrant cosine top-`limit`; threshold `None`, no reranker configured | `RAGMemo._retrieve`: exact cosine in numpy, top `top_k`, best first |

Vendoring that stack would buy byte-identity of code this baseline never
exercises, at the price of qdrant-client, a sqlite history DB per user and
mem0's telemetry (whose process-global Qdrant lock breaks concurrent users —
see `../mem0/memo.py`). The repo's own `baselines/mem0/src/` could not
be reused either: it is mem0 **v2.0.17**, whose `_search_vector_store` is a
**hybrid vector + BM25** search, so NaiveRAG run on it would quietly stop being
naive.

## Integration

| hook | what runs |
|---|---|
| `build_memory_from_data` | ingestion units → embed their bare text (one batch per call) → append. Additive across build calls |
| `retrieve_memory_for_query` | embed the query → cosine top-`top_k` → `{"passages": [...]}`, best first. Read-only |

Answering is not a hook: the SHARED QA agent answers from the passages, as for
every memo, so upstream's own QA and judge prompts are unused.

**Ingestion units — one per message, on both benchmarks**, the rule
upstream defines for LoCoMo and LongMemEval.

| benchmark | init key | one unit per… | embedded text | speaker | time |
|---|---|---|---|---|---|
| locomo | `conversation` | turn | the turn's `text` only (the image caption stays out, as upstream keeps it as metadata) | `speaker` | session `date_time` |
| longmemeval_s | `sessions` | message, every role | `content` | `role` | session `date` |

Each hit renders as `Time: <time>\nMemory: <speaker>: <text>`, a missing field
dropped — upstream's layout. See the next section for why the speaker and time
are there.

## The one deliberate deviation: `render_metadata`

Upstream builds each hit as `"Time: <session timestamp>\nMemory: <text>"`. It
stores the timestamp and the speaker as mem0 metadata — but mem0 1.x returns
custom metadata **nested** under `item["metadata"]`, and upstream reads
`item.get("timestamp")` at the top level. The Time line is therefore always
dropped, and the speaker was never rendered at all: what upstream actually
hands its QA model is the bare utterance.

`render_metadata: true` (the default) restores the date upstream evidently
meant to show and adds the speaker. Three reasons:

- **Parity.** Every other baseline's QA agent sees who said it and when:
  `icl` (`[dia_id] date speaker: text`), `mem0` (which folds both
  into the content for exactly this reason), `zep`.
- **The benchmarks ask for it.** LoCoMo's temporal prompt says "Use DATE of
  CONVERSATION"; LongMemEval's temporal questions need session dates next to
  `question_date`; "who said it" questions need the speaker.
- **Retrieval is untouched.** Only the bare message text is ever embedded,
  under both values — the knob changes what the QA model reads, not what is
  retrieved.

`memo: {render_metadata: false}` reproduces upstream's output exactly, so the
effect of the choice can be measured rather than argued.

## Model configuration (two arms)

This RAG baseline calls no LLM, so the only model it has is the embedder.
`UNIFIED_MODEL_KEYS` maps `embedding` alone; `unified_models.llm` has nothing
to switch here (the answering model is the frame's `llm_model`).

| | faithful arm (`arm: faithful` — `CONFIG_DEFAULTS`) | unified arm (`arm: unified` — example `unified_models`) |
|---|---|---|
| embedder (`embedding_model`) | `sentence-transformers/all-MiniLM-L6-v2`, local, 384-dim — LightMem Table 5 (`f_index`, shared by LightMem and its baselines) and `NaiveRAGConfig`'s default | `text-embedding-3-small`, API, 1536-dim |

The toolkit's shipped `configs/NaiveRAG.json` names `text-embedding-3-small`
instead; where a paper and its code disagree the paper wins, and the shipped
value is exactly the unified arm. No dimension knob moves with the embedder: similarity is computed
on whatever width it returns.

| key | default | source |
|---|---|---|
| `embedding_model` | `sentence-transformers/all-MiniLM-L6-v2` | LightMem Table 5; `naive_rag.py` default |
| `embedding_device` | `null` (cuda if visible, else cpu) | upstream `use_gpu` is `"cpu"`, its shipped config `"cuda"`; speed only |
| `top_k` | `10` | `memory_search.py --top-k` default; the paper does not state k |
| `render_metadata` | `true` | see above; `false` = upstream's actual output |

## Deviations from running upstream

| category | items |
|---|---|
| Same algorithm | one unit per message; the bare message text is what is embedded; mem0's OpenAI newline flattening; cosine similarity; top-k 10; best-first order; upstream's `Key: value` hit layout |
| Integration adaptations (not algorithm) | numpy exact cosine instead of an embedded Qdrant collection (local-mode Qdrant is exact too); one batched embedding request per build call instead of one per message (identical vectors up to float rounding); each message embedded once — upstream embeds it TWICE (`_create_memory` tests `data in existing_embeddings`, a string against the vector, always False, and re-embeds), which doubles the embedding bill and changes nothing else; answering via the shared QA agent |
| Deliberate | `render_metadata: true` (above) |

## Cost profile

The cheap end of the frontier. BUILD makes no LLM call — on the faithful arm
it is local MiniLM compute (not an API call, so it never appears in a token
count: `run_record.json` names the model and its device, and `phase_seconds`
is the only figure that covers it); on the unified arm it is embedding tokens,
priced and tracked through `core.llm.Embedding`. RETRIEVE embeds one query.
The context the QA agent carries is `top_k` short messages.

## Run

```bash
# build rag's isolated env
cd baselines/rag && uv sync

# run it (from the repo root)
uv run --project baselines/rag python -m baselines.eval_harness \
        --config baselines/config.example.yaml      # harness: rag
```

Print the method defaults and what `arm: unified` sets with
`... eval_harness --describe rag`.
