"""RAG over raw messages, as LightMem's NaiveRAG baseline implements it, as a retrieval MemoClass.

The method is the "Naive RAG" of Gao et al. 2023 (arXiv 2312.10997) — retrieve
by the raw query, read what comes back, nothing in between — on the retrieval
idea of Lewis et al. 2020 (arXiv 2005.11401). The concrete recipe is the
NaiveRAG baseline of LightMem (ICLR 2026, arXiv 2510.18866), which reports it
on LoCoMo and LongMemEval-S with gpt-4o-mini and with
Qwen3-30B-A3B-Instruct-2507 (this repo's second backbone):

    upstream  github.com/zjunlp/LightMem @ 8449d574df6bae1bdf3314a1564da65e2f37e046
              src/lightmem/memory_toolkits/memories/layers/naive_rag.py
              (+ memory_construction.py, memory_search.py, memories/datasets/)

BUILD: every ingestion unit — one message — is embedded on its own text and
kept. RETRIEVE: the query is embedded the same way and the `top_k` units with
the highest cosine similarity come back, best first. No LLM anywhere: this is
the honest "does building a memory beat retrieving raw turns?" control.

WRITTEN HERE, NOT VENDORED. Upstream's NaiveRAGLayer is a thin wrapper over
mem0 1.0.2 (`envs/rag_requirements.txt` pins mem0ai==1.0.2): `Memory.add(...,
infer=False)` per message, `Memory.search(query, limit=k)`. On that path mem0
does exactly what this file does, and nothing else:

  add      `Memory._add_to_vector_store` (infer=False branch): embed the
           message content — HuggingFaceEmbedding: `SentenceTransformer.encode`,
           no normalisation; OpenAIEmbedding: "\\n" -> " " first — and insert
           the vector into a Qdrant collection with Distance.COSINE.
  search   `Memory._search_vector_store`: embed the query the same way, Qdrant
           cosine top-`limit`; threshold None and no reranker configured, so
           neither step filters anything.

So the vendored stack (mem0 slice, qdrant-client, sqlite history, telemetry)
would buy byte-identity of code this baseline never exercises. It was NOT
reused from `baselines/mem0/src/` either: that is mem0 v2.0.17, whose
search is HYBRID (vector + BM25, `_search_vector_store` there) — running
NaiveRAG on it would silently stop being naive. The deviations from running
upstream, all listed in README.md:

  * a build call is embedded in one batch, not one request per message
    (identical vectors up to float rounding; far fewer requests);
  * each message is embedded ONCE. Upstream embeds it twice:
    `_create_memory(data, existing_embeddings=<the vector>)` tests
    `data in existing_embeddings` — a string against a list of floats, always
    False — and re-embeds. Retrieval is unaffected; only the bill doubles;
  * exact cosine in numpy instead of an embedded Qdrant collection (local-mode
    Qdrant is an exact search too, so the same hits come back);
  * `render_metadata` (below).

RENDERING — the one deliberate deviation, behind a knob. Upstream builds each
hit as "Time: <session timestamp>\\nMemory: <text>", dropping any field that is
None. It stores the timestamp and speaker name as mem0 metadata, but mem0 1.x
returns custom metadata NESTED under `item["metadata"]` while upstream reads
`item.get("timestamp")` at the top level, so the Time line is ALWAYS dropped,
and the speaker is never rendered at all. What upstream actually hands its QA
model is therefore the bare utterance. With `render_metadata: true` (the
default) each hit carries its date and speaker — what upstream evidently
intended for the date, and what every other baseline here gives its QA agent
(icl, mem0, zep all keep speaker and date), without which LoCoMo's
temporal prompt ("Use DATE of CONVERSATION") and every "who said it" question
cannot be answered from this memory. `memo: {render_metadata: false}`
reproduces upstream's output exactly. RETRIEVAL IS THE SAME EITHER WAY: only
the bare message text is ever embedded, as upstream does.

Ingestion units (recorder.init dispatch), the one-message rule upstream
defines for LoCoMo and LongMemEval:
  locomo ("conversation"): one unit per turn, text = the turn's `text` only
    (upstream: content=text, name=speaker, timestamp=session date — the image
    caption stays metadata, never content), speaker, session date_time.
  longmemeval ("sessions"): one unit per message, every role (upstream stores
    each message as role "user" and keeps the real role as metadata), time =
    session date.

Hits are returned as {"passages": [...]} for the SHARED QA agent, which answers
for every memo (answering is not part of the contract) — the mem0 / lightmem /
simplemem shape. Upstream's own QA and judge prompts are therefore unused.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from core.memo_class import MemoClass
from baselines.model_config import (
    get_embedder, is_api_embedding_model, resolve_device,
)

# No SDK patches are installed here, unlike the vendored baselines: this memo
# makes no chat call, and its only API traffic — the embedder, on the unified
# arm — goes through core.llm.Embedding (via model_config.APIEmbedder),
# which reports its own usage and obeys the model policy. A local embedder is
# loaded by model_config.get_embedder: once per process, shared by every user,
# one encode at a time (baselines/concurrency.py).


# Method config, faithful arm — module-level DATA: eval_harness.py resolves it
# (with `arm` / `unified_models` / `memo:`) and hands the result to the memo's
# constructor; the class itself only reads self.config.
CONFIG_DEFAULTS = {
    # LIGHTMEM PAPER Table 5: the f_index embedding model is all-MiniLM-L6-v2,
    # a black-font entry there, i.e. shared by LightMem AND its baselines —
    # and NaiveRAGConfig's own default (naive_rag.py, `retriever_name_or_path`).
    # 384-dim, local sentence-transformer. The toolkit's shipped
    # configs/NaiveRAG.json names text-embedding-3-small instead; where a paper
    # and its code disagree the paper wins. A
    # `text-embedding-*` name switches to the OpenAI API embedder. No
    # dimension knob: similarity is computed on whatever width the embedder
    # returns.
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    # Device for a LOCAL embedder. None/"auto" => cuda if a GPU is visible,
    # else cpu. Upstream defaults to "cpu" and its shipped config says "cuda";
    # the device changes speed, not results. Ignored for an API embedder.
    "embedding_device": None,
    # Hits per query. UPSTREAM: memory_search.py `--top-k`, default 10 (and
    # the value its README runs with). The paper does not state k.
    "top_k": 10,
    # True: each hit renders as "Time: <date>\nMemory: <speaker>: <text>".
    # False: upstream's output as it actually runs — "Memory: <text>" — see
    # RENDERING in the module docstring. Rendering only: retrieval is
    # identical under both values.
    "render_metadata": True,
}
# `arm: unified` writes unified_models.embedding into this key. There is no
# LLM role to map: this RAG makes no LLM call, so unified_models.llm has
# nothing to switch here (the shared QA agent is the frame's `llm_model`).
UNIFIED_MODEL_KEYS = {"embedding": ("embedding_model",)}


@dataclass(frozen=True)
class _Unit:
    """One stored message: the embedded text plus what rendering may show."""
    text: str
    speaker: Optional[str]
    time: Optional[str]


def _init_to_units(init: Dict) -> List[_Unit]:
    """recorder.init -> ordered [_Unit, ...], one per message."""
    if "conversation" in init:
        from benchmarks.locomo.env import extract_sessions   # MAPE benchmarks — NOT the HF library
        units: List[_Unit] = []
        # extract_sessions yields (session_idx, date_time, turns) in session
        # order; date_time is the per-session timestamp of every turn in it.
        for _idx, date_time, turns in extract_sessions(init["conversation"]):
            for t in turns:
                units.append(_Unit(str(t.get("text", "")),
                                   str(t.get("speaker", "")) or None,
                                   str(date_time or "") or None))
        return units
    if "sessions" in init:
        return [_Unit(str(m.get("content", "")),
                      str(m.get("role", "")) or None,
                      str(s.get("date", "")) or None)
                for s in init["sessions"] for m in s.get("messages", [])]
    raise KeyError(f"unrecognized recorder.init keys: {list(init)}")


def _render(unit: _Unit, with_metadata: bool) -> str:
    """One hit as the QA agent reads it. Same layout as upstream's
    `NaiveRAGLayer.retrieve`: "Key: value" lines, a None field dropped."""
    if with_metadata:
        memory = f"{unit.speaker}: {unit.text}" if unit.speaker else unit.text
        fields = {"Time": unit.time, "Memory": memory}
    else:
        # What upstream's lookup actually yields: Time is None, no speaker.
        fields = {"Time": None, "Memory": unit.text}
    return "\n".join(f"{k}: {v}" for k, v in fields.items() if v is not None)


class RAGMemo(MemoClass):
    def __init__(self, config=None):
        super().__init__(config)
        self._embedder = None                       # shared, from model_config (lazy)
        self._units: List[_Unit] = []
        self._vectors: Optional[np.ndarray] = None  # (len(self._units), dim)

    def _ensure_embedder(self) -> None:
        if self._embedder is not None:
            return
        name = self.config["embedding_model"]
        device = None if is_api_embedding_model(name) else resolve_device(self.config["embedding_device"])
        self._embedder = get_embedder(name, device)

    def _embed(self, texts: List[str]) -> np.ndarray:
        """(len(texts), dim) float32, the way mem0 1.0.2's embedder would."""
        if is_api_embedding_model(self.config["embedding_model"]):
            # mem0's OpenAIEmbedding.embed flattens newlines before the request.
            texts = [t.replace("\n", " ") for t in texts]
        vectors = self._embedder.encode(texts, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32).reshape(len(texts), -1)

    # Embedding is synchronous (a local model, or core.llm's sync entry
    # point): each hook runs its body on a worker thread so other users keep
    # going meanwhile (see baselines/concurrency.py).

    async def build_memory_from_data(self, recorder) -> None:
        await asyncio.to_thread(self._build, recorder)

    async def retrieve_memory_for_query(self, recorder) -> Dict:
        return await asyncio.to_thread(self._retrieve, recorder)

    def _build(self, recorder) -> None:
        units = _init_to_units(recorder.init)
        if not units:
            return
        self._ensure_embedder()
        vectors = self._embed([u.text for u in units])
        # ADDITIVE: a later build call extends the memory.
        self._units.extend(units)
        self._vectors = vectors if self._vectors is None else np.vstack([self._vectors, vectors])

    def _retrieve(self, recorder) -> Dict:
        # Read-only: nothing here writes to self._units / self._vectors.
        if self._vectors is None or not len(self._units):
            return {}
        self._ensure_embedder()
        q = self._embed([str(recorder.init.get("query", ""))])[0]
        # Cosine similarity — Qdrant's Distance.COSINE — on unnormalised vectors.
        norms = np.linalg.norm(self._vectors, axis=1) * np.linalg.norm(q)
        scores = np.divide(self._vectors @ q, norms,
                           out=np.zeros(len(self._units), dtype=np.float32), where=norms > 0)
        k = int(self.config["top_k"])
        best = np.argsort(-scores, kind="stable")[:k]
        with_metadata = bool(self.config["render_metadata"])
        return {"passages": [_render(self._units[i], with_metadata) for i in best]}
