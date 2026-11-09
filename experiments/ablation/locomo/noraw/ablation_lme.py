"""Retrieval-granularity ablation of MAPE-A_LME (7723a2cc870a) on LongMemEval.

MAPE-A_LME retrieves at the level of SESSIONS: every atomic note and raw round is
scored by hybrid RRF (dense + BM25), a session's score is its best item plus 0.15
times the sum of its other items, the top sessions are selected, and ALL notes of
each selected session are returned (plus five raw rounds and a time-anchor
session). This harness keeps the memory, the scorer, the raw rounds and the
compose stage fixed, and changes only which notes are returned:

  full      MAPE-A_LME unchanged (the reference arm; it also writes the memory cache).
  noexpand  no session expansion: the top-N notes by item score, where N is the
            number of notes the full retrieval returns for the same question;
            each session block holds only its retrieved notes.
  flat10    flat top-k: the 10 highest-scoring notes, each its own block
            (no grouping by session); raw rounds still attached.
  noraw     MAPE-A_LME unchanged except that the five raw rounds are not attached:
            raw rounds stay in the index and in session scoring, so the
            selected sessions and their notes are exactly the full arm's.
  noraw_nocompose  `noraw` retrieval with no compose stage: compose_memory_context
            is the contract's default (returns None), so the host renders the
            retrieval as JSON (BaseWorkflow.default_context), exactly as for the
            compose ablation; no `ablation` record is added, since it would show
            up in the context the QA model reads.
  handcrafted  retrieval as a hand-built memory system does it, over the same
            notes: the index holds the notes only (raw rounds removed from it),
            the 10 highest-scoring notes by the same hybrid RRF come back, each
            its own block, and no raw rounds are attached.

Memory is shared byte for byte. The full arm builds it once (LLM note extraction)
and pickles the built state per question under CACHE, keyed by a hash of the
question's haystack; the other arms load that state and do no extraction. A
missing cache entry raises, so a gap fails loudly. Item scores for the two
ablated arms use the same RRF over the full (untruncated) dense and BM25
rankings, so N notes are always available; the full arm keeps its own pool.

`variant.txt` in this directory names the arm. MAPE-A_LME's code is
`mape_a.py` (its memo.py, byte-identical) with its `src/`.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mape_a as M  # noqa: E402
from evolution.memo_class import MemoClass  # noqa: E402
from mape_a import AtomicLedgerMemo  # noqa: E402

VARIANT = (HERE / "variant.txt").read_text().strip()
assert VARIANT in ("full", "noexpand", "flat10", "noraw", "noraw_nocompose", "handcrafted"), VARIANT
CACHE = Path(os.environ.get("MAPE_ABL_CACHE", "/hf-cache/mape_ablation_7723a2"))
FLAT_K = 10
_STATE = ("rounds", "notes", "session_rounds", "session_notes", "session_meta",
          "index_items", "_emb")


def haystack_key(sessions: Any) -> str:
    blob = json.dumps(sessions, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


class RetrievalGranularityMemo(AtomicLedgerMemo):

    async def build_memory_from_data(self, recorder) -> None:
        init = recorder.init or {}
        sessions = init.get("sessions", []) or []
        path = CACHE / f"{haystack_key(sessions)}.pkl"
        if VARIANT == "full":
            await self._ingest(sessions)
            CACHE.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + f".tmp{os.getpid()}")
            with open(tmp, "wb") as f:
                pickle.dump({k: getattr(self, k) for k in _STATE}, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
            return
        state = pickle.loads(path.read_bytes())
        for k, v in state.items():
            setattr(self, k, v)
        texts = [self.notes[i].text if kind == "note" else self.rounds[i].embed_text
                 for kind, _s, i in self.index_items]
        self._bm25 = None
        if M.BM25Okapi is not None and texts:
            try:
                self._bm25 = M.BM25Okapi([M.tokenize(t) or ["_"] for t in texts])
            except Exception:
                self._bm25 = None
        if VARIANT == "handcrafted":
            # notes-only index: the dense matrix lists notes first (mape_a._ingest)
            n = len(self.notes)
            self._emb = self._emb[:n] if self._emb is not None else None
            self.index_items = [it for it in self.index_items if it[0] == "note"]
            assert all(it[2] == j for j, it in enumerate(self.index_items))
            self._bm25 = None
            if M.BM25Okapi is not None and n:
                try:
                    self._bm25 = M.BM25Okapi([M.tokenize(nt.text) or ["_"] for nt in self.notes])
                except Exception:
                    self._bm25 = None
        self._built = True

    async def _item_scores(self, query: str) -> Dict[int, float]:
        """MAPE-A_LME's hybrid RRF item scores, over untruncated rankings."""
        n = len(self.index_items)
        dense_order: List[int] = []
        if self._emb is not None and query.strip():
            try:
                q = np.asarray(await self._embedder.get_embedding(query), dtype=np.float32)
                nq = np.linalg.norm(q)
                if nq > 0:
                    dense_order = list(np.argsort(-(self._emb @ (q / nq)))[:n])
            except Exception:
                dense_order = []
        bm25_order: List[int] = []
        if self._bm25 is not None and query.strip():
            try:
                bm25_order = list(np.argsort(-np.asarray(self._bm25.get_scores(M.tokenize(query))))[:n])
            except Exception:
                bm25_order = []
        if dense_order and bm25_order:
            return M.rrf_fuse([dense_order, bm25_order], k=M._RRF_K)
        order = dense_order or bm25_order
        return {int(i): 1.0 / (r + 1) for r, i in enumerate(order)}

    def _block(self, s: int, notes: List[str], score: float) -> Dict[str, Any]:
        meta = self.session_meta.get(s, {})
        return {
            "session_idx": s,
            "session_id": meta.get("session_id", ""),
            "date_str": meta.get("date_str", ""),
            "date_sort": meta["date"].isoformat() if meta.get("date") else "",
            "notes": notes,
            "score": float(score),
            "is_raw": False,
        }

    async def retrieve_memory_for_query(self, recorder) -> Dict[str, Any]:
        if VARIANT == "handcrafted":
            return await self._retrieve_handcrafted(recorder)
        full = await super().retrieve_memory_for_query(recorder)
        if VARIANT == "noraw_nocompose":
            full["raw_rounds"] = []
            return full
        n_full = sum(len(b["notes"]) for b in full.get("sessions", []) if not b.get("is_raw"))
        full["ablation"] = {"variant": VARIANT, "n_notes_full": n_full}
        if VARIANT == "noraw":
            full["ablation"]["n_raw_rounds_dropped"] = len(full.get("raw_rounds", []))
            full["raw_rounds"] = []
            return full
        if VARIANT == "full" or not self.notes:
            return full
        query = str((recorder.init or {}).get("query", "") or "")
        scores = await self._item_scores(query)
        ranked: List[Tuple[float, int]] = sorted(
            ((float(sc), self.index_items[i][2]) for i, sc in scores.items()
             if 0 <= int(i) < len(self.index_items) and self.index_items[int(i)][0] == "note"),
            key=lambda t: -t[0])
        if VARIANT == "noexpand":
            picked = ranked[:n_full]
            by_sess: Dict[int, List[Tuple[int, float]]] = {}
            for sc, ni in picked:
                by_sess.setdefault(self.notes[ni].session_idx, []).append((ni, sc))
            blocks = [self._block(s, [self.notes[ni].text for ni, _ in sorted(v)], max(sc for _, sc in v))
                      for s, v in by_sess.items()]
        else:
            picked = ranked[:FLAT_K]
            blocks = [self._block(self.notes[ni].session_idx, [self.notes[ni].text], sc) for sc, ni in picked]
        blocks.sort(key=lambda b: -b["score"])
        out = dict(full)
        out["sessions"] = blocks
        out["n_sessions"] = len(blocks)
        out["ablation"].update({"n_notes": len(picked), "n_blocks": len(blocks)})
        return out

    async def _retrieve_handcrafted(self, recorder) -> Dict[str, Any]:
        init = recorder.init or {}
        query = str(init.get("query", "") or "")
        qd = str(init.get("question_date", "") or "")
        scores = await self._item_scores(query) if self.notes else {}
        ranked = sorted(((float(sc), int(i)) for i, sc in scores.items() if 0 <= int(i) < len(self.notes)),
                        key=lambda t: -t[0])[:FLAT_K]
        blocks = [self._block(self.notes[ni].session_idx, [self.notes[ni].text], sc) for sc, ni in ranked]
        return {"sessions": blocks, "raw_rounds": [], "query": query, "question_date": qd,
                "n_sessions": len(blocks),
                "ablation": {"variant": VARIANT, "n_notes": len(blocks), "n_blocks": len(blocks)}}


if VARIANT == "noraw_nocompose":
    # The contract's own default hook (returns None): the host then renders the
    # retrieval with the benchmark's default context and records compose_hook
    # "default" (core.workflow.overrides_compose compares against this object).
    RetrievalGranularityMemo.compose_memory_context = MemoClass.compose_memory_context
