"""AtomicLedgerMemo — a build-time atomic memory ledger for LongMemEval.

STANCE: exploratory departure from the saturated 0.78 query-time-extraction line
(1718e4ddffce / 0a8a173d051b). Those harnesses retrieve whole sessions and then
run a per-session, query-FOCUSED "extract relevant facts" map on EVERY query.
That design saturated at 0.78 with ±3–4 run-to-run churn, and its two diagnosed
structural weaknesses are both properties of query-time extraction:

  * oblique facts are dropped by the per-query relevance filter (the preference
    collapse), forcing a separate brittle advice path;
  * the fact/item set is re-enumerated per query, so counting/temporal answers
    differ run-to-run (the dominant noise source the analysts named).

AtomicLedgerMemo instead builds the memory ONCE at write time:

  BUILD  — every session is read once into comprehensive, self-contained atomic
           NOTES (src/notes.py), cached on self. Notes + raw rounds are both
           indexed (dense + BM25).
  RETRIEVE — hybrid RRF over the combined note+round index selects the relevant
           SESSIONS; for each, ALL of its atomic notes are surfaced (no second
           relevance filter → oblique facts survive structurally), plus the top
           raw rounds for exact wording, plus a time-anchor insurance session.
  COMPOSE — the proven deterministic scaffolds are kept because they carry real,
           isolated wins (temporal date-resolution → ~0.8 temporal; the
           count/sum enumeration; the knowledge-update current-value gate; the
           preference SELECT/WRITE). The aggregation enumerator is rebuilt
           PRECISION-FIRST (per-item grounding, exclude planned/out-of-window),
           because the biggest counting error on the prior line was
           OVER-production (devices 9→4, kits 6→5, doctors 4→3, tanks 4→3). A
           single grounded deliberation produces the answer, framed
           non-authoritatively (the step-9 lesson: an authoritative stamp
           entrenches wrong consensus).

Cost is NOT an optimised axis on this run, so build-time extraction over every
session is a deliberate trade: more write-time LLM calls (parallelised) in
exchange for a consistent, comprehensive, query-agnostic memory.
"""
from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from evolution.memo_class import MemoClass  # noqa: E402
from core.llm import Agent, Embedding  # noqa: E402

from src.retrieval import (  # noqa: E402
    Round,
    build_rounds,
    tokenize,
    rrf_fuse,
    parse_date,
    fmt_time_delta,
    render_round,
)
from src.scaffold import (  # noqa: E402
    classify,
    resolve_anchor,
    build_time_table,
    build_resolved_block,
    parse_resolved_events,
    parse_valued_events,
    build_latest_value_block,
    parse_enumeration,
    fmt_qty,
    day_gap,
    guidance_for,
)
from src.reason import (  # noqa: E402
    DELIB_SYS,
    PREF_SELECT_SYS,
    PREF_WRITE_SYS,
    STATED_DUR_SYS,
    LATEST_VALUE_SYS,
    is_advice_query,
    parse_deliberation,
    parse_lines,
    parse_stated_duration,
    looks_like_refusal,
    draft_is_grounded,
)
from src.notes import Note, NOTE_EXTRACT_SYS, build_extract_user_prompt  # noqa: E402
from src.aggregate import (  # noqa: E402
    AGG_MAP_SYS,
    AGG_REDUCE_SYS,
    AGG_ADJUDICATE_SYS,
    DIFF_SYS,
    is_difference,
    build_map_user,
    build_reduce_user,
    build_adjudicate_user,
    build_diff_user,
    parse_candidates,
    parse_adjudication,
    parse_diff,
)

try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover - base image ships rank_bm25
    BM25Okapi = None


# ---- retrieval tunables ----------------------------------------------------
_RRF_K = 60
_POOL = 220
_MAX_SESSIONS = 10
_MAX_SESSIONS_AGG = 16      # aggregation needs breadth: a missing session => wrong total
_MAX_SESSIONS_PREF = 13     # advice: widen so oblique preferences are in the pool
_SECONDARY_W = 0.15
_REL_FRACTION = 0.28
_REL_FRACTION_AGG = 0.16
_REL_FRACTION_PREF = 0.14
_MIN_SESS = 3
_NOTES_PER_SESSION_CAP = 18   # atomic notes are compact; show (nearly) all of a relevant session
_RAW_ROUNDS_SHOWN = 5
_RAW_TOK = 1200

# ---- build tunables --------------------------------------------------------
_SESSION_CHARS_CAP = 16000    # per-session text fed to the note extractor
_NOTES_PER_SESSION_EXTRACT = 20

# ---- compose tunables ------------------------------------------------------
_FACTS_FOR_SCAFFOLD_TOK = 4200
# Advice SELECT pool: score-ordered (highest-relevance session FIRST) and capped
# per session. The 0.79 record flattened the pool OLDEST-FIRST then cut at a flat
# 60 lines, so the domain-matching gold session (verified rank 0-2 by retrieval
# score, but late in chronology) sat at cumulative note-position 94-142 and was
# cut before SELECT ever saw it — the diagnosed cause of preference 1/6. Ordering
# by score and capping per session guarantees the gold session reaches SELECT.
_PREF_POOL_CAP = 100
_PREF_NOTES_PER_SESSION = 12

_ABSTAIN = ("No relevant information about this question was found in your past "
            "conversations with the user.")

_NONE_RE = re.compile(r"[^a-z]")
# A current-STATE count ("how many followers do I have now/currently") is a
# knowledge-update lookup of the latest value, NOT an aggregation over occasions.
# Kept from the 0.78 line: it is what wins knowledge-update 15/15.
_CURRENT_VALUE_RE = re.compile(
    r"\b(how many|how much)\b.*\b(now|currently|right now|at the moment|"
    r"these days|at present|nowadays|so far)\b"
)
# A cumulative/possession count ("how many tanks do I currently have, INCLUDING
# the one for my friend's kid", "... in total") is a real aggregation even with a
# present-time word, so the current-value gate is overridden.
_CUMULATIVE_RE = re.compile(
    r"\b(including|include|in total|altogether|combined|all together|in all|"
    r"counting|between us|put together)\b"
)

# A "current / most-recent VALUE" question (where/what/which/who is X now; what
# was my most recent X) — a knowledge-update lookup where SEVERAL dated values
# may coexist and the NEWEST one is the answer. This is distinct from a
# current-STATE COUNT ("how many followers now", handled by the aggregation
# gate) and from a DURATION question. Targets the two step-13 knowledge-update
# regressions (9ea5eabc "most recent family trip" -> gold Paris, the newer trip,
# drowned by ~10 salient Hawaii notes; 07741c45 "currently keep sneakers" ->
# gold shoe rack, the newer value, discounted as a future plan): both had the
# right value present and correctly dated, but note salience/phrasing led the
# sealed answer to the older value. A deterministic newest-value resolver makes
# the recency explicit.
_LATEST_VALUE_WH_RE = re.compile(r"\b(where|what|which|who|whose|whom)\b")
_LATEST_CUE_RE = re.compile(
    r"\b(currently|now|right now|these days|at the moment|at present|nowadays|"
    r"most recent|most recently|latest|current|last time|up to date|up-to-date)\b"
)

# ---- aggregation enumerator ------------------------------------------------
# The single-digest precision-first enumerator of the 0.80 record is REPLACED by
# the per-session map-reduce pipeline in src/aggregate.py (see `_aggregate`).
# Rationale: every multi-session failure of the record was an aggregation question
# with PERFECT retrieval recall, lost downstream to signal dilution, truncation,
# modality error, and double-counting — all of which are properties of reading one
# big diluted digest in a single call, and all of which per-session focus removes.

_DATE_SYS = (
    "You identify calendar dates. You do NOT compute durations or answer.\n"
    "You are given the current date, a question about timing, and evidence; each "
    "note is stamped with the date of the conversation it came from. For each "
    "specific event, action, or milestone the question refers to, give its "
    "calendar date. Output one per line, format 'EVENT: <short description> | "
    "DATE: <YYYY/MM/DD>'. If an event happened during a conversation and no other "
    "date is stated in the text, use that conversation's date. If a date truly "
    "cannot be determined, write 'DATE: UNKNOWN'. Do not compute gaps or "
    "durations and do not answer the question."
)


class AtomicLedgerMemo(MemoClass):
    def __init__(self, config=None):
        super().__init__()
        self._config = config or {}
        self.rounds: list[Round] = []
        self.notes: list[Note] = []
        self.session_rounds: dict[int, list[int]] = {}
        self.session_notes: dict[int, list[int]] = {}
        self.session_meta: dict[int, dict] = {}
        # combined index: self.index_items[i] = (kind, session_idx, local_idx)
        self.index_items: list[tuple[str, int, int]] = []
        self._emb: np.ndarray | None = None
        self._bm25 = None
        self._built = False
        self._embedder = Embedding()

    # ------------------------------------------------------------------
    # Phase 1 — build: extract atomic notes (LLM) + dual index
    # ------------------------------------------------------------------
    async def build_memory_from_data(self, recorder) -> None:
        init = recorder.init or {}
        sessions = init.get("sessions", []) or []
        await self._ingest(sessions)

    async def _ingest(self, sessions) -> None:
        rounds = build_rounds(sessions)
        self.rounds = rounds
        self.session_rounds = {}
        self.session_meta = {}
        for pos, r in enumerate(rounds):
            self.session_rounds.setdefault(r.session_idx, []).append(pos)
            if r.session_idx not in self.session_meta:
                self.session_meta[r.session_idx] = {
                    "session_id": r.session_id,
                    "date_str": r.date_str,
                    "date": r.date,
                }

        if not rounds:
            self._built = True
            return

        # ---- build-time note extraction (one LLM call per session, parallel) ----
        sess_ids = sorted(self.session_rounds.keys())
        results = await asyncio.gather(
            *[self._extract_session_notes(s) for s in sess_ids],
            return_exceptions=True,
        )
        self.notes = []
        self.session_notes = {}
        for s, res in zip(sess_ids, results):
            lines = [] if isinstance(res, Exception) else (res or [])
            meta = self.session_meta.get(s, {})
            for ln in lines:
                idx = len(self.notes)
                self.notes.append(Note(
                    text=ln,
                    session_idx=s,
                    session_id=meta.get("session_id", f"session_{s + 1:03d}"),
                    date_str=meta.get("date_str", ""),
                    date=meta.get("date"),
                    note_idx=idx,
                ))
                self.session_notes.setdefault(s, []).append(idx)

        # ---- combined dual index (notes + raw rounds) ----
        item_texts: list[str] = []
        self.index_items = []
        for i, note in enumerate(self.notes):
            item_texts.append(note.text)
            self.index_items.append(("note", note.session_idx, i))
        for j, r in enumerate(self.rounds):
            item_texts.append(r.embed_text)
            self.index_items.append(("round", r.session_idx, j))

        if BM25Okapi is not None and item_texts:
            try:
                self._bm25 = BM25Okapi([tokenize(t) or ["_"] for t in item_texts])
            except Exception:
                self._bm25 = None

        try:
            vecs = await self._embedder.get_batch_embeddings(item_texts)
            mat = np.asarray(vecs, dtype=np.float32)
            if mat.ndim == 2 and mat.shape[0] == len(item_texts):
                norms = np.linalg.norm(mat, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                self._emb = mat / norms
            else:
                self._emb = None
        except Exception:
            self._emb = None

        self._built = True

    async def _extract_session_notes(self, session_idx: int) -> list[str]:
        poss = self.session_rounds.get(session_idx, [])
        if not poss:
            return []
        text = "\n".join(render_round(self.rounds[p]) for p in poss)
        if not text.strip():
            return []
        if len(text) > _SESSION_CHARS_CAP:
            text = text[:_SESSION_CHARS_CAP].rstrip() + " …"
        meta = self.session_meta.get(session_idx, {})
        label = meta.get("date_str") or "(date unknown)"
        out = await self._safe_ask(
            NOTE_EXTRACT_SYS, build_extract_user_prompt(label, text)
        )
        if not out or _is_none(out):
            return []
        return parse_lines(out, cap=_NOTES_PER_SESSION_EXTRACT)

    # ------------------------------------------------------------------
    # Phase 2 — retrieve (read-only; lazy build permitted)
    # ------------------------------------------------------------------
    async def retrieve_memory_for_query(self, recorder) -> dict:
        init = recorder.init or {}
        query = str(init.get("query", "") or "")
        question_date = str(init.get("question_date", "") or "")

        if not self._built:
            await self._ingest(init.get("sessions", []) or [])

        if not self.rounds:
            return {"sessions": [], "raw_rounds": [], "query": query,
                    "question_date": question_date}

        kind = classify(query)
        advice = is_advice_query(query, kind)
        if advice:
            max_sessions, rel_fraction = _MAX_SESSIONS_PREF, _REL_FRACTION_PREF
        elif kind.agg:
            max_sessions, rel_fraction = _MAX_SESSIONS_AGG, _REL_FRACTION_AGG
        else:
            max_sessions, rel_fraction = _MAX_SESSIONS, _REL_FRACTION

        n_items = len(self.index_items)

        # ---- dense ranking over the combined index -------------------
        dense_order: list[int] = []
        max_sim = 0.0
        if self._emb is not None and query.strip():
            try:
                qv = await self._embedder.get_embedding(query)
                q = np.asarray(qv, dtype=np.float32)
                nq = np.linalg.norm(q)
                if nq > 0:
                    sims = self._emb @ (q / nq)
                    dense_order = list(np.argsort(-sims)[:_POOL])
                    max_sim = float(np.max(sims)) if sims.size else 0.0
            except Exception:
                dense_order = []

        # ---- lexical ranking -----------------------------------------
        bm25_order: list[int] = []
        if self._bm25 is not None and query.strip():
            try:
                scores = self._bm25.get_scores(tokenize(query))
                bm25_order = list(np.argsort(-np.asarray(scores))[:_POOL])
            except Exception:
                bm25_order = []

        # ---- fuse to per-item scores ---------------------------------
        if dense_order and bm25_order:
            fused = rrf_fuse([list(dense_order), list(bm25_order)], k=_RRF_K)
        elif dense_order:
            fused = {int(i): 1.0 / (r + 1) for r, i in enumerate(dense_order)}
        elif bm25_order:
            fused = {int(i): 1.0 / (r + 1) for r, i in enumerate(bm25_order)}
        else:
            fused = {}

        # ---- aggregate item scores to session scores -----------------
        sess_best: dict[int, float] = {}
        sess_extra: dict[int, float] = {}
        round_item_scores: list[tuple[int, float]] = []  # (round_pos, score)
        for item_idx, sc in fused.items():
            item_idx = int(item_idx)
            if item_idx < 0 or item_idx >= n_items:
                continue
            kind_i, s, local = self.index_items[item_idx]
            if kind_i == "round":
                round_item_scores.append((local, sc))
            if sc > sess_best.get(s, -1.0):
                if s in sess_best:
                    sess_extra[s] = sess_extra.get(s, 0.0) + sess_best[s]
                sess_best[s] = sc
            else:
                sess_extra[s] = sess_extra.get(s, 0.0) + sc

        sess_score = {
            s: sess_best[s] + _SECONDARY_W * sess_extra.get(s, 0.0)
            for s in sess_best
        }
        if sess_score:
            ranked_sessions = sorted(sess_score.keys(), key=lambda s: -sess_score[s])
        else:
            ranked_sessions = self._recent_sessions()

        top_score = sess_score.get(ranked_sessions[0], 0.0) if ranked_sessions else 0.0
        selected: list[dict] = []
        chosen_sessions: set[int] = set()
        for rank, s in enumerate(ranked_sessions):
            if len(selected) >= max_sessions:
                break
            sc = sess_score.get(s, 0.0)
            if rank >= _MIN_SESS and top_score > 0 and sc < rel_fraction * top_score:
                break
            block = self._render_session_notes(s, sc)
            if block is not None:
                selected.append(block)
                chosen_sessions.add(s)

        # ---- time-anchor insurance: force the session nearest the
        #      query's resolved time anchor (temporal recall safety net).
        now_dt = parse_date(question_date)
        anchor = resolve_anchor(query, now_dt)
        if anchor is not None:
            near_s, near_gap = self._session_nearest(anchor.target)
            if (near_s is not None and near_s not in chosen_sessions
                    and near_gap <= anchor.tol):
                block = self._render_session_notes(near_s, sess_score.get(near_s, 0.0))
                if block is not None:
                    selected.append(block)
                    chosen_sessions.add(near_s)

        # ---- raw rounds for exact wording ----------------------------
        raw_rounds = self._top_raw_rounds(round_item_scores, chosen_sessions)

        return {
            "sessions": selected,
            "raw_rounds": raw_rounds,
            "query": query,
            "question_date": question_date,
            "max_sim": max_sim,
            "n_sessions": len(self.session_rounds),
        }

    def _render_session_notes(self, s: int, score: float) -> dict | None:
        note_ids = self.session_notes.get(s, [])
        meta = self.session_meta.get(s, {})
        notes = [self.notes[i].text for i in note_ids[:_NOTES_PER_SESSION_CAP]]
        if not notes:
            # no notes extracted for this session: fall back to its raw text so a
            # relevant session is never silently empty.
            poss = self.session_rounds.get(s, [])
            raw = "\n".join(render_round(self.rounds[p]) for p in poss)
            raw = _truncate_tokens(raw, _RAW_TOK)
            if not raw.strip():
                return None
            return {
                "session_idx": s,
                "session_id": meta.get("session_id", ""),
                "date_str": meta.get("date_str", ""),
                "date_sort": meta["date"].isoformat() if meta.get("date") else "",
                "notes": [raw],
                "score": float(score),
                "is_raw": True,
            }
        return {
            "session_idx": s,
            "session_id": meta.get("session_id", ""),
            "date_str": meta.get("date_str", ""),
            "date_sort": meta["date"].isoformat() if meta.get("date") else "",
            "notes": notes,
            "score": float(score),
            "is_raw": False,
        }

    def _top_raw_rounds(self, round_item_scores, chosen_sessions) -> list[dict]:
        # best score per round position
        best: dict[int, float] = {}
        for pos, sc in round_item_scores:
            if sc > best.get(pos, -1.0):
                best[pos] = sc
        ordered = sorted(best.keys(), key=lambda p: -best[p])
        out: list[dict] = []
        seen: set[int] = set()
        for pos in ordered:
            if pos in seen:
                continue
            seen.add(pos)
            r = self.rounds[pos]
            out.append({
                "date_str": r.date_str,
                "date_sort": r.date.isoformat() if r.date else "",
                "text": _truncate_tokens(render_round(r), _RAW_TOK),
            })
            if len(out) >= _RAW_ROUNDS_SHOWN:
                break
        return out

    def _recent_sessions(self) -> list[int]:
        items: list[tuple[int, float, int]] = []
        for s, meta in self.session_meta.items():
            d = meta.get("date")
            if d is None:
                items.append((1, 0.0, s))
            else:
                items.append((0, -d.timestamp(), s))
        items.sort(key=lambda it: (it[0], it[1]))
        return [it[2] for it in items]

    def _session_nearest(self, target):
        best_s = None
        best_gap = 10 ** 9
        for s, meta in self.session_meta.items():
            d = meta.get("date")
            if d is None:
                continue
            g = day_gap(d, target)
            if g < best_gap:
                best_gap = g
                best_s = s
        return best_s, best_gap

    # ------------------------------------------------------------------
    # Phase 2b — compose
    # ------------------------------------------------------------------
    async def compose_memory_context(self, recorder, retrieved):
        sessions = (retrieved or {}).get("sessions", []) if isinstance(retrieved, dict) else []
        raw_rounds = (retrieved or {}).get("raw_rounds", []) if isinstance(retrieved, dict) else []

        init = getattr(recorder, "init", None)
        init = init if isinstance(init, dict) else {}
        q = str(init.get("query", "") or "")
        if isinstance(retrieved, dict) and not q:
            q = str(retrieved.get("query", "") or "")

        qd = str(init.get("question_date", "") or "")
        if not qd and isinstance(retrieved, dict):
            qd = str(retrieved.get("question_date", "") or "")
        now_dt = parse_date(qd)

        if not sessions:
            return _ABSTAIN

        kind = classify(q)
        advice = is_advice_query(q, kind)

        # ----- order sessions chronologically -------------------------
        def _skey(b):
            ds = b.get("date_sort", "")
            return (ds == "", ds, b.get("session_idx", 0))

        ordered = sorted(sessions, key=_skey)
        by_score = sorted(sessions, key=lambda b: -float(b.get("score", 0.0)))

        # ============================================================
        # ADVICE / PREFERENCE PATH — notes already carry oblique prefs
        # ============================================================
        if advice:
            return await self._compose_advice(q, qd, now_dt, ordered)

        # notes digest fed to the deterministic scaffolds
        facts_digest = _truncate_tokens(
            "\n".join(
                f"[{b.get('date_str') or 'date unknown'}]\n" +
                "\n".join(f"- {ln}" for ln in b.get("notes", []))
                for b in ordered
            ),
            _FACTS_FOR_SCAFFOLD_TOK,
        )

        scaffold_blocks: list[str] = []
        extra_guidance: list[str] = []

        ql = q.lower()
        is_current_value = bool(_CURRENT_VALUE_RE.search(ql)) and not bool(
            _CUMULATIVE_RE.search(ql)
        )
        # A DIFFERENCE question ("how much faster / exceeded by / how much more")
        # is a subtraction of two values, NOT a sum or a count. Routing it through
        # the sum enumerator baked the wrong number (marathon finish time instead
        # of the 12-minute overage). Handle it with a dedicated two-value scaffold
        # and suppress the aggregation path. An explicit "in total / including"
        # cue (cumulative) wins back the sum path. Temporal differences (time gaps
        # between two events) are LEFT to the strong temporal date-resolver, which
        # already scores ~0.85-0.89; this handles only non-temporal differences
        # (overage, speed, money, count-diff) — the marathon-overage failure mode.
        wants_diff = (
            is_difference(ql)
            and not kind.temporal
            and not bool(_CUMULATIVE_RE.search(ql))
        )
        run_agg = kind.agg and not is_current_value and not wants_diff

        if wants_diff:
            dblk = await self._difference(q, facts_digest, qd, now_dt)
            if dblk:
                scaffold_blocks.append(dblk)
                extra_guidance.append(
                    "- This asks for a DIFFERENCE between two values (e.g. how much "
                    "more/faster/earlier, or by how much something was exceeded). A "
                    "COMPUTED DIFFERENCE block above shows the two values and their "
                    "subtraction; check the two values are the right ones, then "
                    "report that difference — do not report one of the raw values "
                    "on its own."
                )

        if run_agg:
            blk = await self._aggregate(q, ordered, qd, kind, now_dt)
            if blk:
                scaffold_blocks.append(blk)
        if not run_agg:
            kind.agg = False

        # Newest-value resolver for "current / most-recent VALUE" questions
        # (where/what/which is X now; most recent X). Only fires for a non-
        # aggregation, non-temporal wh-value question carrying a recency cue, and
        # only emits a block when >=2 distinct dated values actually exist (a real
        # update to disambiguate) — otherwise the normal path is unchanged.
        if (
            not run_agg
            and not kind.temporal
            and _LATEST_CUE_RE.search(ql)
            and _LATEST_VALUE_WH_RE.search(ql)
        ):
            lv = await self._latest_value(q, facts_digest, qd)
            if lv:
                scaffold_blocks.append(lv)
                extra_guidance.append(
                    "- This asks for the CURRENT / most-recent value of something "
                    "that changed over time. Use the newest-dated value shown in "
                    "the CURRENT (MOST RECENT) VALUE block above, even if an older "
                    "value is described in more detail."
                )

        anchor_idx = None
        anchor_desc = ""
        if kind.temporal:
            if kind.duration:
                stated = await self._stated_duration(q, facts_digest)
                if stated:
                    scaffold_blocks.append(
                        "STATED DURATION (a length of time written explicitly in "
                        "your past conversations — if the question asks how long / "
                        "how many days/weeks, use THIS value rather than computing "
                        "one from the conversation dates):\n- " + stated
                    )
                    extra_guidance.append(
                        "- A length of time is stated explicitly in the "
                        "conversations (see STATED DURATION above). Use that stated "
                        "duration as the answer; do not recompute it from the "
                        "conversation dates."
                    )

            resolved_block = ""
            if kind.duration or kind.order:
                resolved_block = await self._resolve_dates(q, facts_digest, qd, now_dt)
            if resolved_block:
                scaffold_blocks.append(resolved_block)
            else:
                dated = [
                    (b.get("date_str") or "(date unknown)", parse_date(b.get("date_str", "")))
                    for b in by_score
                ]
                dated = [(lbl, dt) for lbl, dt in dated if dt is not None][:6]
                table = build_time_table(dated, now_dt)
                if table:
                    scaffold_blocks.append(table)

            anchor = resolve_anchor(q, now_dt)
            if anchor is not None:
                anchor_desc = anchor.desc
                best_gap = 10 ** 9
                for i, b in enumerate(ordered):
                    dt = parse_date(b.get("date_str", ""))
                    if dt is None:
                        continue
                    g = day_gap(dt, anchor.target)
                    if g < best_gap:
                        best_gap = g
                        anchor_idx = i
                if anchor_idx is not None and best_gap <= anchor.tol:
                    extra_guidance.append(
                        f"- The phrase '{anchor.desc}' in the question points to "
                        f"around {anchor.target.date().isoformat()}; the "
                        f"conversation marked [MATCHES '{anchor.desc}'] below is the "
                        f"one it refers to. Base the answer on that conversation."
                    )
                else:
                    anchor_idx = None

        evidence_ctx = self._assemble_evidence(
            kind, extra_guidance, scaffold_blocks, ordered, raw_rounds,
            now_dt, anchor_idx, anchor_desc,
        )

        answer, quote = await self._deliberate(q, qd, evidence_ctx)
        return self._compose_with_answer(answer, quote, evidence_ctx)

    # ------------------------------------------------------------------
    # ADVICE / PREFERENCE path
    # ------------------------------------------------------------------
    async def _compose_advice(self, q, qd, now_dt, ordered) -> str:
        # Rank sessions by RETRIEVAL SCORE (highest relevance first). The
        # domain-matching gold session is rank 0-2 by hybrid score (verified in
        # the step-13 traces) but can be late in chronology, so the prior
        # oldest-first flatten+cut dropped it before SELECT. Score-ordering puts
        # it at the top of the pool; a per-session cap keeps breadth.
        by_score = sorted(ordered, key=lambda b: -float(b.get("score", 0.0)))
        pool: list[tuple[dict, list[str]]] = []
        for blk in by_score:
            lines = [ln for ln in blk.get("notes", []) if ln.strip()]
            if lines:
                pool.append((blk, lines))
        if not pool:
            return _ABSTAIN

        all_specifics: list[str] = []
        for _blk, lines in pool:
            all_specifics.extend(lines[:_PREF_NOTES_PER_SESSION])
            if len(all_specifics) >= _PREF_POOL_CAP:
                break
        all_specifics = all_specifics[:_PREF_POOL_CAP]
        pool_text = "\n".join(f"- {ln}" for ln in all_specifics)

        specifics = await self._select_specifics(q, pool_text)
        write_on = specifics if specifics else all_specifics[:3]
        draft = await self._write_preference(q, qd, write_on)
        if draft and specifics and not draft_is_grounded(draft, specifics):
            draft = ""
        return self._compose_preference_ctx(q, specifics, draft, pool, now_dt)

    async def _select_specifics(self, q: str, pool_text: str) -> list[str]:
        if not pool_text.strip():
            return []
        user = (
            f"The user's request: {q}\n\n"
            f"Things you know about the user:\n{pool_text}\n\n"
            "The 1-3 most useful details for this request (one per line, or NONE):"
        )
        out = await self._safe_ask(PREF_SELECT_SYS, user)
        if not out or _is_none(out):
            return []
        return parse_lines(out, cap=3)

    async def _write_preference(self, q: str, qd: str, specifics: list[str]) -> str:
        if not specifics:
            return ""
        spec_text = "\n".join(f"- {s}" for s in specifics)
        user = (
            f"Current date: {qd or '(unknown)'}\n"
            f"The user asks: {q}\n\n"
            f"Specific things you know about this user (build on these, by name):\n"
            f"{spec_text}\n\n"
            "Your specific, personalised reply:"
        )
        out = await self._safe_ask(PREF_WRITE_SYS, user)
        if not out or looks_like_refusal(out):
            return ""
        return _truncate_chars(out, 900)

    def _compose_preference_ctx(self, q, specifics, draft, pool, now_dt) -> str:
        blocks: list[str] = []
        if specifics:
            blocks.append(
                "WHAT YOU KNOW ABOUT THIS USER THAT IS RELEVANT TO THIS REQUEST "
                "(these come from the user's own past messages — use them):"
            )
            for s in specifics:
                blocks.append(f"- {s}")
            blocks.append("")
        blocks.append("HOW TO ANSWER:")
        blocks.append(
            "- The user wants a PERSONALIZED recommendation or opinion. Build your "
            "answer explicitly on the user's specific details shown here and refer "
            "to them by name."
        )
        blocks.append(
            "- Do NOT give generic advice that ignores these details, and do NOT "
            "refuse or say you lack information / real-time access — answer from "
            "what the user has told you."
        )
        blocks.append("")
        if draft:
            blocks.append(
                "SUGGESTED RESPONSE (a draft grounded in the details above — you "
                "may use or refine it; keep it specific to this user):"
            )
            blocks.append(draft.strip())
            blocks.append("")
        blocks.append("---")
        blocks.append("More of what the user has told you (oldest first):")
        # pool is score-ordered for SELECT; display it chronologically so the
        # "oldest first" framing holds and recency reads correctly.
        disp_pool = sorted(
            pool,
            key=lambda pl: (
                (pl[0].get("date_sort", "") == ""),
                pl[0].get("date_sort", ""),
                pl[0].get("session_idx", 0),
            ),
        )
        for blk, lines in disp_pool:
            label = blk.get("date_str") or "(date unknown)"
            sess_dt = parse_date(blk.get("date_str", ""))
            delta = fmt_time_delta(sess_dt, now_dt)
            tail = f" — {delta}" if delta else ""
            blocks.append(f"=== Conversation on {label}{tail} ===")
            for ln in lines:
                blocks.append(f"- {ln}")
            blocks.append("")
        return "\n".join(blocks).rstrip()

    # ------------------------------------------------------------------
    # Evidence assembly
    # ------------------------------------------------------------------
    def _assemble_evidence(self, kind, extra_guidance, scaffold_blocks, ordered,
                           raw_rounds, now_dt, anchor_idx, anchor_desc) -> str:
        header = self._header(kind, extra_guidance)
        blocks = [header]
        for sb in scaffold_blocks:
            blocks.append("")
            blocks.append(sb)

        blocks.append("")
        blocks.append("RECALLED FACTS (distilled from your past conversations; "
                      "oldest first):")
        last_i = len(ordered) - 1
        for i, b in enumerate(ordered):
            label = b.get("date_str") or "(date unknown)"
            sess_dt = parse_date(b.get("date_str", ""))
            delta = "" if kind.temporal else fmt_time_delta(sess_dt, now_dt)
            tail = f" — {delta}" if delta else ""
            if i == anchor_idx and anchor_desc:
                tail += f" [MATCHES '{anchor_desc}']"
            if i == last_i and len(ordered) > 1:
                tail += " [MOST RECENT conversation]"
            blocks.append(f"=== Conversation on {label}{tail} ===")
            for ln in b.get("notes", []):
                blocks.append(f"- {ln}")
            blocks.append("")

        if raw_rounds:
            blocks.append("SOURCE EXCERPTS (exact wording from the most relevant "
                          "moments — quote from here when precision matters):")
            for rr in raw_rounds:
                label = rr.get("date_str") or "(date unknown)"
                blocks.append(f"--- From the conversation on {label} ---")
                blocks.append(str(rr.get("text", "")).strip())
                blocks.append("")

        return "\n".join(blocks).rstrip()

    def _compose_with_answer(self, answer: str, quote: str, evidence_ctx: str) -> str:
        if not answer:
            nudge = (
                "A careful review of the conversations below did not surface the "
                "specific information this question asks for. If it is genuinely "
                "not present, tell the user they did not mention it rather than "
                "guessing; only give a specific answer if the conversations clearly "
                "contain it.\n\n---\n\n"
            )
            return nudge + evidence_ctx
        head = [
            "BEST ANSWER TO THE QUESTION (worked out from your past conversations "
            "below — state this answer; double-check it against the conversations "
            "and correct it only if they clearly contradict it):",
            answer.strip(),
        ]
        if quote:
            head.append(f'Supporting quote from the conversations: "{quote.strip()}"')
        head.append("")
        head.append("---")
        head.append("")
        return "\n".join(head) + evidence_ctx

    # ------------------------------------------------------------------
    # header
    # ------------------------------------------------------------------
    def _header(self, kind, extra_guidance) -> str:
        base = [
            "The following are facts recalled from your past conversations with "
            "the user that are relevant to the question, organized by conversation "
            "and ordered from OLDEST to NEWEST. Each conversation's date and how "
            "long before the current date it took place are shown.",
            "",
            "How to use these facts:",
            "- Answer using only the facts below and the current date; quote the "
            "user's exact wording when possible.",
            "- If a detail changed over time, the value in the MOST RECENT "
            "conversation (last below) is the current one.",
            "- If these facts do not contain the information asked for, say you do "
            "not have that information rather than guessing.",
        ]
        guide = guidance_for(kind) + list(extra_guidance)
        if guide:
            base.append("")
            base.append("For THIS question specifically:")
            base.extend(guide)
        return "\n".join(base)

    # ------------------------------------------------------------------
    # SCAFFOLD: aggregation (precision-first enumerate -> Python count/sum)
    # ------------------------------------------------------------------
    async def _aggregate(self, query: str, ordered: list, qd: str, kind,
                         now_dt) -> str:
        """PER-SESSION MAP-REDUCE aggregation.

        MAP: each selected session is read on its own (only its notes in context)
        and asked to list candidate items in the question's category, each tagged
        with QTY + STATUS (did/planned/missed) + a verbatim WHY. Per-session focus
        eliminates the single-digest enumerator's dilution and truncation (the
        trace-verified cause of model-kit 5->3, jewelry 3->1, devices 4->3).
        REDUCE: the small structured candidate list is filtered + deduped in one
        call (category / modality / time window / true-duplicate dedup). PYTHON:
        the kept list is counted / summed exactly.
        """
        sess_blocks = [b for b in (ordered or []) if b.get("notes")]
        if not sess_blocks:
            return ""

        # ---- MAP: per-session candidate extraction (parallel) ----
        map_results = await asyncio.gather(
            *[self._map_session_candidates(query, b) for b in sess_blocks],
            return_exceptions=True,
        )
        candidates: list[dict] = []
        for b, res in zip(sess_blocks, map_results):
            if isinstance(res, Exception) or not res:
                continue
            dt = parse_date(b.get("date_str", ""))
            for c in res:
                c["date_str"] = b.get("date_str", "")
                c["date"] = dt
                candidates.append(c)
        if not candidates:
            return ""

        money = kind.money
        cand_text = self._render_candidates(candidates, now_dt)

        # ---- TWO INDEPENDENT FILTERS over the SAME candidate pool ----
        # PATH A (holistic REDUCE): decide the whole kept list in one judgment.
        # PATH B (per-item ADJUDICATE): emit a KEEP/DROP verdict for EVERY
        # candidate. The two reasoning modes have different error profiles
        # (A over/under-counts holistically; B must rule on each line), so
        # cross-checking them turns run-to-run counting noise into an explicit,
        # resolvable signal. Re-running ONE mode (self-consistency, harness
        # 6d1d628678ef) did NOT help — correlated samples agree on the same wrong
        # answer; two DIFFERENT modes do not.
        reduce_out, adj_out = await asyncio.gather(
            self._safe_ask(AGG_REDUCE_SYS, build_reduce_user(query, qd, cand_text)),
            self._safe_ask(
                AGG_ADJUDICATE_SYS, build_adjudicate_user(query, qd, cand_text)
            ),
            return_exceptions=True,
        )
        if isinstance(reduce_out, Exception):
            reduce_out = ""
        if isinstance(adj_out, Exception):
            adj_out = ""

        # PATH A result: parse the holistic kept list.
        items_a: list[tuple[str, float]] = []
        if reduce_out and not _is_none(reduce_out):
            items_a = parse_enumeration(reduce_out)

        # PATH B result: KEPT indices -> (label, qty) read back from the ORIGINAL
        # MAP candidates, so the summed quantities are the map's verbatim numbers
        # and never depend on the adjudicator re-typing a value.
        items_b: list[tuple[str, float]] = []
        if adj_out:
            for idx in parse_adjudication(adj_out, len(candidates)):
                c = candidates[idx - 1]
                items_b.append((c.get("label", ""), float(c.get("qty", 1.0) or 0.0)))

        return self._compose_agg_block(items_a, items_b, money)

    # ---- aggregation block composition (dual-method reconciliation) ----
    def _compose_agg_block(self, items_a, items_b, money) -> str:
        have_a, have_b = bool(items_a), bool(items_b)
        if not have_a and not have_b:
            return ""
        # Only one method produced a usable list -> fall back to the proven
        # single-pass block (byte-identical wording to the 0.80/0.82 record).
        if have_a != have_b:
            return self._single_agg_block(items_a if have_a else items_b, money)

        ka, sa = len(items_a), sum(q for _, q in items_a)
        kb, sb = len(items_b), sum(q for _, q in items_b)
        if ka == kb and abs(sa - sb) < 1e-6:
            # Agreement is a real confidence signal (two independent filters).
            return self._single_agg_block(items_a, money, cross_checked=True)
        return self._disagree_agg_block(items_a, items_b, money)

    def _single_agg_block(self, items, money, cross_checked: bool = False) -> str:
        k = len(items)
        s = sum(qty for _, qty in items)
        if cross_checked:
            head = (
                "ENUMERATION (matching items found across your past conversations "
                "and CROSS-CHECKED by two independent methods — an overall filter "
                "and an item-by-item check — which AGREED on this list; the counts "
                "below were computed exactly in code. This agreement makes the list "
                "high-confidence: use the count/total below as the answer unless a "
                "specific item is flatly contradicted by the conversations):"
            )
        else:
            head = (
                "ENUMERATION (matching items found across your past conversations; "
                "each was located in the specific conversation it came from, then "
                "filtered and de-duplicated, and the counts below were computed "
                "exactly in code — your job is only to make sure the list is "
                "complete and correct, with no item that fails the question's "
                "condition):"
            )
        lines = [head]
        for label, qty in items:
            lines.append(f"- {label} (quantity: {fmt_qty(qty, money)})")
        lines.append(f"Number of distinct items listed: {k}.")
        lines.append(f"Total of all quantities: {fmt_qty(s, money)}.")
        return "\n".join(lines)

    def _disagree_agg_block(self, items_a, items_b, money) -> str:
        ka, sa = len(items_a), sum(q for _, q in items_a)
        kb, sb = len(items_b), sum(q for _, q in items_b)
        lines = [
            "ENUMERATION — two independent methods counted the matching items and "
            "DISAGREE, so at least one item is borderline. Both located each item "
            "in the specific conversation it came from; decide the final list "
            "yourself using the two candidate lists below.",
            "",
            f"Method 1 (overall filter) kept {ka} item(s), total {fmt_qty(sa, money)}:",
        ]
        for label, qty in items_a:
            lines.append(f"- {label} (quantity: {fmt_qty(qty, money)})")
        lines.append("")
        lines.append(
            f"Method 2 (item-by-item check) kept {kb} item(s), total {fmt_qty(sb, money)}:"
        )
        for label, qty in items_b:
            lines.append(f"- {label} (quantity: {fmt_qty(qty, money)})")
        lines.append("")
        lines.append(
            "To settle the final count/total, check every item that appears in "
            "only ONE of the two lists against the question: (a) is it really a "
            "thing of the kind asked about, not merely similar? (b) did the user "
            "ACTUALLY do / have it (not just plan it or miss it), unless the "
            "question says to include it? (c) is its date inside any time window "
            "the question gives? (d) is it the same real-world thing as another "
            "listed item, so it should be counted only once? Keep only items that "
            "pass all of these, then state that count/total."
        )
        return "\n".join(lines)

    async def _map_session_candidates(self, query: str, block: dict) -> list[dict]:
        notes = block.get("notes", []) or []
        notes_text = "\n".join(f"- {ln}" for ln in notes if str(ln).strip())
        if not notes_text.strip():
            return []
        out = await self._safe_ask(
            AGG_MAP_SYS, build_map_user(query, block.get("date_str", ""), notes_text)
        )
        if not out or _is_none(out):
            return []
        return parse_candidates(out)

    def _render_candidates(self, candidates: list[dict], now_dt) -> str:
        """Render candidates as a numbered, structured list shared VERBATIM by
        both the holistic REDUCE and the per-item ADJUDICATE, so the index each
        emits refers to the same candidate (parse_adjudication maps indices back).
        """
        lines: list[str] = []
        for i, c in enumerate(candidates, 1):
            delta = fmt_time_delta(c.get("date"), now_dt)
            dtail = f" [{delta}]" if delta else ""
            qty = c.get("qty", 1.0)
            qty_s = f"{int(round(qty)):,}" if abs(qty - round(qty)) < 1e-9 else f"{qty:,.2f}"
            why = str(c.get("why", "")).strip()
            why_s = f" | WHY: {why}" if why else ""
            lines.append(
                f"{i}. {c.get('label', '')} | QTY: {qty_s} | "
                f"STATUS: {c.get('status', 'other')} | "
                f"DATE: {c.get('date_str', '') or '?'}{dtail}{why_s}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # SCAFFOLD: difference (LLM reads two values -> Python subtracts)
    # ------------------------------------------------------------------
    async def _difference(self, query: str, facts: str, qd: str, now_dt) -> str:
        out = await self._safe_ask(DIFF_SYS, build_diff_user(query, qd, facts))
        if not out or _is_none(out):
            return ""
        pair = parse_diff(out)
        if not pair:
            return ""
        a, al, b, bl = pair
        diff = abs(a - b)
        # A==B is ALWAYS an extraction failure for a "how much more / earlier /
        # faster / exceeded by" question (the answer is never 0): it means the
        # model read the same quantity into both slots (e.g. the efc3f7c2 failure,
        # 6:30AM parsed as 630 for BOTH values -> diff 0, which then overrode a
        # correct downstream "30 minutes"). Suppress the block so deliberation
        # reasons from the evidence instead of parroting a spurious 0.
        if diff == 0:
            return ""
        return (
            "COMPUTED DIFFERENCE (the two values below were read from your past "
            "conversations and subtracted in code — confirm these are the two "
            "values the question compares, then use this difference as the answer):"
            f"\n- {al}: {fmt_qty(a)}\n- {bl}: {fmt_qty(b)}\n"
            f"=> Difference (larger minus smaller): {fmt_qty(diff)}"
        )

    # ------------------------------------------------------------------
    # SCAFFOLD: temporal date resolution (LLM identifies dates -> Python math)
    # ------------------------------------------------------------------
    async def _resolve_dates(self, query: str, facts: str, qd: str, now_dt) -> str:
        user = (
            f"Current date: {qd or '(unknown)'}\n"
            f"Question: {query}\n\nEvidence:\n{facts}\n\n"
            "Event dates (one per line, 'EVENT: <description> | DATE: <YYYY/MM/DD>'):"
        )
        out = await self._safe_ask(_DATE_SYS, user)
        if not out:
            return ""
        events = parse_resolved_events(out)
        if not any(dt is not None for _, dt in events):
            return ""
        return build_resolved_block(events, now_dt)

    # ------------------------------------------------------------------
    # SCAFFOLD: newest-value resolver (LLM lists dated values -> Python picks newest)
    # ------------------------------------------------------------------
    async def _latest_value(self, query: str, facts: str, qd: str) -> str:
        user = (
            f"Current date: {qd or '(unknown)'}\n"
            f"Question: {query}\n\nEvidence:\n{facts}\n\n"
            "Dated mentions of the value asked about (one per line, "
            "'VALUE: <value> | DATE: <YYYY/MM/DD>', earliest first), or NONE:"
        )
        out = await self._safe_ask(LATEST_VALUE_SYS, user)
        if not out or _is_none(out):
            return ""
        values = parse_valued_events(out)
        if not values:
            return ""
        return build_latest_value_block(values)

    # ------------------------------------------------------------------
    # SCAFFOLD: stated-duration reader
    # ------------------------------------------------------------------
    async def _stated_duration(self, query: str, facts: str) -> str:
        user = (
            f"Question: {query}\n\nEvidence:\n{facts}\n\n"
            "Stated duration (one 'DURATION: ... | FROM: ...' line, or NONE):"
        )
        out = await self._safe_ask(STATED_DUR_SYS, user)
        if not out:
            return ""
        return parse_stated_duration(out)

    # ------------------------------------------------------------------
    # DELIBERATION
    # ------------------------------------------------------------------
    async def _deliberate(self, query: str, qd: str, evidence_ctx: str) -> tuple[str, str]:
        user = (
            f"Current date: {qd or '(unknown)'}\n"
            f"Question: {query}\n\n"
            f"Evidence recalled from past conversations:\n{evidence_ctx}\n\n"
            "Now follow the format (REASONING, ANSWER, QUOTE):"
        )
        out = await self._safe_ask(DELIB_SYS, user, temperature=0.1)
        if not out:
            return ("", "")
        return parse_deliberation(out)

    async def _safe_ask(self, system_prompt: str, user: str, temperature=None) -> str:
        try:
            agent = Agent(system_prompt=system_prompt)
            out = await agent.ask(user, temperature=temperature)
        except Exception:
            return ""
        if isinstance(out, str):
            return out.strip()
        return str(out or "").strip()


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _is_none(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True
    collapsed = _NONE_RE.sub("", t)
    if collapsed == "none":
        return True
    if len(t) <= 40 and (
        t.startswith("none")
        or "no relevant" in t
        or "not relevant" in t
        or "nothing relevant" in t
        or "entirely unrelated" in t
        or t in ("n/a", "na", "no.", "no")
    ):
        return True
    return False


def _truncate_chars(text: str, cap: int) -> str:
    text = text or ""
    if len(text) <= cap:
        return text
    return text[:cap].rstrip() + " …"


def _truncate_tokens(text: str, tok_cap: int) -> str:
    return _truncate_chars(text or "", tok_cap * 4)
