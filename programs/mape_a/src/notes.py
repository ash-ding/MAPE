"""Build-time atomic memory notes — the core departure of AtomicLedgerMemo.

WHY THIS MODULE EXISTS
----------------------
The prior frontier (1718/0a8a, 0.78) extracts facts at QUERY time: it retrieves
whole sessions, then runs a per-session "extract facts relevant to THIS question"
map-reduce on every query. Trace analysis of that line showed two structural
costs of query-time extraction:

  1. RELEVANCE DROP. "Facts relevant to <this question>" silently discards oblique
     facts — the diagnosed preference collapse (it returned NONE on the
     French-podcast session for "recommend cultural events"). The prior line
     patched this with a SEPARATE brittle advice path.
  2. RE-EXTRACTION NOISE. The same session is re-read and re-enumerated on every
     query, so the fact/item set the answer sees is different run-to-run — the
     dominant noise source the analysts flagged on counting ("the LLM enumeration
     produces a different item list each run").

This module moves extraction to BUILD time: each session is read ONCE into a set
of atomic, self-contained, comprehensive memory notes, cached on `self`. Notes
are then the retrieval unit (finer than a session) and, once a session is judged
relevant, ALL of its notes are shown — no second relevance filter, so oblique
facts survive structurally rather than via a special case. The item set a
counting/temporal question sees is fixed at build time, which removes the
re-extraction variance.

Every prompt calls the run's unified QA model via core.llm.Agent (model unset).
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Note:
    """One atomic, self-contained memory note distilled from a session."""
    text: str
    session_idx: int
    session_id: str
    date_str: str
    date: Optional[_dt.datetime]
    note_idx: int  # position within the global note list


# Comprehensive, query-AGNOSTIC extraction. It captures everything that could be
# asked about later (both the user's facts/preferences AND the assistant's
# answers), so retrieval never has to re-decide relevance per query and oblique
# facts are never dropped at write time.
NOTE_EXTRACT_SYS = (
    "You are building a durable memory of a user from ONE past conversation "
    "between the user and the assistant (you). Extract a list of concise, "
    "self-contained memory notes capturing everything that could help answer the "
    "user's FUTURE questions about their own history. You do NOT answer any "
    "question.\n"
    "Write ONE note per line. Each note MUST:\n"
    "- be a complete standalone statement, understandable on its own without the "
    "rest of the conversation (no dangling pronouns);\n"
    "- quote specific details verbatim: names, numbers, amounts, dates, places, "
    "titles, brands, products, and the user's exact wording where it matters;\n"
    "- preserve any time expression that appears in the text ('last week', 'for "
    "two weeks', 'since March', 'a month ago') and any quantity or amount.\n"
    "Capture ALL of the following that appear:\n"
    "- facts the USER stated about themselves: situations, possessions, "
    "relationships, family, health, work, studies, finances, numbers;\n"
    "- the USER's preferences, tastes, likes and dislikes, hobbies, interests, "
    "goals, and plans (keep even loosely-relevant personal details);\n"
    "- events, actions, purchases, trips, and milestones, together with WHEN they "
    "happened or are planned;\n"
    "- notable information, recommendations, names, titles, or answers the "
    "ASSISTANT gave the user (these are frequently asked about later).\n"
    "Rules:\n"
    "- Be INCLUSIVE and comprehensive: prefer keeping a concrete detail over "
    "dropping it.\n"
    "- Distinguish things the user ACTUALLY did/has from things they only "
    "considered, planned, wished for, or asked about — say which, in the note.\n"
    "- Report ONLY what is actually present in this conversation. Never invent or "
    "infer beyond the text, do NOT compute totals or durations, and do NOT merge "
    "separate facts into one line.\n"
    "- Output the single word NONE only if the conversation contains nothing "
    "worth remembering about the user."
)


def build_extract_user_prompt(date_label: str, conversation_text: str) -> str:
    return (
        f"Conversation (took place on {date_label}):\n{conversation_text}\n\n"
        "Memory notes (one per line, or NONE):"
    )
