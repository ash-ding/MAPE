"""Retrieval primitives for the extractive-recall memory.

Pure helpers + data structures used by memo.ExtractiveRecallMemo. No LLM calls
live here; embedding/scoring is driven from the memo hooks so each cost lands in
the right phase (build vs retrieve vs compose).

Carried over verbatim from the session-centric baseline (8ceea48e8ceb): a
session->rounds index so retrieval can expand a matched session to all of its
rounds, and a deterministic time-delta formatter so the weak QA model reads
pre-computed "how long ago" arithmetic instead of doing it itself.
"""
from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Tokenisation (for BM25)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# A small, conservative stopword list. BM25's idf already down-weights
# ubiquitous terms; an aggressive list risks dropping discriminative query
# words.
_STOPWORDS = frozenset(
    """
    a an the and or but if then else of to in on at for with without from by
    is are was were be been being am do does did doing have has had having
    this that these those i you he she it we they them my your his her its our
    their me him us as so not no yes can could will would should shall may might
    about into over under again more most some any all each both few other such
    """.split()
)


def tokenize(text: str) -> List[str]:
    """Lowercase alnum tokens with light stopword removal."""
    if not text:
        return []
    toks = _TOKEN_RE.findall(text.lower())
    out = [t for t in toks if t not in _STOPWORDS]
    # If stopword removal nuked everything (very short query), keep raw tokens.
    return out or toks


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

_DATE_RE = re.compile(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2}).*?(?:(\d{1,2}):(\d{2}))?")


def parse_date(date_str: str) -> Optional[_dt.datetime]:
    """Parse LongMemEval dates like '2023/05/20 (Sat) 06:13' -> datetime.

    Returns None when the string does not carry a parseable Y/M/D prefix.
    """
    if not date_str:
        return None
    m = _DATE_RE.match(date_str.strip())
    if not m:
        return None
    try:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        hh = int(m.group(4)) if m.group(4) else 0
        mm = int(m.group(5)) if m.group(5) else 0
        return _dt.datetime(y, mo, d, hh, mm)
    except (ValueError, TypeError):
        return None


def fmt_time_delta(sess_dt: Optional[_dt.datetime],
                   now_dt: Optional[_dt.datetime]) -> str:
    """Pre-computed, human-readable "how long ago" string.

    The weak QA model cannot reliably do day/month arithmetic (diagnosed on the
    hybrid baseline's 14 temporal failures), so we do it for it. The judge
    forgives off-by-one on the requested unit, so computing days / weeks /
    months deterministically is safe and strictly helpful.
    """
    if sess_dt is None or now_dt is None:
        return ""
    days = (now_dt.date() - sess_dt.date()).days
    if days == 0:
        return "earlier today"
    if days < 0:
        return f"{abs(days)} days in the future"
    out = f"{days} day{'s' if days != 1 else ''} ago"
    extras: List[str] = []
    if days >= 14:
        extras.append(f"~{round(days / 7)} weeks")
    if days >= 60:
        extras.append(f"~{round(days / 30.44)} months")
    if extras:
        out += " (" + ", ".join(extras) + ")"
    return out


# A sentinel far in the future so undated rounds sort to the END of a
# chronological ordering.
_FAR_FUTURE = _dt.datetime(9999, 1, 1)


# ---------------------------------------------------------------------------
# Round: one user turn plus the assistant turns that answer it
# ---------------------------------------------------------------------------

@dataclass
class Round:
    session_idx: int
    session_id: str
    date_str: str
    date: Optional[_dt.datetime]
    round_idx: int                      # position of the round within its session
    turns: List[Tuple[str, str]]        # [(role, content), ...]
    embed_text: str                     # text used for dense embedding
    tokens: List[str] = field(default_factory=list)  # BM25 tokens


_MSG_EMBED_CAP = 2000   # chars per message fed to the embedder / BM25
# Show at least as much as we matched on: a fact in the 1400-2000 char window
# otherwise gets matched (embed cap) but clipped out of the shown text (old
# 1400 show cap), so the model never sees the evidence that caused retrieval.
_MSG_SHOW_CAP = 2000    # chars per message shown in the composed context


def _clip(text: str, cap: int) -> str:
    text = text or ""
    if len(text) <= cap:
        return text
    return text[:cap].rstrip() + " …"


def _role_label(role: str) -> str:
    r = (role or "").lower()
    if r == "user":
        return "User"
    if r == "assistant":
        return "Assistant"
    return role.capitalize() if role else "Message"


def build_rounds(sessions: List[Dict]) -> List[Round]:
    """Split each session into rounds.

    A round starts at a user message and absorbs the assistant (and any other)
    messages that follow it until the next user message. A session that opens
    with a non-user message yields a leading round for those turns.
    """
    rounds: List[Round] = []
    for s_idx, sess in enumerate(sessions or []):
        if not isinstance(sess, dict):
            continue
        sid = str(sess.get("session_id", f"session_{s_idx + 1:03d}"))
        date_str = str(sess.get("date", "") or "")
        date = parse_date(date_str)
        msgs = sess.get("messages", []) or []

        groups: List[List[Tuple[str, str]]] = []
        cur: List[Tuple[str, str]] = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role", "") or "")
            content = str(m.get("content", "") or "").strip()
            if not content:
                continue
            if role == "user" and cur:
                groups.append(cur)
                cur = []
            cur.append((role, content))
        if cur:
            groups.append(cur)

        for r_idx, turns in enumerate(groups):
            embed_text = "\n".join(
                f"{_role_label(role)}: {_clip(content, _MSG_EMBED_CAP)}"
                for role, content in turns
            )
            if not embed_text.strip():
                continue
            rounds.append(
                Round(
                    session_idx=s_idx,
                    session_id=sid,
                    date_str=date_str,
                    date=date,
                    round_idx=r_idx,
                    turns=turns,
                    embed_text=embed_text,
                    tokens=tokenize(embed_text),
                )
            )
    return rounds


# ---------------------------------------------------------------------------
# Scoring utilities
# ---------------------------------------------------------------------------

def rrf_fuse(rank_lists: List[List[int]], k: int = 60) -> Dict[int, float]:
    """Reciprocal-rank fusion over several ranked id lists.

    Each list is item ids in descending relevance. Returns id -> fused score.
    """
    scores: Dict[int, float] = {}
    for ranks in rank_lists:
        for pos, item in enumerate(ranks):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + pos + 1)
    return scores


def est_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) for context budgeting."""
    return max(1, len(text) // 4)


def render_round(r: Round) -> str:
    """Render a round's turns for the composed context (message-capped)."""
    return "\n".join(
        f"{_role_label(role)}: {_clip(content, _MSG_SHOW_CAP)}"
        for role, content in r.turns
    )
