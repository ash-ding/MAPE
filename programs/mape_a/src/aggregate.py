"""Per-session map-reduce aggregation — the targeted rebuild of the 0.80 record's
counting/summation path.

WHY THIS MODULE EXISTS (trace-verified, record 23d91358a4a1)
-----------------------------------------------------------
Every multi-session failure in the 0.80 record is an aggregation question, and
retrieval recall is PERFECT on all of them (all gold sessions are retrieved, at
top ranks). The loss is entirely downstream, in a SINGLE-PASS enumerator that
reads a 4,200-token digest of ~290 verbose notes and tries, in one call, to find
all matching items, judge each item's modality, dedup them, and apply the time
window. Three failure modes were verified in the traces:

  * SIGNAL DILUTION. The relevant items (e.g. 5 model-kit mentions) are buried in
    hundreds of adjacent technique notes; the single pass picks up 3 and misses 2.
  * TRUNCATION. The digest is chronological and cut at 4,200 tokens, so the LAST
    session's item is dropped (the Spitfire completed in the final session).
  * MODALITY ERROR. One pass over raw notes mis-judges whether each item was
    actually done. On "graduation ceremonies attended", the enumerator counted the
    one the user MISSED ("feeling guilty about missing Jack's graduation") and
    dropped the two attended ones ("inspired by Rachel's ... graduation").
  * DOUBLE-COUNT. The same amount appearing in two notes ($500 raised, stated in
    two different notes of one session) was summed twice -> $8,750 vs gold $3,750.

This module replaces the single pass with a three-stage pipeline that gives each
stage one job:

  MAP   (`AGG_MAP_SYS`): each SELECTED session is read on its OWN, with only its
        notes in context, and asked only to list candidate items in the question's
        category, each tagged with QTY + STATUS (did / planned / missed) + a
        verbatim WHY. Per-session focus kills dilution and truncation: the final
        session gets a dedicated call, so its item is never cut, and 18 notes are
        never diluted by 290.
  REDUCE(`AGG_REDUCE_SYS`): the small candidate list (now ~5-20 structured lines,
        each carrying a pre-computed "how long before the current date" gloss) is
        filtered in ONE call — category, modality, time window, and dedup of true
        duplicates. Reasoning over a short structured list is far more reliable
        than over 290 raw notes.
  PYTHON: the kept list is counted / summed exactly in code.

A DIFFERENCE sub-path (`DIFF_SYS`) handles "how much more/faster/exceeded by"
questions, which are not sums or counts: it reads the two comparable values and
subtracts them in Python (targets the marathon "exceeded my target by 12 min"
failure, where the single enumerator / deliberate gave the finish time instead).

Every prompt calls the run's unified QA model via core.llm.Agent (model unset).
The emitted blocks remain NON-AUTHORITATIVE so the downstream deliberation can
still override them from the full evidence (the step-18 lesson: baking the number
and skipping deliberation removed its rescue of mis-routed questions).
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Difference detection (a comparison, not a count/sum)
# ---------------------------------------------------------------------------
# Specific comparison cues only — deliberately EXCLUDES "save/spend/total" so a
# genuine SUM question ("how much did I save in total") stays on the sum path.
DIFF_RE = re.compile(
    r"\b(how much (more|less|further|farther|higher|lower|faster|slower|earlier|"
    r"later|older|younger|longer|shorter|bigger|smaller|cheaper|greater)\b"
    r"|how many (more|fewer|less)\b"
    r"|\bexceed(ed|s)?\b|\bby how much\b|\bdifference between\b"
    r"|over my (target|goal|budget|limit)|under my (target|goal|budget|limit))"
)


def is_difference(query: str) -> bool:
    return bool(DIFF_RE.search((query or "").lower()))


# ---------------------------------------------------------------------------
# MAP — per-session candidate extraction (recall-oriented)
# ---------------------------------------------------------------------------
AGG_MAP_SYS = (
    "You are scanning the memory notes from ONE past conversation to find items "
    "that could help answer a COUNT or TOTAL question about the user's history. "
    "You do NOT answer the question and you do NOT count or add anything up.\n"
    "First read what CATEGORY of thing the question is about. Then list every item "
    "in THESE notes that plausibly belongs to that category. Be INCLUSIVE at this "
    "step — a later step decides what to keep; your job is not to miss anything.\n"
    "Output ONE line per item, in exactly this format:\n"
    "ITEM: <short label> | QTY: <number> | STATUS: <did|planned|missed|other> | "
    "WHY: <the verbatim phrase from the notes that shows this item>\n"
    "STATUS meanings:\n"
    "- did     = the user ACTUALLY did / has / acquired / attended / completed / "
    "paid / raised it;\n"
    "- planned = the user only planned, intends, is considering, wishes, or asked "
    "about it (it has not happened);\n"
    "- missed  = the user did NOT do it (skipped, missed, cancelled, could not);\n"
    "- other   = unclear from the notes.\n"
    "QTY rules:\n"
    "- For a single thing or a single occurrence, QTY is 1.\n"
    "- For a measured amount (money, weight, distance, calories, or a number of "
    "times something happened on ONE occasion), QTY is that amount — digits only, "
    "no units, taken verbatim from the notes. Never invent a number.\n"
    "- If the same item is stated more than once in these notes, output it ONCE.\n"
    "If no item in these notes belongs to the question's category, output exactly: "
    "NONE"
)


def build_map_user(query: str, date_label: str, notes_text: str) -> str:
    return (
        f"Question (for context only — do NOT answer it): {query}\n"
        f"These notes are from the conversation on {date_label or '(date unknown)'}.\n\n"
        f"Notes:\n{notes_text}\n\n"
        "Candidate items (one per line in the required format, or NONE):"
    )


# ---------------------------------------------------------------------------
# REDUCE — filter + dedup over the small structured candidate list
# ---------------------------------------------------------------------------
AGG_REDUCE_SYS = (
    "You are given a COUNT or TOTAL question and a list of CANDIDATE items gathered "
    "from the user's past conversations. Each candidate carries a quantity, a "
    "status, its date, and how long before the current date it occurred. Produce "
    "the FINAL list of items that satisfy EVERY part of the question's condition. "
    "You do NOT output the count or the total — only the kept items.\n"
    "Apply ALL of these filters:\n"
    "- CATEGORY: keep only items that genuinely match what the question asks "
    "about; drop items that merely look similar.\n"
    "- MODALITY: if the question asks what the user ACTUALLY did / has / acquired / "
    "attended, DROP items whose status is 'planned' or 'missed'. BUT: if the "
    "question explicitly says to include something (e.g. 'including the one I set "
    "up for my friend', 'counting the free trial'), keep that one regardless. If "
    "the condition is a DISJUNCTION (e.g. 'worked on OR bought', 'tried or "
    "cooked'), keep an item that satisfies ANY one of the alternatives.\n"
    "- TIME WINDOW: if the question states a window (e.g. 'in the last two months', "
    "'from July to October', 'this year', 'in the past three months'), keep only "
    "items whose date falls inside it. Use the 'before the current date' gloss on "
    "each line to decide; do not do calendar math yourself.\n"
    "- DEDUPLICATE true duplicates: if two or more candidates clearly describe the "
    "SAME single real-world thing or the SAME single occasion mentioned more than "
    "once (e.g. the same $500 raised, reported in two notes), keep only ONE of "
    "them. Do NOT merge separate things or separate occasions, even if similar.\n"
    "Output one line per kept item, in exactly this format:\n"
    "- <short label> (<date>) | QTY: <number>\n"
    "ALWAYS include the item's date in the label so that two separate occasions of "
    "a similar thing (e.g. rollercoaster rides on two different days) stay as two "
    "DISTINCT lines and are not accidentally merged. Copy each kept item's QTY "
    "unchanged from its candidate line. If nothing qualifies, output exactly: NONE"
)


def build_reduce_user(query: str, qd: str, candidates_text: str) -> str:
    return (
        f"Current date: {qd or '(unknown)'}\n"
        f"Question: {query}\n\n"
        f"Candidate items:\n{candidates_text}\n\n"
        "Final kept items (one per line as '- <label> | QTY: <number>', or NONE):"
    )


# ---------------------------------------------------------------------------
# ADJUDICATE — a SECOND, independent filter that runs item-by-item.
#
# WHY A SECOND METHOD (trace-verified, record 39403f52ebc6)
# --------------------------------------------------------
# The single holistic REDUCE above decides the whole kept-list in one judgment,
# and its errors are directional and consistent: it OVER-counts (keeps a
# substrate "Amazonia" as a "tank", sums the same $500 twice -> $8,750 vs $3,750)
# on some questions and UNDER-counts (drops the third attended graduation -> 2 vs
# 3) on others. Re-running the SAME holistic pass (self-consistency, harness
# 6d1d628678ef) did not help because the samples share one evidence block and one
# reasoning mode, so they agree on the same wrong answer.
#
# ADJUDICATE is a structurally DIFFERENT reasoning mode over the same candidate
# pool: instead of producing a list, it forces a KEEP/DROP VERDICT for EVERY
# candidate line with the specific condition it passes or fails. A per-line
# verdict has a different error profile from a holistic list (it is harder to
# silently drop an item you must rule on, and harder to wave through "Amazonia"
# when you must assert it IS a tank). The two methods are then cross-checked in
# `_aggregate`: AGREEMENT is a real confidence signal (two independent filters,
# not two correlated samples); DISAGREEMENT flags that an item is borderline and
# hands deliberation BOTH candidate answers with their item lists to resolve.
#
# Crucially, ADJUDICATE returns INDICES, not re-typed quantities: the kept QTYs
# are read back from the original MAP candidates in Python, so the sum path never
# depends on the adjudicator re-copying a number correctly.
# ---------------------------------------------------------------------------
AGG_ADJUDICATE_SYS = (
    "You are given a COUNT or TOTAL question and a NUMBERED list of candidate "
    "items gathered from the user's past conversations. Each candidate carries a "
    "quantity, a status (did/planned/missed/other), its date, and how long before "
    "the current date it occurred. Your job is to rule on EVERY candidate, one at "
    "a time, deciding whether it belongs in the final answer.\n"
    "Output EXACTLY one verdict line for EACH numbered candidate, in order, in "
    "this format:\n"
    "<index>. KEEP | REASON: <why it qualifies>\n"
    "   or\n"
    "<index>. DROP | REASON: <the ONE condition it fails>\n"
    "Decide KEEP only if the candidate passes ALL of these, and DROP naming the "
    "FIRST that fails:\n"
    "- CATEGORY: it genuinely IS a thing of the kind the question asks about — not "
    "merely related or similar (e.g. a fish-tank substrate or decoration is NOT a "
    "tank; a store you browsed is NOT a thing you bought). If unsure it truly "
    "belongs to the category, DROP it (reason: not-in-category).\n"
    "- MODALITY: if the question asks what the user ACTUALLY did / has / acquired / "
    "attended / completed, DROP candidates whose status is 'planned' or 'missed' "
    "(reason: planned-or-missed). EXCEPTION: if the question explicitly says to "
    "include one (e.g. 'including the one I set up for my friend'), KEEP that one. "
    "If the condition is a DISJUNCTION ('bought OR made', 'tried or cooked'), KEEP "
    "an item satisfying ANY alternative.\n"
    "- TIME WINDOW: if the question gives a window ('in the past two months', "
    "'from July to October', 'this year'), DROP candidates dated outside it "
    "(reason: outside-window). Use the 'before the current date' gloss on each "
    "line; do NOT do calendar math yourself.\n"
    "- DUPLICATE: if a candidate describes the SAME single real-world thing or the "
    "SAME single occasion as an EARLIER candidate you already marked KEEP, DROP "
    "the later one (reason: duplicate-of-#<that index>). Treat differently-worded "
    "mentions of one thing (e.g. 'community tank' and '20-gallon community tank', "
    "or the same $500 reported in two notes) as the SAME thing. Do NOT merge "
    "genuinely separate things or separate occasions on different dates.\n"
    "Rule on every candidate. Do not output a count or total — only the verdicts."
)


def build_adjudicate_user(query: str, qd: str, candidates_text: str) -> str:
    return (
        f"Current date: {qd or '(unknown)'}\n"
        f"Question: {query}\n\n"
        f"Numbered candidates:\n{candidates_text}\n\n"
        "Your verdicts (one '<index>. KEEP|DROP | REASON: ...' line per candidate):"
    )


_ADJ_LINE_RE = re.compile(
    r"^\s*(?:[-*•]\s*)?(\d+)\s*[.)]?\s*[:|-]?\s*(KEEP|DROP)\b", re.IGNORECASE
)


def parse_adjudication(text: str, n_candidates: int) -> List[int]:
    """Parse the per-item adjudication into the set of KEPT candidate indices.

    Returns 1-based indices (matching the numbering shown to the model) that were
    marked KEEP. Robust to extra prose: only lines whose leading token is
    '<int>. KEEP/DROP' are read, duplicates collapse, out-of-range indices are
    ignored.
    """
    kept: List[int] = []
    seen = set()
    for raw in (text or "").splitlines():
        m = _ADJ_LINE_RE.match(raw)
        if not m:
            continue
        try:
            idx = int(m.group(1))
        except ValueError:
            continue
        if idx < 1 or idx > n_candidates or idx in seen:
            continue
        seen.add(idx)
        if m.group(2).upper() == "KEEP":
            kept.append(idx)
    return kept


# ---------------------------------------------------------------------------
# DIFFERENCE — extract two comparable values, subtract in Python
# ---------------------------------------------------------------------------
DIFF_SYS = (
    "You read evidence from the user's past conversations and identify the TWO "
    "comparable numeric values needed to answer a 'how much more / less / faster / "
    "earlier / exceeded by' question. You do NOT compute the difference yourself.\n"
    "Work out the two quantities the question compares — for example a target time "
    "and an actual time, a planned amount and an actual amount, a budget and the "
    "amount spent, two prices. Express BOTH values in the SAME unit, and in the "
    "unit the question asks for (e.g. if it asks 'how many minutes', give both in "
    "minutes).\n"
    "Output EXACTLY two lines, nothing else:\n"
    "VALUE_A: <number> | LABEL: <what A is>\n"
    "VALUE_B: <number> | LABEL: <what B is>\n"
    "Order them so that A is the larger / actual / later value and B is the "
    "smaller / target / earlier one, so that A minus B is the positive answer. Use "
    "digits only, no units. If both values are not clearly present in the evidence, "
    "output exactly: NONE"
)


def build_diff_user(query: str, qd: str, facts: str) -> str:
    return (
        f"Current date: {qd or '(unknown)'}\n"
        f"Question: {query}\n\nEvidence:\n{facts}\n\n"
        "The two values (two lines in the required format, or NONE):"
    )


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------
_NONE_RE = re.compile(r"[^a-z]")
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")

_ITEM_LABEL_RE = re.compile(r"ITEM\s*[:=]\s*(.*?)(?:\s*\||$)", re.IGNORECASE)
_QTY_RE = re.compile(r"\bQTY\b\s*[:=]?\s*\$?\s*(\d[\d,]*(?:\.\d+)?)", re.IGNORECASE)
_STATUS_RE = re.compile(r"\bSTATUS\b\s*[:=]?\s*([a-zA-Z]+)", re.IGNORECASE)
_WHY_RE = re.compile(r"\bWHY\b\s*[:=]?\s*(.*)$", re.IGNORECASE)

_VALID_STATUS = {"did", "planned", "missed", "other"}


def _is_none_line(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True
    return _NONE_RE.sub("", t) == "none"


def parse_candidates(text: str) -> List[dict]:
    """Parse the per-session MAP output into structured candidate dicts.

    Each dict: {label, qty (float), status (str), why (str)}. Robust to missing
    fields — an unparseable line that still has an ITEM label is kept with
    defaults so recall is not sacrificed to format strictness.
    """
    out: List[dict] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if low in ("none", "n/a", "na") or _is_none_line(line):
            continue
        # label
        m_item = _ITEM_LABEL_RE.search(line)
        if m_item:
            label = m_item.group(1).strip()
        else:
            # tolerate a bullet line without the ITEM: tag
            stripped = _BULLET_RE.sub("", line)
            label = stripped.split("|", 1)[0].strip()
        if not label:
            continue
        # qty
        qty = 1.0
        m_qty = _QTY_RE.search(line)
        if m_qty:
            try:
                qty = float(m_qty.group(1).replace(",", ""))
            except ValueError:
                qty = 1.0
        # status
        status = "other"
        m_st = _STATUS_RE.search(line)
        if m_st:
            st = m_st.group(1).strip().lower()
            if st in _VALID_STATUS:
                status = st
        # why
        why = ""
        m_why = _WHY_RE.search(line)
        if m_why:
            why = m_why.group(1).strip()
        out.append({"label": label, "qty": qty, "status": status, "why": why})
    return out


_VAL_A_RE = re.compile(
    r"VALUE_A\s*[:=]\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)(?:\s*\|\s*LABEL\s*[:=]\s*(.*))?",
    re.IGNORECASE,
)
_VAL_B_RE = re.compile(
    r"VALUE_B\s*[:=]\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)(?:\s*\|\s*LABEL\s*[:=]\s*(.*))?",
    re.IGNORECASE,
)


def parse_diff(text: str) -> Optional[Tuple[float, str, float, str]]:
    """Parse the difference output into (a_val, a_label, b_val, b_label)."""
    if not text or _is_none_line(text):
        return None
    a = b = None
    al = bl = ""
    for raw in text.splitlines():
        line = raw.strip()
        ma = _VAL_A_RE.search(line)
        if ma:
            try:
                a = float(ma.group(1).replace(",", ""))
                al = (ma.group(2) or "").strip()
            except ValueError:
                pass
            continue
        mb = _VAL_B_RE.search(line)
        if mb:
            try:
                b = float(mb.group(1).replace(",", ""))
                bl = (mb.group(2) or "").strip()
            except ValueError:
                pass
    if a is None or b is None:
        return None
    return (a, al or "value A", b, bl or "value B")
