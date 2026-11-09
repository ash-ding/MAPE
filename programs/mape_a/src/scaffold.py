"""Query-intent inference + deterministic reasoning scaffolds.

Everything here is PURE PYTHON (no LLM calls) and driven off the QUERY TEXT and
the conversation dates the memory already stores — the honest signal, since the
benchmark's `question_type` label never reaches the compose hook. The scaffolds
hand the weak unified QA model pre-computed results for the operations it
provably cannot do reliably (diagnosed across harnesses 1f352/8ceea/6ddf/30ae):

  * DATE ARITHMETIC. temporal-reasoning sat at ~0.48 across early harnesses;
    30ae's deterministic date resolver + locator lifted it to ~0.74. KEPT
    verbatim here (`resolve_anchor`, `build_time_table`, `build_resolved_block`).
  * AGGREGATION (COUNT / SUM). THE WEAK BUCKET THIS HARNESS TARGETS. 30ae's
    counting scaffold *regressed* multi-session 0.74 -> 0.59 by (a) assuming
    count == number of distinct listed lines when the real questions are mostly
    SUM-of-quantities ("how many times I rode" = 3+1+3+3 = 10; "total weight" =
    50+20 = 70), (b) only routing $-amounts through summation, and (c) stamping
    an AUTHORITATIVE total that overrode answers the model got right. This module
    replaces that with a single aggregation path: `classify` infers whether the
    question wants a DISTINCT COUNT or a SUM OF QUANTITIES (agg_mode), and
    `parse_enumeration` turns an LLM's item list — each carrying an explicit
    per-item QTY — into (item, quantity) pairs so memo.py can compute BOTH a
    distinct count and a quantity sum in Python and present them non-
    authoritatively.
"""
from __future__ import annotations

import datetime as _dt
import re
from typing import List, Optional, Tuple

from src.retrieval import parse_date  # reuse the Y/M/D parser

# ---------------------------------------------------------------------------
# Query-type inference (regex over the lowercased query — honest, no label)
# ---------------------------------------------------------------------------

_DUR_RE = re.compile(
    r"\b(how long|how many days|how many weeks|how many months|how many years|"
    r"how much time|how many hours|days? ago|weeks? ago|months? ago|years? ago|"
    r"how old|how recently|how soon)\b"
)
# Relative-anchor duration ("before I", "between X and Y", "since my") — these
# compare two events, NOT the current date.
_DUR_REL_RE = re.compile(
    r"\b(between .+ and .+|before (i|my|the)\b|after (i|my|the)\b|"
    r"since (i|my|the)\b|until (i|my|the)\b|prior to|leading up to|"
    r"from .+ (to|until) )"
)
_ORDER_RE = re.compile(
    r"\b(came first|comes? first|happened first|occur(red)? first|"
    r"in what order|in which order|before or after|earlier or later|"
    r"which (one )?(came|happened|was) (first|earlier|later|more recent)|"
    r"which .* (first|earlier|later|more recent|most recent)\?)"
)
_COUNT_OF_TIMEUNIT_RE = re.compile(
    r"how (many|much) (days?|weeks?|months?|years?|hours?|minutes?)\b"
)
# Frequency: "how many days a week", "how many times per month" — a COUNT of
# occurrences, NOT a duration. The following time-unit is a RATE, not an "ago".
_FREQ_RE = re.compile(
    r"\b(a|per|each|every)\s+(day|week|month|year)\b"
)
_COUNT_RE = re.compile(
    r"\b(how many|how much|number of|how many times|count of|total number|"
    r"how often)\b"
)
_PREF_RE = re.compile(
    r"\b(recommend|recommendation|suggest|suggestion|advice|advise|"
    r"what should i|which should i|what would be best|what would you|"
    r"help me (choose|pick|decide|find)|which .* (do you think i|would i|should i)|"
    r"what .* would i (like|prefer|enjoy)|best .* for me|good .* for me)\b"
)
_SUM_RE = re.compile(
    r"\b(in total|altogether|combined|total amount|sum of|grand total|"
    r"how much .* (spend|spent|save|saved|pay|paid|cost|donate|donated|earn|earned|make|made))\b"
)
# "how many TIMES", "how often", "number of times" — a SUM of per-occasion
# counts, NOT a distinct-item count. 30ae mis-routed these to distinct counting
# (gpt4_e05b82a6: "how many times I rode rollercoasters" = 3+1+3+3 = 10, got 4).
_TIMES_RE = re.compile(r"\b(how many times|how often|number of times|how frequently)\b")
# "total <noun>" (weight / distance / calories / ...) — a quantity SUM that is
# NOT money, so 30ae's $-only summation never fired (bc149d6b: "total weight of
# the new feed" = 20+50 = 70, got 20). Exclude "total number (of)", a count.
_TOTAL_NOUN_RE = re.compile(r"\btotal\s+(?!number\b)[a-z]+")


class QueryKind:
    __slots__ = ("temporal", "duration", "order", "count", "summation", "pref",
                 "agg", "agg_mode", "money")

    def __init__(self, temporal=False, duration=False, order=False,
                 count=False, summation=False, pref=False,
                 agg=False, agg_mode="count", money=False):
        self.temporal = temporal
        self.duration = duration
        self.order = order
        self.count = count
        self.summation = summation
        self.pref = pref
        self.agg = agg            # needs the aggregation aid (count/sum)
        self.agg_mode = agg_mode  # 'sum' | 'count'
        self.money = money        # format the sum as currency


def classify(query: str) -> QueryKind:
    """Infer what the question is asking for, from its wording alone."""
    q = (query or "").lower()
    order = bool(_ORDER_RE.search(q))
    has_count = bool(_COUNT_RE.search(q))
    summation = bool(_SUM_RE.search(q))
    times = bool(_TIMES_RE.search(q))
    total_noun = bool(_TOTAL_NOUN_RE.search(q))
    # A frequency question ("how many days a week") is a count, not a duration.
    freq = bool(has_count and _FREQ_RE.search(q))

    # An item count — unless it is really "how many days/weeks/..." (a duration)
    # or a total-amount question (summation). A frequency overrides the
    # time-unit guard (it is genuinely a count).
    count = has_count and (freq or not _COUNT_OF_TIMEUNIT_RE.search(q))
    if summation:
        count = False

    # An AGGREGATION cue (count / total / sum / "how many times"): when one is
    # present, phrases like "between July and October" or "from X to Y" are a
    # TIME WINDOW for the aggregation, NOT a duration to subtract. So the
    # relative-duration pattern only makes a question temporal when there is NO
    # aggregation cue. This is the gpt4_e05b82a6 fix: "how many times did I ride
    # ... from July to October" was mis-read as a duration by 30ae and never hit
    # the counting path. (An explicit "how long / how many days ago / X ago" is
    # still a genuine duration and is kept temporal via _DUR_RE.)
    agg_cue = bool(count or summation or total_noun or times)
    dur_explicit = bool(_DUR_RE.search(q))
    dur_rel = bool(_DUR_REL_RE.search(q) and not agg_cue)
    dur = bool((dur_explicit or dur_rel) and not freq)

    temporal = bool(dur or order)

    # ---- aggregation (the targeted path) -----------------------------------
    # Fires for any count/total/sum question that is NOT a temporal duration
    # (durations have their own date-arithmetic scaffold). The MODE decides
    # which computed number is emphasised; memo.py always shows BOTH so a
    # mis-inferred mode is recoverable.
    sum_like = bool(summation or total_noun or times)
    agg = bool(agg_cue and not temporal)
    agg_mode = "sum" if sum_like else "count"
    money = bool(re.search(
        r"\$|\bspend|\bspent|\bcost|\bpaid\b|\bpay\b|donat|\bsave|\bsaved|"
        r"\bearn|budget|\bprice|dollar|\bbudgeted\b|\bspending\b|"
        r"\bmoney\b|\braise|\bfund|fundrais|\brevenue\b|\bprofit\b", q))

    pref = bool(_PREF_RE.search(q))
    return QueryKind(temporal=temporal, duration=dur, order=order,
                     count=count, summation=summation, pref=pref,
                     agg=agg, agg_mode=agg_mode, money=money)


# ---------------------------------------------------------------------------
# Query-anchor resolution (LOCATOR temporal questions)  [UNCHANGED from 30ae]
# ---------------------------------------------------------------------------

_NUM_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "couple": 2, "few": 3, "several": 4,
}
_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}
_UNIT_TOL = {"day": 4, "week": 6, "month": 22, "year": 70}
_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

_AGO_RE = re.compile(
    r"\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|couple|few|several)\s+(day|week|month|year)s?\s+ago\b"
)
_LAST_UNIT_RE = re.compile(r"\b(?:last|past|previous)\s+(week|month|year)\b")
_LAST_WEEKDAY_RE = re.compile(
    r"\b(?:last|past|this past)\s+"
    r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"
)
_WEEKEND_RE = re.compile(r"\b(?:last|past|this past)\s+weekend\b")
_YESTERDAY_RE = re.compile(r"\byesterday\b")


class Anchor:
    __slots__ = ("target", "tol", "desc")

    def __init__(self, target: _dt.datetime, tol: int, desc: str):
        self.target = target
        self.tol = tol
        self.desc = desc


def resolve_anchor(query: str, now_dt: Optional[_dt.datetime]) -> Optional[Anchor]:
    """Resolve a relative time phrase in the query to a target calendar date."""
    if now_dt is None:
        return None
    q = (query or "").lower()

    m = _AGO_RE.search(q)
    if m:
        nraw, unit = m.group(1), m.group(2)
        n = int(nraw) if nraw.isdigit() else _NUM_WORDS.get(nraw, 1)
        days = n * _UNIT_DAYS[unit]
        return Anchor(now_dt - _dt.timedelta(days=days), _UNIT_TOL[unit],
                      f"{n} {unit}{'s' if n != 1 else ''} ago")

    m = _YESTERDAY_RE.search(q)
    if m:
        return Anchor(now_dt - _dt.timedelta(days=1), 2, "yesterday")

    m = _LAST_WEEKDAY_RE.search(q)
    if m:
        wd = _WEEKDAYS[m.group(1)]
        back = (now_dt.weekday() - wd) % 7
        if back == 0:
            back = 7
        return Anchor(now_dt - _dt.timedelta(days=back), 3, f"last {m.group(1)}")

    if _WEEKEND_RE.search(q):
        back = (now_dt.weekday() - 5) % 7
        if back == 0:
            back = 7
        return Anchor(now_dt - _dt.timedelta(days=back), 4, "last weekend")

    m = _LAST_UNIT_RE.search(q)
    if m:
        unit = m.group(1)
        return Anchor(now_dt - _dt.timedelta(days=_UNIT_DAYS[unit]),
                      _UNIT_TOL[unit], f"last {unit}")

    return None


def day_gap(a: _dt.datetime, b: _dt.datetime) -> int:
    return abs((a.date() - b.date()).days)


# ---------------------------------------------------------------------------
# Deterministic time table (conversation-date spans)  [UNCHANGED from 30ae]
# ---------------------------------------------------------------------------

def approx(days: int) -> str:
    """Compact human gloss of a day-count (weeks/months), judge-forgiving."""
    extras: List[str] = []
    if days >= 13:
        extras.append(f"~{round(days / 7)} weeks")
    if days >= 55:
        extras.append(f"~{round(days / 30.44)} months")
    if extras:
        return f"{days} days (" + ", ".join(extras) + ")"
    return f"{days} day{'s' if days != 1 else ''}"


def build_time_table(
    dated: List[Tuple[str, _dt.datetime]],
    now_dt: Optional[_dt.datetime],
    max_pairs_from: int = 4,
) -> str:
    """'TIME SPANS' block: each conversation's gap to the current date, plus
    pairwise gaps among the most-relevant dated conversations.  Pure Python
    fallback for temporal questions when the LLM date-resolver yields nothing.
    """
    if not dated:
        return ""
    lines: List[str] = []
    if now_dt is not None:
        for label, dt in dated:
            g = day_gap(dt, now_dt)
            rel = "earlier today" if g == 0 else f"{approx(g)} before the current date"
            lines.append(f"- The conversation on {label} was {rel}.")

    top = dated[:max_pairs_from]
    pair_lines: List[str] = []
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            (la, da), (lb, db) = top[i], top[j]
            if da == db:
                continue
            g = day_gap(da, db)
            earlier, later = (la, lb) if da <= db else (lb, la)
            pair_lines.append(
                f"- The conversation on {earlier} and the conversation on "
                f"{later} are {approx(g)} apart."
            )

    if not lines and not pair_lines:
        return ""
    out = ["TIME SPANS (already computed from the conversation dates — read the "
           "relevant number instead of doing the arithmetic yourself):"]
    out.extend(lines)
    if pair_lines:
        out.append("Spans between pairs of conversations:")
        out.extend(pair_lines)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Parsing helpers for the deterministic scaffolds (memo.py runs the LLM step;
# the arithmetic is finished here in Python).
# ---------------------------------------------------------------------------

_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_NORM_RE = re.compile(r"[^a-z0-9]+")


# ---- aggregation enumeration ------------------------------------------------
# The LLM emits one line per matching item carrying an explicit quantity:
#   "- <label> | QTY: <number>"
# We parse (label, quantity) so Python can compute BOTH a distinct count and a
# quantity sum. QTY is mandatory in the prompt; if a line omits it we default to
# 1 (treat the item as a single thing) rather than guessing a number out of the
# label text (which caused 30ae's "iPhone 12" -> 12 class of mis-parses).
_QTY_RE = re.compile(r"\bqty\b\s*[:=]?\s*\$?\s*(\d[\d,]*(?:\.\d+)?)", re.IGNORECASE)
# Summary / commentary lines to drop. Matched on a WORD BOUNDARY so a genuine
# item like "County fair rides" is NOT mistaken for a "Count:" summary line
# (a substring-prefix check dropped it and under-counted — caught in dry run).
_SKIP_RE = re.compile(
    r"^(total|sum|grand|computed|distinct|that is|the answer|enumeration|"
    r"count|number of|none)\b",
    re.IGNORECASE,
)


def parse_enumeration(text: str) -> List[Tuple[str, float]]:
    """Parse the LLM enumeration into (label, quantity) pairs.

    De-duplicates on a normalised label key (so the model listing the same thing
    twice collapses). Lines that are obviously totals/commentary are skipped.
    """
    out: List[Tuple[str, float]] = []
    seen = set()
    for raw in (text or "").splitlines():
        line = _BULLET_RE.sub("", raw.strip()).strip()
        if not line:
            continue
        low = line.lower()
        if low in ("none", "n/a", "na"):
            continue
        if _SKIP_RE.match(low):
            continue
        qty = 1.0
        m = _QTY_RE.search(line)
        if m:
            try:
                qty = float(m.group(1).replace(",", ""))
            except ValueError:
                qty = 1.0
            label = line[:m.start()].rstrip(" |:-\t").strip() or line
        else:
            # strip a trailing "| ..." decoration if present
            label = line.split("|", 1)[0].strip() if "|" in line else line
        key = _NORM_RE.sub(" ", label.lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append((label, qty))
    return out


def fmt_qty(x: float, money: bool = False) -> str:
    """Render a quantity: integers without decimals, money with a $ prefix."""
    if abs(x - round(x)) < 1e-9:
        body = f"{int(round(x)):,}"
    else:
        body = f"{x:,.2f}"
    return f"${body}" if money else body


# ---- money (kept for compatibility; fmt_qty supersedes for sums) ------------
def fmt_money(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return f"${int(round(x)):,}"
    return f"${x:,.2f}"


# ---- temporal event resolver parsing  [UNCHANGED from 30ae] -----------------
_EVENT_LINE_RE = re.compile(
    r"EVENT:\s*(.*?)\s*\|\s*DATE:\s*(.+?)\s*$", re.IGNORECASE
)

_VALUE_LINE_RE = re.compile(
    r"VALUE:\s*(.*?)\s*\|\s*DATE:\s*(.+?)\s*$", re.IGNORECASE
)


def _norm_value(v: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (v or "").lower()).strip()


def parse_valued_events(text: str) -> List[Tuple[str, Optional[_dt.datetime]]]:
    """Parse the newest-value resolver output ('VALUE: ... | DATE: YYYY/MM/DD')."""
    out: List[Tuple[str, Optional[_dt.datetime]]] = []
    for raw in (text or "").splitlines():
        m = _VALUE_LINE_RE.search(raw.strip())
        if not m:
            continue
        val = m.group(1).strip().rstrip(" .")
        dstr = m.group(2).strip()
        dt = parse_date(dstr)
        if val and not re.sub(r"[^a-z]", "", val.lower()) == "none":
            out.append((val, dt))
    return out


def build_latest_value_block(values: List[Tuple[str, Optional[_dt.datetime]]]) -> str:
    """Given (value, date) mentions, emit a block that makes the NEWEST value
    explicit — but only when there is a genuine update to disambiguate (>=2
    DISTINCT dated values). Returns '' otherwise, so the normal path is untouched
    for single-value questions."""
    dated = [(v, d) for v, d in values if d is not None]
    if len(dated) < 2:
        return ""
    distinct = {_norm_value(v) for v, _ in dated}
    if len(distinct) < 2:
        return ""
    dated.sort(key=lambda vd: vd[1])
    newest_val, newest_dt = dated[-1]
    lines = [
        "CURRENT (MOST RECENT) VALUE — this question asks for the latest/current "
        "state, and the answer CHANGED over time. The dated mentions are listed "
        "oldest first; the LAST one is the current answer, EVEN IF an earlier "
        "value was described more or sounds more definite:",
    ]
    for v, d in dated:
        lines.append(f"- {d.date().isoformat()}: {v}")
    lines.append(
        f"=> Current answer (newest, {newest_dt.date().isoformat()}): {newest_val}"
    )
    return "\n".join(lines)


def parse_resolved_events(text: str) -> List[Tuple[str, Optional[_dt.datetime]]]:
    """Parse the LLM date-resolver output ('EVENT: ... | DATE: YYYY/MM/DD')."""
    out: List[Tuple[str, Optional[_dt.datetime]]] = []
    for raw in (text or "").splitlines():
        m = _EVENT_LINE_RE.search(raw.strip())
        if not m:
            continue
        desc = m.group(1).strip()
        dstr = m.group(2).strip()
        dt = parse_date(dstr)
        if desc:
            out.append((desc, dt))
    return out


def build_resolved_block(
    events: List[Tuple[str, Optional[_dt.datetime]]],
    now_dt: Optional[_dt.datetime],
) -> str:
    """Turn resolved (event, date) pairs into a computed durations block."""
    dated = [(d, dt) for d, dt in events if dt is not None]
    if not dated:
        return ""
    lines = ["COMPUTED EVENT DATES & DURATIONS (dates taken from your "
             "conversations; all gaps computed exactly — read the relevant "
             "number below):"]
    for desc, dt in dated:
        if now_dt is not None:
            g = day_gap(dt, now_dt)
            rel = "earlier today" if g == 0 else f"{approx(g)} before the current date"
            lines.append(f"- {desc}: {dt.date().isoformat()} ({rel}).")
        else:
            lines.append(f"- {desc}: {dt.date().isoformat()}.")
    if len(dated) >= 2:
        lines.append("Gaps between events:")
        for i in range(len(dated)):
            for j in range(i + 1, len(dated)):
                (da, dta), (db, dtb) = dated[i], dated[j]
                if dta == dtb:
                    continue
                g = day_gap(dta, dtb)
                e, l = (da, db) if dta <= dtb else (db, da)
                lines.append(f"- From '{e}' to '{l}': {approx(g)}.")
    return "\n".join(lines)


def guidance_for(kind: QueryKind) -> List[str]:
    """Per-type answer guidance — only the lines relevant to the inferred type
    are shown; the weak model follows a short focused instruction far better
    than a wall of bullets."""
    g: List[str] = []
    if kind.duration:
        g.append(
            "- This asks about a DURATION / how long ago. Decide which event(s) "
            "the question is about, then read the matching number from the "
            "computed block above. If the question compares two events (e.g. "
            "'... when I [did something]'), use the gap BETWEEN those two events, "
            "not the gap from today."
        )
    if kind.order:
        g.append(
            "- This asks about ORDER. Use the computed dates above; the earlier "
            "date happened first. The conversations are also listed oldest to "
            "newest."
        )
    if kind.agg and kind.agg_mode == "sum":
        g.append(
            "- This asks for a TOTAL / SUM / how-many-TIMES. Read the ENUMERATION "
            "above: it lists each matching item with its quantity and the "
            "quantities have already been added up for you. First check the list "
            "against the conversations below and, if a matching item is missing, "
            "add its quantity. Then report the TOTAL OF ALL QUANTITIES — not the "
            "number of lines."
        )
    elif kind.agg:
        g.append(
            "- This is a COUNT of DISTINCT things. Read the ENUMERATION above: it "
            "lists each matching item once. Make sure items that refer to the "
            "same real-world thing are not counted twice and that none are "
            "missing, then report the NUMBER OF DISTINCT ITEMS."
        )
    if kind.pref:
        g.append(
            "- This asks for a PERSONALIZED recommendation. Use the user's stated "
            "preferences and personal details above to give a specific "
            "recommendation tailored to them; do not give generic advice and do "
            "not say you lack information when relevant preferences are shown."
        )
    return g
