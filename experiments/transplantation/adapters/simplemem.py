"""Turn a hand-built baseline's recorded retrieval into the retrieval shape
MAPE-A_LME's (7723a2cc870a) compose stage reads.

MAPE-A_LME's compose reads `retrieved["sessions"]`: a list of
    {"session_idx", "session_id", "date_str", "date_sort", "notes": [str, ...],
     "score", "is_raw"}
with `date_str` in LongMemEval's own format ("2023/05/20 (Sat) 06:13", the only
one its `parse_date` accepts), plus `raw_rounds` (verbatim turns; the baselines
return none, so it is empty).

The same mechanical rule is applied to every baseline:
  * each retrieved memory unit becomes one note:
      SimpleMem  `Content:` (+ " (location: ...)" when a Location field is set)
      LightMem   the text after the leading "<ISO time> <weekday>" stamp
      A-mem      the `memory content:` field of each "talk start time:" entry
                 (its context / keywords / tags fields are dropped)
  * units are grouped by CALENDAR DATE into one pseudo-session per date; the
    session's date is its earliest unit's time; units with no parseable time
    go into one undated session;
  * a session's score is 1/(rank+1) of its best-ranked unit, rank being the
    unit's position in the baseline's own retrieval order;
  * exact duplicate notes within a session are kept once.
Nothing is dropped beyond those duplicates and A-mem's three side fields, and
nothing is re-ranked or re-retrieved.
"""
from __future__ import annotations

import datetime as dt
import re
from typing import Any, Dict, List, Optional, Tuple

_DAY = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_LME_RE = re.compile(r"(\d{4})/(\d{1,2})/(\d{1,2})(?:\s*\(\w+\))?\s*(\d{1,2})?:?(\d{2})?")


def _iso(s: str) -> Optional[dt.datetime]:
    try:
        return dt.datetime.fromisoformat(s.strip()[:19])
    except ValueError:
        return None


def _lme(s: str) -> Optional[dt.datetime]:
    m = _LME_RE.match((s or "").strip())
    if not m:
        return None
    try:
        return dt.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                           int(m.group(4) or 0), int(m.group(5) or 0))
    except ValueError:
        return None


def lme_date_str(t: dt.datetime) -> str:
    return f"{t:%Y/%m/%d} ({_DAY[t.weekday()]}) {t:%H:%M}"


def units(baseline: str, rm: Dict[str, Any]) -> List[Tuple[Optional[dt.datetime], str]]:
    """(time, note) per retrieved unit, in the baseline's retrieval order."""
    out: List[Tuple[Optional[dt.datetime], str]] = []
    if baseline == "simplemem":
        for p in rm.get("passages") or []:
            fields = {}
            for line in str(p).split("\n"):
                k, sep, v = line.partition(":")
                if sep and k.strip() in ("Content", "Time", "Location", "Persons", "Related Entities", "Topic"):
                    fields[k.strip()] = v.strip()
            text = fields.get("Content") or str(p).strip()
            if fields.get("Location"):
                text += f" (location: {fields['Location']})"
            out.append((_iso(fields.get("Time", "")), text))
    elif baseline == "lightmem":
        for p in rm.get("passages") or []:
            m = re.match(r"\s*(\d{4}-\d{2}-\d{2}T[\d:.]+)\s+[A-Za-z]{3}\s+(.*)", str(p), re.S)
            out.append((_iso(m.group(1)), m.group(2).strip()) if m else (None, str(p).strip()))
    elif baseline == "amem":
        for chunk in str(rm.get("memories") or "").split("talk start time:")[1:]:
            head, _, rest = chunk.partition("memory content:")
            content = re.split(r"memory context:", rest, maxsplit=1)[0].strip()
            if content:
                out.append((_lme(head), content))
    else:
        raise ValueError(f"unknown baseline {baseline!r}")
    return out


def to_memorph_retrieval(baseline: str, rm: Dict[str, Any], query: str, question_date: str) -> Dict[str, Any]:
    groups: Dict[Any, Dict[str, Any]] = {}
    order: List[Any] = []
    for rank, (t, note) in enumerate(units(baseline, rm)):
        key = t.date() if t else None
        g = groups.get(key)
        if g is None:
            g = groups[key] = {"t": t, "notes": [], "seen": set(), "score": 0.0}
            order.append(key)
        if t and (g["t"] is None or t < g["t"]):
            g["t"] = t
        if note not in g["seen"]:
            g["seen"].add(note)
            g["notes"].append(note)
        g["score"] = max(g["score"], 1.0 / (rank + 1))
    sessions = []
    for i, key in enumerate(order):
        g = groups[key]
        t = g["t"]
        sessions.append({
            "session_idx": i,
            "session_id": f"{baseline}_{key.isoformat() if key else 'undated'}",
            "date_str": lme_date_str(t) if t else "",
            "date_sort": t.isoformat() if t else "",
            "notes": g["notes"],
            "score": g["score"],
            "is_raw": False,
        })
    return {"sessions": sessions, "raw_rounds": [], "query": query,
            "question_date": question_date, "max_sim": 0.0, "n_sessions": len(sessions)}
