"""Hand-written input adapter: LoCoMo recorder.init -> the LongMemEval shape.

The harness in memo_lme.py + src/ was evolved on LongMemEval and reads
recorder.init["sessions"] (a list of {"session_id", "date", "messages":
[{"role", "content"}]}) plus "query" / "question_date". LoCoMo passes
{"conversation": {...}} at build and {"conversation": {...}, "query": ...} at
retrieve. This module maps one onto the other and nothing else:

  * sessions session_1..N, in numeric order, become the session list;
  * session_k_date_time ("1:56 pm on 8 May, 2023") is rewritten to
    LongMemEval's "2023/05/08 (Mon) 13:56", which the harness's date parser reads;
  * each turn becomes a message whose role is the speaker name (the harness
    renders "<Role>: <content>"); an image caption is folded into the text as
    "[shared an image: ...]", the same rendering the LoCoMo-evolved MAPE-A_LME
    harness uses;
  * LoCoMo has no question date, so question_date is passed as "".

role_mode (default "speaker", as used for lme_qwen_m step 19):
  * "speaker"         role = the speaker name.
  * "user_assistant"  role = "user" for speaker_a and "assistant" for speaker_b, with the
                      speaker name kept at the start of the content ("Caroline: ...").
                      For harnesses that split a session into rounds at each "user"
                      message (lme_qwen_acc step 20): with speaker-name roles no message
                      is "user" and every session would collapse into one round.
"""
from __future__ import annotations

import datetime as _dt
import re
from typing import Dict, List

_LOCOMO_DT = re.compile(
    r"(\d{1,2}):(\d{2})\s*([ap]m)\s+on\s+(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})", re.I)
_MONTHS = ["january", "february", "march", "april", "may", "june", "july",
           "august", "september", "october", "november", "december"]


def _month(name: str) -> int:
    n = name.lower()
    for i, m in enumerate(_MONTHS, 1):
        if n == m or (len(n) >= 3 and m.startswith(n[:3])):
            return i
    return 0


def lme_date(locomo_dt: str) -> str:
    m = _LOCOMO_DT.search(locomo_dt or "")
    if not m:
        return str(locomo_dt or "")
    hh, mm, ap, day, mon, year = m.groups()
    month = _month(mon)
    if not month:
        return str(locomo_dt)
    h = int(hh) % 12 + (12 if ap.lower() == "pm" else 0)
    d = _dt.datetime(int(year), month, int(day), h, int(mm))
    return d.strftime("%Y/%m/%d (%a) %H:%M")


def _turn_text(turn: Dict) -> str:
    text = str(turn.get("text", "") or "").strip()
    cap = turn.get("blip_caption") or turn.get("caption")
    if isinstance(cap, str) and cap.strip():
        text = f"{text} [shared an image: {cap.strip()}]".strip()
    return text


def to_lme_init(init: Dict, role_mode: str = "speaker") -> Dict:
    init = init or {}
    if "sessions" in init:  # already LongMemEval-shaped
        return init
    conv = init.get("conversation") or {}
    nums = []
    for k in conv:
        m = re.fullmatch(r"session_(\d+)", str(k))
        if m:
            nums.append(int(m.group(1)))
    sessions: List[Dict] = []
    for k in sorted(nums):
        turns = conv.get(f"session_{k}") or []
        msgs = []
        for t in turns:
            if not isinstance(t, dict):
                continue
            spk = str(t.get("speaker", "") or "")
            if role_mode == "user_assistant":
                role = "user" if spk == conv.get("speaker_a") else "assistant"
                content = f"{spk}: {_turn_text(t)}" if spk else _turn_text(t)
            else:
                role, content = spk, _turn_text(t)
            msgs.append({"role": role, "content": content})
        sessions.append({"session_id": f"session_{k}",
                         "date": lme_date(str(conv.get(f"session_{k}_date_time", ""))),
                         "messages": msgs})
    out: Dict = {"sessions": sessions}
    if "query" in init:
        out["query"] = init["query"]
        out["question_date"] = ""
    return out
