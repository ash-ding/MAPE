"""Compose-side reasoning: deliberation, robust preference handling, stated
durations.

WHY THIS MODULE EXISTS (lineage: 58e72198d2e5 / df183083b251)
------------------------------------------------------------
Diagnosis of the 0.77 record (58e7): retrieval is solved (gold session retrieved
in 24/24 wrong answers) — the losses are reasoning-side. The weak 30B QA model
mis-reads dates, drops/double-counts items, and — on advice/preference questions —
answers generically or refuses (preference bucket = 1/6).

A TRACE-VERIFIED root cause for the preference collapse: in EVERY one of the 5
preference failures the gold detail (turbinado sugar, Fender/Les Paul, lemon
poppyseed cake, debate team, French-podcast language-learning) was present IN THE
RETRIEVED SESSION TEXT. It was lost at two later points:
  1. the RELEVANCE-FILTERED map-extraction ("facts relevant to <this question>")
     dropped the oblique preference — e.g. it returned NONE on the French-podcast
     session for "recommend cultural events", because "French podcasts" does not
     look relevant to "cultural events";
  2. even when the fact survived, the generated/sealed answer went generic.

This module fixes both WITHOUT retrieval changes:
  * PREF_EXTRACT_SYS — an INCLUSIVE per-session extraction of the user's own
    preferences/experiences/choices (not "relevant to the question"), so oblique
    preferences survive to the answer stage.
  * PREF_SELECT_SYS — pick the 1-3 specific user details most relevant to this
    advice request (bridges the oblique gap: French/listening -> language
    practice at a cultural event).
  * PREF_WRITE_SYS — write a response that explicitly builds on those specifics.
  The selected specifics are shown PROMINENTLY in the composed context, above the
  draft, so they are load-bearing even if the draft drifts generic (the 58e7
  failure mode where a generic stamped draft was echoed).

  * STATED_DUR_SYS — for "how long did I X (before Y)" questions, read an
    EXPLICITLY STATED duration ("for two weeks", "a week later") from the
    evidence instead of computing from conversation dates. Targets binoculars
    (2 weeks, got 7 days) and tennis racket (1 week, abstained) in the 58e7
    traces, where the stated duration was in the text but the date-math scaffold
    produced the wrong number / abstained.

All prompts call the run's unified QA model via core.llm.Agent (model unset).
"""
from __future__ import annotations

import re
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Advice / preference detection  [carried from 58e7, unchanged]
# ---------------------------------------------------------------------------
_ADVICE_CUES = re.compile(
    r"\b("
    r"recommend|recommendation|recommendations|suggest|suggestions?|"
    r"advice|advise|any tips|tips on|tips for|some tips|"
    r"what to look for|what should i (get|buy|look|choose|pick|do|try|make|bake|cook|read|watch|wear)|"
    r"should i\b|do you think|good idea|would it be (a )?good|"
    r"thinking of|thinking about|planning to|planning on|"
    r"any ideas|ideas for|some ideas|help me (choose|pick|decide|plan|find|figure)|"
    r"what would you (recommend|suggest)|"
    r"looking for (some|a|any)|"
    r"what (kind|sort|type) of .* should i"
    r")\b"
)
_HARD_FACTUAL = re.compile(
    r"\b(how many|how much|how long|how old|when did|when was|what date|"
    r"what time|which day|how often|number of)\b"
)
# RECALL cues: the user is asking us to retrieve a specific stored fact
# ("remind me of that dessert shop", "do you remember the name"), NOT for a
# recommendation. These look superficially advice-shaped ("planning to", "I was
# wondering") but need the exact factual answer, so they must stay on the normal
# deliberation path. Verified: none of the genuine preference questions use these
# cues, and this keeps the Orlando "remind me of the dessert shop" recall (a
# currently-correct single-session-assistant answer) off the advice path.
_RECALL_CUES = re.compile(
    r"\b(remind me|remind|do you remember|you remember|recall|"
    r"what (was|were) the|what did (you|i) (say|tell|mention|call|name))\b"
)


def is_advice_query(query: str, kind) -> bool:
    """True when the user is asking for a personalised recommendation/opinion
    rather than recalling a specific stored fact."""
    q = (query or "").lower()
    if getattr(kind, "temporal", False) or getattr(kind, "agg", False):
        return False
    if _HARD_FACTUAL.search(q) or _RECALL_CUES.search(q):
        return False
    return bool(_ADVICE_CUES.search(q) or getattr(kind, "pref", False))


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

DELIB_SYS = (
    "You are the assistant. The user is asking you about THEIR OWN history. You "
    "are given dated evidence recalled from your past conversations with the "
    "user, plus the current date. Determine the correct answer using ONLY this "
    "evidence.\n"
    "Think carefully and step by step:\n"
    "- DATES / DURATIONS: read the date stamp on each conversation and the "
    "current date; if pre-computed dates or time gaps are shown, trust them. If a "
    "length of time is explicitly STATED in the evidence (e.g. 'for two weeks'), "
    "use the stated duration rather than computing one from the dates.\n"
    "- COUNTS / TOTALS: if an enumeration with quantities is shown, use its "
    "computed total; otherwise first list the matching items, then count or add "
    "them.\n"
    "- CHANGES OVER TIME: if a fact was later updated, the value from the MOST "
    "RECENT conversation is the current one.\n"
    "- Do NOT guess. If the evidence does not actually state the information "
    "asked for, the answer is NONE.\n\n"
    "Respond in EXACTLY this format, nothing else:\n"
    "REASONING: <your step-by-step reasoning in 1-4 sentences>\n"
    "ANSWER: <the answer as a short phrase using the user's exact wording where "
    "possible; or the single word NONE if the evidence does not contain it>\n"
    "QUOTE: <copy ONE short sentence or phrase from the evidence above that "
    "directly supports the answer; or NONE if the answer is a computed "
    "date/count/total or is not supported by any single line>"
)

# INCLUSIVE preference extraction — the key fix. Extract the user's OWN
# preferences/experiences/choices regardless of apparent relevance to the
# question, so oblique preferences (French podcasts for a cultural-events
# question) survive the map step.
PREF_EXTRACT_SYS = (
    "You read ONE past conversation and extract what it reveals about the USER "
    "themselves. You do NOT answer any question.\n"
    "List, one per line, every concrete thing the USER revealed about their own:\n"
    "- preferences, tastes, likes and dislikes;\n"
    "- hobbies, interests, activities, and skills they are learning;\n"
    "- goals, plans, and things they are excited about;\n"
    "- specific named choices (brands, product names, titles, dishes, recipes, "
    "places, tools) and how they felt about them;\n"
    "- notable personal experiences and past successes or struggles.\n"
    "Rules:\n"
    "- Be INCLUSIVE. Include a detail even if it seems only loosely related to "
    "anything — a concrete personal detail is always worth keeping.\n"
    "- Quote the specific wording (names, numbers) so each line stays concrete "
    "and meaningful on its own.\n"
    "- Report only what the USER actually said or did; do not invent, and do not "
    "list the assistant's suggestions as if they were the user's.\n"
    "- Output the single word NONE only if the user revealed nothing personal at "
    "all in this conversation."
)

# Pick the specific user details most relevant to THIS advice request — this is
# where the oblique bridge happens (listening to French podcasts -> wanting to
# practice French at a cultural event).
PREF_SELECT_SYS = (
    "The user is asking you for a recommendation, advice, or an opinion. Below is "
    "a list of specific things you know about the user (their preferences, "
    "experiences, and choices), gathered from past conversations.\n"
    "Choose the 1 to 3 specific details that are MOST useful for giving a "
    "personalised answer to this particular request. A detail is useful if the "
    "answer could meaningfully build on it, even indirectly (e.g. the user is "
    "learning French -> suggest options that let them practice French).\n"
    "Output each chosen detail on its own line, copied with its specifics "
    "(names/numbers) intact. Output the single word NONE only if truly nothing in "
    "the list could inform the answer."
)

# Write the personalised response, explicitly building on the selected specifics.
PREF_WRITE_SYS = (
    "You are the assistant replying to the user. The user is asking you for a "
    "recommendation, advice, or an opinion. You are given the SPECIFIC things you "
    "know about this user that are most relevant.\n"
    "Write a specific, personalised reply (1-3 sentences) that EXPLICITLY builds "
    "on those details and refers to them by name.\n"
    "Rules:\n"
    "- DO name and build on the user's specific details.\n"
    "- DO NOT give generic advice that ignores them.\n"
    "- DO NOT refuse, and do NOT say you lack information or real-time access — "
    "answer from what the user has told you."
)

# Explicitly-stated duration reader — only fires for duration questions.
STATED_DUR_SYS = (
    "You read evidence from past conversations and look for a length of time that "
    "is EXPLICITLY STATED in the text and that answers the question's 'how long' / "
    "'how many days/weeks/...'.\n"
    "Only report a duration that is written out in words or numbers in the "
    "evidence itself — for example 'for two weeks', 'a week later', 'after three "
    "days', 'it took about a month', 'within 10 days'. Do NOT compute or estimate "
    "a duration from dates, and do NOT guess.\n"
    "If such a stated duration exists, output exactly one line:\n"
    "DURATION: <the duration, e.g. 'two weeks'> | FROM: <the verbatim phrase it "
    "appears in>\n"
    "If no length of time is explicitly stated in the evidence, output exactly: "
    "NONE"
)

# Newest-value resolver — only fires for "current / most-recent VALUE" questions
# (where/what/which is X now; what was my most recent X). The weak QA model, when
# a fact was updated over time, anchors on the LOUDER or EARLIER-stated value
# rather than the newest one (step-13 KU regressions: Hawaii drowned Paris; "plan
# to store in a shoe rack" discounted as future). This reads every dated mention
# of the value the question asks about, so Python can pick the newest.
LATEST_VALUE_SYS = (
    "You read dated evidence from past conversations and track how the ANSWER to "
    "ONE question changed over time. You do NOT answer the question yourself.\n"
    "Identify exactly what single thing the question asks for (a place, an item, a "
    "name, a status, a value, etc.). Then find EVERY time the evidence states a "
    "value for that thing, including earlier values that were later changed, and "
    "including a value the user says they plan, intend, or are about to adopt (a "
    "stated intention to change still counts as the newest value).\n"
    "Output one line per distinct dated mention, EARLIEST first:\n"
    "VALUE: <the value, in the user's words> | DATE: <YYYY/MM/DD>\n"
    "Use the conversation's date when the text gives no other date. Do not invent "
    "mentions. If the evidence never states a value for the thing asked about, "
    "output exactly: NONE"
)


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

_FIELD_RE = re.compile(r"^\s*(REASONING|ANSWER|QUOTE)\s*:\s*(.*)$", re.IGNORECASE)
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_ABSTAIN_RE = re.compile(r"[^a-z]")
_STATED_DUR_RE = re.compile(
    r"DURATION:\s*(.+?)\s*(?:\|\s*FROM:\s*(.+))?$", re.IGNORECASE
)


def parse_deliberation(text: str) -> Tuple[str, str]:
    """Return (answer, quote). answer == '' means abstain (NONE / empty)."""
    if not text:
        return ("", "")
    answer_lines: List[str] = []
    quote_lines: List[str] = []
    current = None
    for raw in str(text).splitlines():
        m = _FIELD_RE.match(raw)
        if m:
            tag = m.group(1).upper()
            val = m.group(2).strip()
            if tag == "ANSWER":
                current = "answer"
                if val:
                    answer_lines.append(val)
            elif tag == "QUOTE":
                current = "quote"
                if val:
                    quote_lines.append(val)
            else:  # REASONING
                current = "reasoning"
            continue
        if current == "answer":
            answer_lines.append(raw.strip())
        elif current == "quote":
            quote_lines.append(raw.strip())

    answer = " ".join(a for a in answer_lines if a).strip()
    quote = " ".join(q for q in quote_lines if q).strip()

    if not answer_lines and not quote_lines:
        tail = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
        answer = tail[-1] if tail else ""

    if _is_abstain(answer):
        answer = ""
    if _is_abstain(quote):
        quote = ""
    return (answer, quote)


def parse_lines(text: str, cap: int = 12) -> List[str]:
    """Parse a bulleted/line list into clean strings, dropping NONE/empties."""
    out: List[str] = []
    for raw in (text or "").splitlines():
        line = _BULLET_RE.sub("", raw.strip()).strip()
        if not line:
            continue
        low = line.lower()
        if low in ("none", "n/a", "na") or low.startswith("none"):
            continue
        # drop a line that is purely a heading like "preferences:"
        if line.endswith(":") and len(line) <= 24:
            continue
        out.append(line)
        if len(out) >= cap:
            break
    return out


def parse_stated_duration(text: str) -> str:
    """Return a short 'DURATION (phrase)' string, or '' when none is stated."""
    t = (text or "").strip()
    if not t or _looks_none(t):
        return ""
    for raw in t.splitlines():
        m = _STATED_DUR_RE.search(raw.strip())
        if m:
            dur = (m.group(1) or "").strip().rstrip(" .|")
            phrase = (m.group(2) or "").strip().rstrip(" .|")
            if not dur or _looks_none(dur):
                continue
            if phrase and not _looks_none(phrase):
                return f"{dur}  (stated as: \"{phrase}\")"
            return dur
    return ""


def _looks_none(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True
    return _ABSTAIN_RE.sub("", t) == "none"


def _is_abstain(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True
    if _ABSTAIN_RE.sub("", t) == "none":
        return True
    if len(t) <= 48 and (
        t.startswith("none")
        or t.startswith("no answer")
        or t.startswith("not stated")
        or t.startswith("not mentioned")
        or t.startswith("not specified")
        or t.startswith("unknown")
        or t == "n/a"
    ):
        return True
    return False


def looks_like_refusal(text: str) -> bool:
    """A generated answer that refused / punted — dropped so it is not stamped."""
    t = (text or "").strip().lower()
    if not t:
        return True
    cues = (
        "i don't have access",
        "i do not have access",
        "i can't provide",
        "i cannot provide",
        "i'm not able to",
        "i am not able to",
        "as an ai",
        "i don't have enough information",
        "i do not have enough information",
        "no relevant information",
        "i don't have any information",
        "i do not have any information",
    )
    return any(c in t for c in cues)


# Distinctive content tokens shared between a draft and the selected specifics —
# used to confirm a draft is actually grounded (not generic) before it is shown
# as a suggested response. A light filter: even when it drops a draft, the
# specifics themselves remain in the context, so the sealed call still has them.
_WORD_RE = re.compile(r"[a-z0-9]+")
_COMMON = frozenset(
    """the a an and or but if of to in on at for with without from by is are was
    were be been being have has had do does did your you user their they them it
    this that these those some any all more most new good great best try use using
    would could should will can may might also just like really very much many
    recommend suggest suggestion advice tip tips idea ideas response reply answer
    personalized specific build building based given consider since because when
    what which who how why about into over under again both few other such your
    something anything looking feeling excited nostalgic""".split()
)


def draft_is_grounded(draft: str, specifics: List[str]) -> bool:
    """True if the draft shares a distinctive content token with the specifics."""
    if not specifics:
        return True  # nothing to ground against; keep the draft
    draft_toks = {t for t in _WORD_RE.findall((draft or "").lower())
                  if len(t) >= 4 and t not in _COMMON}
    spec_toks = {t for t in _WORD_RE.findall(" ".join(specifics).lower())
                 if len(t) >= 4 and t not in _COMMON}
    return bool(draft_toks & spec_toks)
