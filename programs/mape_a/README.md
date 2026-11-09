# MAPE-A<sub>LME</sub>

The memory program that MAPE's accuracy-only search discovered on
LongMemEval, and the one the paper's case study takes apart. It is an
ordinary `MemoClass` program: the search wrote `memo.py` and everything under
`src/`; `adapters/locomo.py` is the hand-written input adapter used for the
transfer experiments.

Evaluate it from the repository root:

```bash
uv run python scripts/run_method.py --benchmark longmemeval --n 20
uv run python scripts/run_method.py --benchmark locomo --conversations 1 --n 40
```

## What it does

**Build** reads each session once and turns it into a set of atomic,
self-contained notes (`src/notes.py`). Notes and the raw conversation rounds
are both embedded and BM25-indexed. Extraction happens at write time, so the
facts a question sees do not change from query to query.

**Retrieve** fuses dense and BM25 rankings over notes and rounds (reciprocal
rank fusion, `src/retrieval.py`) to pick the relevant sessions. For each chosen
session it surfaces all of that session's notes, with no second relevance
filter, plus the five highest-ranked raw rounds for exact wording and one
"time anchor" session near the question date.

**Compose** is the stage the paper calls Query-Routed Scaffolding. A
rule-based classifier infers from the question text alone whether it asks for
a date, a count or sum, a latest value, a duration, a preference or
recommendation, or something else (`src/scaffold.py`). Each kind gets its own
deterministic scaffold: resolved absolute dates and day gaps for temporal
questions, a per-session map-reduce enumeration with per-item quantities for
aggregation (`src/aggregate.py`), a current-value gate for knowledge updates,
and an inclusive preference extraction for advice questions (`src/reason.py`).
A final deliberation call drafts an answer from the scaffolded evidence, and
the whole package, evidence plus draft, is handed to the fixed QA model as its
context. The QA model still writes the answer.

Every model call in all three stages goes through `core.llm.Agent` with the
run's serving backbone, so each one is charged to its stage in the cost
accounting.

## Files

```text
memo.py              AtomicLedgerMemo: the three lifecycle hooks and the per-kind compose paths
src/notes.py         build-time atomic note extraction
src/retrieval.py     rounds index, BM25 tokenisation, RRF fusion, date parsing and rendering
src/scaffold.py      question-kind classifier and the deterministic scaffolds (pure Python)
src/aggregate.py     map-reduce enumeration for count and sum questions
src/reason.py        system prompts for deliberation, preference selection and durations
adapters/locomo.py   LoCoMo conversation -> LongMemEval session shape (dates, roles, captions)
```

The constants at the top of `memo.py` (session caps, note caps, raw-round
budget) are the values the search settled on; the paper's numbers use them
unchanged.

## Transfer to LoCoMo

The program never saw LoCoMo during search. `adapters/locomo.py` rewrites a
LoCoMo conversation into the session list the program expects (session dates
in LongMemEval's format, speaker names as roles, image captions folded into
the text) and passes an empty question date. `scripts/run_method.py` and
`experiments/transfer/` apply the adapter automatically; nothing in `memo.py`
or `src/` changes.
