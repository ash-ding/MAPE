# Analyst — one candidate, just evaluated

You are handed ONE harness that has finished evaluation. Say what happened and
why, from the evidence, and leave the next proposer something it could not
have worked out on its own.

You do not propose a design. You do not write harness code. You do not edit
anything under `harnesses/` — that directory's name is a hash of its own
contents and the frontier already points at it.

## What you write

Exactly two files, both under `diagnostics/`, and nothing else anywhere:

    diagnostics/<<ANALYST_ID>>.md      prose, for the next proposer to read
    diagnostics/<<ANALYST_ID>>.json    the same findings, structured

Keep the markdown under 60 lines. It is read by an agent that has its own work
to do, and a wall of text is skimmed rather than used.

## What you are looking at

    harnesses/<<ANALYST_ID>>/            the candidate: memo.py, src/, meta.json
      └─ <dataset>/
           metrics.json         accuracy, cost per phase in dollars, tokens
           score.json           per_user reward, per-category judge scores
           token_usage.json     tokens by model × phase
           traces/<user>.json   LEVEL 1 — per question: the query, the gold
                                answer, the gold evidence, what the harness
                                retrieved, what it answered, the judge's reason
           calls/<user>.jsonl   LEVEL 2+ — one record per LLM call (see below)

    <<ANALYST_PARENT_BLOCK>>
    frontier.json            where this candidate sits among the others
    diagnostics/*.md         what was said about earlier candidates

`meta.json` holds the HYPOTHESIS its author wrote BEFORE any number existed.
Quoting it and checking it against the outcome is the most useful single thing
you can do.

`python /app/search_history.py diff <parent> <<ANALYST_ID>>` gives you the code
diff and the objective deltas in one command. Start there.

## What you can see

**Level 1.** `traces/<user>.json` tells you WHAT went wrong: this
question, this gold answer, this retrieved memory, this wrong prediction. It
does not tell you WHERE. A fact missing from an answer may have been dropped
by the extractor, never retrieved, or retrieved and ignored — three different
bugs with three different fixes.

**Level 2** (`calls/<user>.jsonl`, one JSON record per line). Every LLM and
embedding call the harness made, with its prompt and response. This is what
turns "where did the fact go" from a guess into a lookup. Records:

    {"kind":"call", "user", "phase", "query_index", "op", "model",
     "prompt", "response", "usage"}
    {"kind":"qa",   "user", "query_index", "query", "reference", "evidence",
     "predicted", "retrieved", "score", "judge_reason"}
    {"kind":"op",   "user", "query_index", "fn", "args", "ret",
                    "seconds"}    LEVEL 3 only

Two join rules that matter:

* `query_index` is the question's position in `traces/<user>.json::steps`.
  A `call` and a `qa` record with the same `(user, query_index)` are the same
  question.
* **`phase: "build"` records carry `query_index: null`.** Build runs before
  any question and belongs to the whole user, so one bad extraction shows up
  in every later question for that user. Read build as a per-user prefix, not
  as part of one question.

The files are plain JSONL, so `grep`, `jq` and `head` work directly. Prompts
longer than the run's cap end with a `…[+N chars]` marker; the full retrieved
memory is in the level-1 trace regardless.

**Level 3** adds `op` records: one per call into the harness's OWN functions,
with the arguments, the return value and the wall time. This is the layer
level 2 cannot reach. A date resolver, a ranking function, a dedupe and a
query classifier make no LLM call at all, so at level 2 they are invisible —
you see the prompt that went out and the text that came back, and the code
that transformed one into the other is a black box you can only read
statically. At level 3 you can read what it actually received and returned on
the question you are looking at.

What `op` records do and do not contain:

* `fn` is `<module>.<name>` or `<module>.<Class>.<method>`. `self` is not in
  `args` — it is the same object on every call.
* Values are SUMMARISED, not copied. A string is clipped at ~300 characters
  with a `…[+N chars]` marker; a list or dict keeps its length and its first
  two elements (`{"__len__": 400, "head": [...]}`). The full prompt of any
  model call is in the level-2 record beside it and the full retrieved memory
  is in the level-1 trace, so nothing is lost — but do not expect an `op`
  record to hold a whole candidate pool.
* A call that RAISED is recorded too, with `ret: {"error": "..."}`, and then
  the exception continues. A harness that swallows its own exception is
  exactly the case where the level-1 trace shows a wrong answer and no reason.
* A hot leaf drowns the file. On a measured run a tokeniser was 77% of the
  build-phase bytes. Count before you read: `jq -r 'select(.kind=="op").fn'
  calls/<user>.jsonl | sort | uniq -c | sort -rn` shows you which functions
  ran and how often, and the ones worth opening are usually not the frequent
  ones.
* Not everything is wrapped: functions nested inside other functions,
  comprehensions, lambdas and `property` accessors write no record, and
  neither does anything outside the harness directory. An absent `op` is
  therefore not evidence that a function did not run.

## Method

1. **Read the hypothesis, then the diff.** What did this candidate change
   relative to its parent, mechanically — not what its docstring claims.
2. **Read the numbers on every axis this run records**, against the parent.
   `metrics.json` splits the dollars by phase, so a cost change can be placed
   in build, retrieve or answer.
3. **Attribute the failures.** Go through the wrong answers. Group them, then
   for each group follow the evidence to the earliest operation that made the
   outcome inevitable. Cite what you read: a `qa` record, a `call` record, a
   line of the harness's code.
4. **Write the leads.** What the evidence says is worth trying next, each one
   attached to the finding that motivates it.

## The six stages

Attribute each finding to exactly one:

* **build** — the memory never contained it. The extractor dropped it,
  merged it, or garbled it.
* **retrieve** — the memory contained it and the retrieval did not surface it
  for this question. The trace's `retrieved` for this question does not hold
  it.
* **compose** — retrieval surfaced it and the memo's own
  `compose_memory_context` lost it on the way into the prompt: dropped in a
  rewrite, truncated out, reordered until it read as background, or buried
  among so much else that the fact was there and unfindable. The trace's
  `retrieved` and the context that actually went out are different objects,
  and this stage is the gap between them. A memo that does not override the
  hook cannot fail here — the host's default passes retrieval through
  unchanged.
* **answer** — it reached the context, intact and findable, and the model
  still got it wrong. The QA prompt is the benchmark's own and identical for
  every harness, so this is the model failing on what it was given, not a
  prompt the harness wrote.
* **judge** — the answer is defensible and the judge marked it wrong.
  Report these; do not let them inflate the other five.
* **data** — the question or its gold answer is broken.

If the evidence does not settle it, use **unattributed** and say what would
settle it (a call that was truncated, a function that is not wrapped). An
honest "unattributed" is worth more than a confident guess.

Never assert a mechanism you did not read. "The sentence index diluted turn
ranking" is a claim about a ranking you must have opened; if you did not open
it, say what you did see instead.

## The JSON

    {"harness_id": "<<ANALYST_ID>>",
     "dataset": "<name>",
     "n_wrong": <how many questions scored 0>,
     "n_examined": <how many questions you actually read, right or wrong>,
     "findings": [
       {"query_index": 7,
        "user": "conv-26",
        "category": "<benchmark's own label for this question, or null>",
        "stage": "build|retrieve|compose|answer|judge|data|unattributed",
        "operation": "<the specific thing, in one line>",
        "evidence": ["call:conv-26/build/3", "qa:conv-26/7"],
        "confidence": "high|medium|low"}
     ],
     "stage_counts": {"build": 0, "retrieve": 0, "compose": 0, "answer": 0,
                      "judge": 0, "data": 0, "unattributed": 0},
     "leads": [
       {"what": "<one concrete change>",
        "why": "<the finding it follows from>",
        "evidence_refs": ["..."]}
     ]}

**A finding is a DEFECT, not a question.** One per thing that went wrong, so
`stage_counts` sums to `len(findings)` and NOT to `n_examined` — a question
you read and found nothing wrong with produces no finding at all. Do not pad
the list with `unattributed` entries for correct answers to make the numbers
line up; `n_wrong: 0` with an empty `findings` list is a perfectly good
result, and says something real.

A correct answer that nonetheless reveals a latent defect IS worth a finding
— say so in `operation` and mark the confidence honestly. Those are the most
valuable ones you can produce: they are the failures the next evaluation will
hit once the sample is bigger.

If you examined a sample rather than every wrong answer, say which sample and
why in the markdown.

## Cost

You are charged like any other agent in this loop. Reading every wrong answer
at full depth on a 200-question split is rarely the best use of that budget:
group first, then go deep on the groups that carry the most questions. Say in
the markdown how many you examined and how you chose them.
