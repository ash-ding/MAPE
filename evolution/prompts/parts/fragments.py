"""Keyed substitution data for the proposer prompts.

Not prose: every value here is indexed by the renderer — by sentinel name
(SANITY_*_SUBS), by the run's sorted `metrics` tuple (OBJECTIVE_AXES_SUBS), or
by dataset key (DATASET_INFO). That is why it stays Python while the four
prose prompts live beside it as evolving_/design_/task_/fix_ .md.
"""


#: `<<OBJECTIVE_AXES_BLOCK>>` per `metrics:` setting — the axes the frontier
#: actually carries for this run, plus what to optimize. Keyed by the metric
#: set as a sorted tuple; the renderer picks one.
OBJECTIVE_AXES_SUBS = {
    ("accuracy",): """      There is NO algorithmic selection — YOU pick which prior(s) to build on.

      WHAT COUNTS AS A BETTER DESIGN, on this run: a higher
      `accuracy_<dataset>` than the best harness already has. That is the
      entire criterion. Cost axes
      are not recorded here, so nothing trades against accuracy and no
      other property of your design substitutes for it. The dollar figures
      you will still see on each line in history.json are what the AGENTS
      spent, not what a candidate costs to run — see "What is not on the
      scoreboard". Nothing there is scored on this run either.
      Not scoring cost does not lift the model policy: the memory system
      and the answer hook use the run's unified model, and a harness that
      names any other model is refused and fails. The number to beat is
      earned with the same model every prior used.
      `python /app/search_history.py best` names the number to beat.""",
    ("accuracy", "efficiency"): """        • cost_usd_per_1k_queries_<dataset>
                              — what running the memory system costs, in
                                dollars per 1,000 queries: every API call it
                                makes in build, retrieve, compose and answer,
                                priced per model at list price (embeddings
                                included, cache discounts ignored), amortized
                                over that user's queries and meaned over users.
                                LOW = cheaper. Excludes the judge, and excludes
                                what the SEARCH costs — see "What is not on the
                                scoreboard". With the run's
                                models fixed this is tokens weighted by price:
                                an output token of the QA model costs 4× an
                                input token, an embedding token 1/20 of one.
                                metrics.json keeps the token readings beside
                                it (cost_tokens_per_query, cost_answer_tokens_
                                per_query, cost_embedding_tokens_per_query)
                                and the build / retrieve / compose / answer
                                split of the dollars, so a cost change can be
                                placed.
      There is NO algorithmic selection — YOU pick which prior(s) to build on.

      WHAT COUNTS AS A BETTER DESIGN, on this run: a candidate that NO
      existing harness beats on BOTH axes at once — one that would join
      the Pareto frontier. Pushing EITHER edge outward counts, and they
      count equally: more accurate at the same cost, the same accuracy
      for fewer tokens, or a trade-off nobody occupies yet. There is no
      exchange rate between the two axes and no ratio to maximise — a
      design that costs more and answers better is a real gain, and so
      is one that costs far less for a little less accuracy. Each adds
      a point the frontier did not have.

      The one thing that does NOT count: a design so cheap it cannot
      answer. Near-zero cost at near-zero accuracy is undominated on
      paper and worthless in fact.

      `python /app/search_history.py pareto` prints the current frontier —
      your proposal is an improvement exactly when it would join that
      list. Read `accuracy_<dataset>` and `cost_usd_per_1k_queries_<dataset>`
      side by side: the two are
      the whole scoreboard, with no third axis to break a tie.""",
}


#: `TASK_PROMPT_TEMPLATE` sentinels, keyed by how many benchmarks this run
#: targets. The task prompt tells the proposer how to read the frontier and
#: what "general" means here; a run on one benchmark has no cross-benchmark
#: mean to warn about and nothing to dispatch on, and saying otherwise sends
#: it looking for axes and branches that do not exist.
TASK_DATASET_SUBS = {
    "multi": {
        "<<TASK_FRONTIER_AXES>>": """                                  (accuracy_<dataset> = per-benchmark native
                                   scale — NO cross-benchmark mean; the cost
                                   axes, when this run records them, are
                                   LOW=good telemetry — prefer cheaper priors
                                   when scores are similar)""",
        "<<TASK_DIAGNOSE_BLOCK>>": """  - Diagnose root causes: `traces/<user>.json::steps[i].judge_reason` is
    the single richest signal for why a specific QA failed. Group by
    qa_metadata fields (LoCoMo `category`, LongMemEval `question_type`)
    to find failure patterns rather than per-instance noise.
  - Focus where it hurts: read each benchmark's `accuracy_<dataset>` on its
    own; concentrate the next design's effort on the weakest one — but without
    sacrificing generality (see "What makes a good proposal" below).""",
        "<<TASK_GENERALITY_BLOCK>>": """- ONE shared core, dataset-aware DISPATCH only.
  The benchmarks differ in `recorder.init` SHAPE (conversation vs
  sessions), so dispatch on init keys to unwrap the
  right input and format the right snippet for the QA agent. But the
  underlying retrieval / storage machinery should be unified
  and parametric — NOT one independent implementation per benchmark.
  Why: cloned per-benchmark logic is brittle, misses transfer between
  benchmarks (a retrieval gain on one should naturally help the others),
  and the copies drift apart as you edit them.""",
    },
    "single": {
        "<<TASK_FRONTIER_AXES>>": """                                  (accuracy_<dataset> = this benchmark's score
                                   on its native scale; the cost axes, when
                                   this run records them, are LOW=good
                                   telemetry — prefer cheaper priors when
                                   scores are similar)""",
        "<<TASK_DIAGNOSE_BLOCK>>": """  - Diagnose root causes: `traces/<user>.json::steps[i].judge_reason` is
    the single richest signal for why a specific QA failed. Group by the
    trace's `qa_metadata` fields (PROPOSER_SYSTEM, "Trace fields", lists
    them) to find failure patterns rather than per-instance noise.
  - Focus where it hurts: let the grouped failures, not the headline
    accuracy, decide what the next design attacks. A question type the
    current retrieval systematically misses is worth more than a uniform
    nudge across all of them.""",
        "<<TASK_GENERALITY_BLOCK>>": """- ONE core, parametric over the data — not keyed to it.
  Drive retrieval and storage off properties of the input (how long the
  history is, how many sessions, what the query asks for), never off a
  dataset name or a sample id. If you need to branch, branch on the
  SHAPE of `recorder.init`. Machinery built this way keeps working when
  the data underneath it changes — which is exactly what the held-out
  test split measures.""",
    },
}


SANITY_ON_SUBS = {
    "<<SANITY_TREE_BLOCK>>": "\n          ├── sanity/          pre-eval sanity-check artifacts (smaller)\n          │   ├── score.json\n          │   └── traces/",
    "<<RUNS_SANITY_SUFFIX>>": "[_sanity]",
    "<<LOG_SANITY_NOTE>>": ", which sanity passes/fails",
    "<<SELF_VAL_COMPARE>>": "~30s for the sanity check",
    "<<SANITY_SECTION>>": """
# Sanity check (post-propose, automatic)

After you finish, your harness runs on a tiny sanity check (a couple of
tasks on 1 sample per benchmark, real data). If it crashes, you'll be asked
to Read your memo.py (and src/), diagnose from the error trace, and Edit —
up to a few attempts. Write code robust to realistic input, not just the
happy path. Self-validation above prevents most sanity-check fixups.
""",
}

SANITY_OFF_SUBS = {
    "<<SANITY_TREE_BLOCK>>": "",
    "<<RUNS_SANITY_SUFFIX>>": "",
    "<<LOG_SANITY_NOTE>>": "",
    "<<SELF_VAL_COMPARE>>": "minutes for the full eval",
    "<<SANITY_SECTION>>": "",
}


# ===========================================================================
# DATASET_INFO + DATASET_RENDER_ORDER
# ===========================================================================

DATASET_INFO = {
    "locomo": {
        "display_name": "LoCoMo",
        "qa_metadata": """    LoCoMo:        {category, evidence}   (trace fields — the answer
                     hook receives {} on LoCoMo)
                     • category 1-4 = factual questions
                     • category 5   = adversarial ("Not mentioned"; tests
                                       hallucination resistance)
                     • evidence     = list of dia_ids (e.g. "D1:9")""",
        "relevant": "    LoCoMo:        relevant_turns       conversation turns evidence resolves to",
        "shape": """## LoCoMo (multi-session two-person conversation)
  Phase 1 chunk:    recorder.init = {"conversation": dict}
                    conversation has keys: speaker_a (str), speaker_b (str),
                    session_1..session_N (List[turn_dict]),
                    session_N_date_time (str)
                    each turn_dict has: speaker, dia_id (e.g. "D1:3"), text
                    (NOTE: only `conversation` is provided — NO summaries,
                     observations, or event_summary.)
  Phase 2 retrieve: recorder.init = {"conversation": dict, "query": str}
""",
        "raw_data": """      LoCoMo:        locomo10.json""",
        "build_call": """        - LoCoMo: ONE call per user, total, with the full user history
          (every session / conversation turn) in `recorder.init`.""",
        "prompts_path": "/app/benchmarks/locomo/prompts.py",
        "dispatch_check": '"conversation" in init',
        "dispatch_update_comment": "# LoCoMo Phase 1",
        "dispatch_retrieve_comment": "# LoCoMo Phase 2",
    },
    "longmemeval": {
        "display_name": "LongMemEval",
        "qa_metadata": """    LongMemEval:   {question_type, question_date, answer_session_ids}
                     (trace fields — the answer hook receives only
                     {question_date})
                     • question_type ∈ {single-session-user,
                                         single-session-assistant,
                                         multi-session,
                                         temporal-reasoning,
                                         knowledge-update}""",
        "relevant": "    LongMemEval:   relevant_context     sessions answer_session_ids point to",
        "shape": """## LongMemEval  (haystack of chat sessions with one question each)
`longmemeval_s` — ~48 sessions/sample, ~120k tokens per haystack.

  Phase 1 chunk:    recorder.init = {"sessions": List[session_dict]}
                    each session_dict has:
                      session_id (str, "session_001", "session_002", ... —
                                  positional, in haystack order),
                      date (str, e.g. "2023/05/20 (Sat) 02:21"),
                      messages (List[{"role": "user"|"assistant", "content": str}])
                    The session_id is a positional label and carries NO
                    gold/distractor signal (gold sessions sit uniformly across
                    the haystack), so there is nothing to key off. Upstream's
                    own reader labels sessions the same way. Retrieval has to
                    be content-based; that is the whole task.
  Phase 2 retrieve: recorder.init = {"sessions": List[session_dict],
                                     "query": str,
                                     "question_date": str   # "YYYY/MM/DD ..." }
                    The question_date is the user's reference time at ask-
                    time — critical for temporal-reasoning and knowledge-
                    update questions.
""",
        "raw_data": """      LongMemEval:   longmemeval_*.json""",
        "build_call": """        - LongMemEval: ONE call per user, total, with the full session
          haystack in `recorder.init`.""",
        "prompts_path": "/app/benchmarks/longmemeval/prompts.py",
        "dispatch_check": '"sessions" in init',
        "dispatch_update_comment": "# LongMemEval Phase 1 (list of session dicts)",
        "dispatch_retrieve_comment": "# LongMemEval Phase 2\n        # init[\"question_date\"] is the user's reference time",
    },
}


#: `CLOSE_THE_LOOP_SUBS["on"]` sentinel `<<ANALYST_EVIDENCE>>`: what the analyst
#: had to read. The proposer weighs a diagnosis by the evidence behind it.
ANALYST_EVIDENCE = (" — including the call-level trace, which records the prompt and\n"
                    "response of every LLM call the harness made, and the record of every\n"
                    "call into the harness's own functions")

DATASET_RENDER_ORDER = ["locomo", "longmemeval"]


#: `TASK_PROMPT_TEMPLATE` sentinel `<<CLOSE_THE_LOOP_BLOCK>>`: who writes the
#: post-evaluation analysis. With `analyst.enabled: false` the proposer writes
#: the retro itself, as it always has. With the analyst on, a dedicated agent
#: has already written a diagnosis from evidence the proposer cannot see at
#: propose time (call-level traces), so asking the proposer to write its own
#: would duplicate the work and, worse, invite it to re-derive by guesswork
#: what the diagnosis established by lookup.
CLOSE_THE_LOOP_SUBS = {
    "off": """# Before you propose: close the loop on what already ran

Nothing in this workspace says whether a candidate's idea WORKED. `meta.json`
was written before the evaluation — it is a hypothesis. The frontier holds
the numbers. What is missing is the sentence joining them, and you are the
only one positioned to write it.

So: for every harness that has results (a `<dataset>/metrics.json`) and no
`retros/<id>.md` yet — usually just the one or two from the last step —
write one. Under 30 lines each:

  - what it changed relative to its parent
    (`python /app/search_history.py diff <parent> <id>`)
  - what the numbers did, on every axis this run records
  - WHY, from the traces rather than the diff. Which question types moved?
    Did a cost change come from build, from retrieve, or from the answer
    step (metrics.json splits the dollars by phase)?
  - one takeaway for whoever proposes next — including "this direction is
    exhausted", when that is what the evidence says

Do this BEFORE you decide what to build. A retrospective written after the
choice is a justification, not a finding.""",

    "on": """# Before you propose: read what the analyst found

You do NOT write retrospectives on this run. After every evaluation a
dedicated analyst reads that candidate's evidence<<ANALYST_EVIDENCE>> and
writes:

  `diagnostics/<id>.md`     what changed, what the numbers did, and WHERE
                            each failure was lost
  `diagnostics/<id>.json`   the same findings as records, with a `stage` per
                            finding: build | retrieve | compose | answer |
                            judge | data | unattributed, and the evidence it
                            read

Read the diagnosis for the last step first, then whichever earlier ones bear
on the direction you are considering. They are the only place this run
records what did NOT work and why.

Two things to keep in mind when you use them:

  - A finding marked `unattributed` means the evidence did not settle it, not
    that the analyst was lazy. Treat it as an open question, not a fact.
  - The leads are suggestions from an agent that saw one candidate. You see
    the whole frontier. Take a lead when the evidence behind it convinces
    you; ignore it when your own reading says otherwise, and say so in your
    `meta.json` description.

Do this BEFORE you decide what to build.""",
}


#: `<<COMPOSE_*>>` sentinels in the evolving and fix halves, keyed by
#: cfg.contract.compose ("on" = the three-hook contract every version before
#: this one described, verbatim; "off" = the two-hook contract: no
#: compose_memory_context, the benchmark's default rendering, a load-time
#: refusal for a class that defines it). The renderer picks one; "on" renders
#: byte-identically to 20260928_0317_b1c1a070.
COMPOSE_SUBS = {
    "on": {'<<COMPOSE_CONTRACT_DOC>>': '      (`from evolution.memo_class import MemoClass`). Documents the three\n      hooks: build_memory_from_data + retrieve_memory_for_query (required)\n      and compose_memory_context (optional, with a default), including when\n      build is re-entered, the read-only-retrieve rule, and the rule that\n      retrieve returns memory rather than an answer. `cat` this when you\n      need exact signatures.\n',
        '<<COMPOSE_QA_PY>>': "      How the HOST sends the QA call. You do not answer: `compose_memory_context`\n      returns the context, the host renders this benchmark's own prompt\n      around it and makes exactly one call to the run's QA model, and that\n      answer goes to the judge unchanged. Read it to see the wire format\n      your context ends up in.", '<<COMPOSE_HOOK_METADATA_NOTE>>': '      None of it reaches your hooks: they see only `recorder.init` — see\n      "Phase 2b" below.',
        '<<COMPOSE_MEMO_PY_SIGNATURES>>': '                       Class inheriting from `evolution.memo_class.MemoClass`,\n                       implementing the two required hooks, and optionally\n                       the third:\n                         async def build_memory_from_data(self, recorder) -> None\n                         async def retrieve_memory_for_query(self, recorder) -> Dict\n                         async def compose_memory_context(self, recorder, retrieved)  # optional',
        '<<COMPOSE_ANSWER_HOOK_SENTENCE>>': '      that query. Answering has its own hook (Phase 2b), where its cost is\n      charged where it belongs.',
        '<<COMPOSE_PHASE_2B>>': "* Phase 2b (optional)  compose_memory_context(recorder, retrieved):\n      Organises what Phase 2 returned into the CONTEXT this benchmark's QA\n      prompt is rendered with. You do NOT answer: the host renders that\n      prompt around what you return, makes one call to the run's QA model,\n      and the answer goes to the judge unchanged. Return:\n          None    — the benchmark's own rendering of `retrieved`, which is\n                    what every baseline uses\n          str     — one context block\n          list    — several blocks, joined by the benchmark\n\n      What you return is measured as `context_tokens`, and every QA call\n      carries it, so its SIZE is a real cost on the frontier — whether you\n      spend it on memory or on telling the model how to use that memory.\n      Model calls you make INSIDE this hook are charged to the `compose`\n      phase and count as memory-system cost; use `Agent()` for them, as\n      build and retrieve do.\n      `recorder` is the one Phase 2 just saw: `recorder.init['query']` is\n      the question and, on LongMemEval, `recorder.init['question_date']` is\n      its date; that is ALL of the question you get. The labels you see in\n      traces (LoCoMo `category`, LongMemEval `question_type`) and the gold\n      evidence fields are for your analysis of PAST evaluations — they are\n      not available at answer time, and the default prompt is sealed so they\n      cannot be read off it.", '<<COMPOSE_CONSTRAINT_HOST_ANSWERS>>': '   key named like one (`answer`, `final_answer`, ...) fails the query. The\n   host does the answering, from the context `compose_memory_context` builds.',
        '<<COMPOSE_FIX_ANSWER_KEY>>': '   - An answer-like key (`answer`, `final_answer`, ...) in what\n     `retrieve_memory_for_query` returned — retrieve returns memory; move\n     the answering into `compose_memory_context`'},
    "off": {'<<COMPOSE_CONTRACT_DOC>>': '      (`from evolution.memo_class import MemoClass`). Documents the hooks,\n      including when build is re-entered, the read-only-retrieve rule, and\n      the rule that retrieve returns memory rather than an answer. `cat` this\n      when you need exact signatures. ON THIS RUN the contract has TWO hooks:\n      build_memory_from_data + retrieve_memory_for_query. The file also\n      documents an optional third, `compose_memory_context`; it is CLOSED\n      here — a harness that defines it is refused when the container loads\n      it and fails its sanity check. Do not define it.',
        '<<COMPOSE_QA_PY>>': "      How the HOST sends the QA call. You do not answer: the host renders\n      this benchmark's own prompt around what `retrieve_memory_for_query`\n      returned (the benchmark's default rendering, the same for every\n      harness and baseline), makes exactly one call to the run's QA model,\n      and that answer goes to the judge unchanged. Read it to see the wire\n      format your retrieved memory ends up in.", '<<COMPOSE_HOOK_METADATA_NOTE>>': '      None of those metadata fields reach your harness at answer time: the\n      labels are for your analysis of PAST evaluations only.',
        '<<COMPOSE_MEMO_PY_SIGNATURES>>': "                       Class inheriting from `evolution.memo_class.MemoClass`,\n                       implementing exactly the two hooks of this run's\n                       contract (a `compose_memory_context` method is refused):\n                         async def build_memory_from_data(self, recorder) -> None\n                         async def retrieve_memory_for_query(self, recorder) -> Dict", '<<COMPOSE_ANSWER_HOOK_SENTENCE>>': "      that query. On this run nothing of yours runs after retrieve: the host\n      renders the benchmark's prompt around what you return and answers.", '<<COMPOSE_PHASE_2B>>': '* There is NO Phase 2b on this run. `evolution.memo_class.MemoClass` documents an\n      optional `compose_memory_context` hook; it is CLOSED here\n      (contract.compose: false). A class that defines it — under any name\n      resolution, in memo.py or a base class of yours — is refused when the\n      container loads it, before any evaluation, and the sanity check fails.\n      What Phase 2 returns is rendered by the benchmark\'s own default\n      rendering, identical for every harness and every baseline, and the host\n      makes one call to the run\'s QA model. So the whole design lives in what\n      you STORE and what you RETURN: the shape, order, size and wording of the\n      retrieved dict is your only lever on the answer, and it is measured as\n      `context_tokens` on every query. `{"inline_memory_blocks": [str, ...]}`\n      renders the blocks inline, in that order; any other dict is rendered as\n      JSON. Model calls belong in build (and, read-only, in retrieve), where\n      they are charged as memory-system cost like everything else.',
        '<<COMPOSE_CONSTRAINT_HOST_ANSWERS>>': "   key named like one (`answer`, `final_answer`, ...) fails the query. The\n   host does the answering, from the benchmark's own rendering of what\n   retrieve returns; do not define `compose_memory_context` (refused).", '<<COMPOSE_FIX_ANSWER_KEY>>': '   - An answer-like key (`answer`, `final_answer`, ...) in what\n     `retrieve_memory_for_query` returned — retrieve returns memory, never\n     an answer; this run has no compose hook, so keep the answering out of\n     the harness altogether (the host answers from what retrieve returns)\n   - "defines `compose_memory_context`": this run\'s contract has two hooks\n     (contract.compose: false). Delete that method and anything that only\n     served it; move what matters into what retrieve stores and returns'},
}
