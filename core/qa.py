"""How the HOST sends the QA call.

`QAContext` binds the run's QA model behind `ask()`. The memo does not reach
it: `compose_memory_context` returns the context, and the host renders this
benchmark's own prompt around it and makes exactly one call, so the answering
step is identical for every memo and every baseline. That is what makes the
comparison about memory rather than about prompt engineering; a memo that
wants stronger answering instructions puts them in its context and pays for
them in `context_tokens`.

`SealedPrompt` survives because the benchmark's prompt still dispatches on
labels the memo may not see (LoCoMo's category, LongMemEval's question_type):
it is assembled host-side and sent, never read.

What the hooks do NOT see
-------------------------
The benchmark's per-question metadata carries two kinds of field: labels
(LoCoMo's `category`, LongMemEval's `question_type`) and gold evidence
(LoCoMo `evidence`, LongMemEval `answer_session_ids`). Neither reaches the
memo: its hooks receive only `recorder.init` (the query and, on LongMemEval,
`question_date` — the fields that are part of the question itself). The
benchmark's own prompt still dispatches on its labels: `default_prompt` is
bound host-side to the full metadata, so the standard answer path is
unchanged for everyone, and it returns a `SealedPrompt` — sendable,
appendable, not readable — so a label cannot be recovered from the prompt
text either. A memo that wants question-type-specific behaviour infers the
type from the query.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

Messages = List[Dict[str, Any]]


class SealedPrompt:
    """The benchmark's QA prompt, sealed: `qa.ask` can send it, a memo can
    append its own instruction to the user turn, nothing can read it.

    Why sealed: the prompt is built host-side from the benchmark's full
    per-question metadata (LoCoMo renders a different instruction per
    category), so its text would leak the label the hook is not given."""

    __slots__ = ("_messages",)

    def __init__(self, messages: Messages):
        self._messages = [dict(m) for m in messages]

    def with_instruction(self, text: str) -> "SealedPrompt":
        """A copy with `text` appended to the last user turn."""
        msgs = [dict(m) for m in self._messages]
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].get("role") == "user":
                msgs[i]["content"] = f"{msgs[i].get('content', '')}\n\n{text}"
                break
        return SealedPrompt(msgs)

    def _unseal(self) -> Messages:
        return [dict(m) for m in self._messages]

    def __repr__(self) -> str:
        return "<SealedPrompt: the benchmark's QA prompt; send it with qa.ask()>"

    __str__ = __repr__

    def __iter__(self):
        raise TypeError("SealedPrompt is not readable — send it with qa.ask() or build your own messages")

    __getitem__ = __len__ = __iter__


class QAContext:
    """The run's QA model plus the benchmark's default prompt, for one query."""

    def __init__(self, *, agent: Any,
                 default_prompt: Optional[Callable[[str, Dict], Messages]] = None,
                 reasoning_effort: Optional[str] = None):
        self._agent = agent
        self._default_prompt = default_prompt
        self._reasoning_effort = reasoning_effort

    @property
    def model(self) -> str:
        """The QA model this run answers with. Read-only."""
        return str(getattr(self._agent, "model", ""))

    def default_prompt(self, query: str, retrieved: Dict, _ignored: Any = None, *,
                       extra_instruction: str = "") -> SealedPrompt:
        """The benchmark's standard QA prompt over `retrieved`, sealed. Send it
        as-is with `ask`, or add an instruction of your own with
        `extra_instruction=` (appended to the user turn); to change the prompt
        itself, write your own message list instead.

        The benchmark's per-question metadata is bound host-side; a third
        positional argument is accepted and ignored."""
        sealed = SealedPrompt(self._default_prompt(query, retrieved))
        return sealed.with_instruction(extra_instruction) if extra_instruction else sealed

    async def ask(self, messages: Union[Messages, SealedPrompt]) -> str:
        """Send `messages` to the QA model and return its text.

        `messages` is a SealedPrompt from `default_prompt`, or one system
        message (may be empty) followed by one or more user / assistant
        turns. A single user turn is sent exactly as the benchmark's shared
        agent always sent it; a longer conversation is sent as-is.
        """
        if isinstance(messages, SealedPrompt):
            messages = messages._unseal()
        if not messages:
            raise ValueError("QAContext.ask: empty message list")
        if messages[0].get("role") == "system":
            system, rest = str(messages[0].get("content", "")), list(messages[1:])
        else:
            system, rest = "", list(messages)
        if not rest:
            raise ValueError("QAContext.ask: no user message after the system message")
        self._agent.messages = [{"role": "system", "content": system}]
        if len(rest) == 1:
            # The exact call the benchmarks' shared agent always made
            # (with_history is the Agent default).
            out = await self._agent.ask(str(rest[0].get("content", "")),
                                        reasoning_effort=self._reasoning_effort)
        else:
            out = await self._agent.ask([{"role": "system", "content": system}] + rest,
                                        with_full_msg=True, reasoning_effort=self._reasoning_effort)
        return "" if out is None else str(out)
