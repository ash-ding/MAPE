"""Tests for the standardized memory-system eval contract.

Run:
    uv run pytest -q tests/test_eval_contract.py
"""
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_build_and_retrieve_are_required_compose_is_optional():
    from core.memo_class import MemoClass

    class Bare(MemoClass):
        pass

    class Misspelled(MemoClass):
        async def build_memory_from_data(self, recorder): return None
        async def retrieve_memory_for_querry(self, recorder): return {}   # typo

    for cls, missing in ((Bare, "build_memory_from_data"), (Misspelled, "retrieve_memory_for_query")):
        try:
            cls()
        except TypeError as e:
            assert missing in str(e), str(e)
        else:
            raise AssertionError(f"{cls.__name__} must not be instantiable")

    class Minimal(MemoClass):
        async def build_memory_from_data(self, recorder): return None
        async def retrieve_memory_for_query(self, recorder): return {}

    m = Minimal()
    assert not hasattr(m, "database")
    # Answering IS a hook, with a default a memo may override; a memo that
    # does not override it answers exactly as the shared QA path always did.
    assert type(m).compose_memory_context is MemoClass.compose_memory_context


def test_harness_loaders_name_the_unimplemented_hook():
    """The harness loader picks the concrete class and, when there is none,
    says which hook is missing."""
    import tempfile
    from pathlib import Path
    from core.memo_select import select_memo_class
    from evolution.launch import _load_harness_class

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "memo.py").write_text(
            "from evolution.memo_class import MemoClass\n"
            "class Half(MemoClass):\n"
            "    async def build_memory_from_data(self, recorder): return None\n",
            encoding="utf-8")
        try:
            _load_harness_class(d)
        except ImportError as e:
            assert "retrieve_memory_for_query" in str(e) and "Half" in str(e), str(e)
        else:
            raise AssertionError("_load_harness_class accepted an abstract harness")

    from core.memo_class import MemoClass

    class _AbstractHelper(MemoClass):
        async def build_memory_from_data(self, recorder): return None

    class _Concrete(_AbstractHelper):
        async def retrieve_memory_for_query(self, recorder): return {}

    # an abstract helper base listed first is skipped, not returned
    assert select_memo_class([MemoClass, _AbstractHelper, _Concrete], "x.py") is _Concrete


def test_loads_the_harness_class_not_the_base():
    """The class-discriminator must pick the harness's own class, not the
    imported base."""
    from pathlib import Path
    from evolution.launch import _load_harness_class
    seed = Path(__file__).resolve().parents[1] / "baselines" / "no_memory"
    assert _load_harness_class(seed).__name__ == "NoMemoryMemo"


def test_phase1_update_calls_build_memory_once():
    """The workflow hands the whole data in ONE build_memory_from_data call."""
    from core.workflow import BaseWorkflow
    from core.memo_class import MemoClass

    calls = []

    class _Memo(MemoClass):
        async def build_memory_from_data(self, recorder):
            calls.append(list(recorder.init.get("items", [])))
        async def retrieve_memory_for_query(self, recorder): return {}

    # minimal BaseWorkflow with the 7 abstract hooks stubbed + a recorder that
    # stores init under "items" via phase1_log_init.
    class _Rec:
        def __init__(self): self.init = {}
    class _W(BaseWorkflow):
        recorder_class = _Rec
        async def load_user_data(self, *a, **k): return None
        async def phase1_log_init(self, r, chunk): r.init = {"items": chunk}
        def build_query_recorder_init(self, *a, **k): return {}
        def build_qa_prompt(self, *a, **k): return [{"content": ""}, {"content": ""}]
        def extract_relevant_context(self, *a, **k): return None
        def build_qa_metadata(self, *a, **k): return {}
        async def log_qa_step(self, *a, **k): return None
    w = _W(memo_class=_Memo, model="gpt-5-mini")
    m = _Memo()
    asyncio.new_event_loop().run_until_complete(w._phase1_update(m, [1, 2, 3, 4, 5]))
    assert calls == [[1, 2, 3, 4, 5]]   # ONE call, whole data (not chunked by the workflow)


def test_memo_class_is_pure_contract():
    """core.memo_class holds only the contract: no baseline vocabulary
    (Sub_memo_layer), and Basic_Recorder is defined in core/recorder.py."""
    import core.memo_class as hb
    import core.recorder as rec

    # Sub_memo_layer is gone from memo_class (not defined, not re-exported)
    assert not hasattr(hb, "Sub_memo_layer"), \
        "Sub_memo_layer leaked back into core.memo_class"

    assert rec.Basic_Recorder.__module__ == "core.recorder"



def test_shared_answer_site_composes_then_answers_in_its_own_phases():
    """Every benchmark composes under `tokens.phase(tokens.COMPOSE)` and then
    answers under `tokens.phase(tokens.ANSWER)`, in that order. The memo's own
    calls inside the hook are memory-system cost; the host's single QA call is
    the answer cost, and its size is what the memo chose."""
    import inspect
    from core import workflow as bw
    from core.memo_class import MemoClass
    assert callable(getattr(MemoClass, "compose_memory_context"))
    for src in (inspect.getsource(bw.BaseWorkflow.run_single_user),):
        assert "memo.compose_memory_context(" in src
        compose_phase = src.index("tokens.phase(tokens.COMPOSE)")
        hook = src.index("memo.compose_memory_context(")
        answer_phase = src.index("tokens.phase(tokens.ANSWER)")
        assert compose_phase < hook < answer_phase, "compose, then answer"
        assert "check_retrieved(retrieved)" in src
