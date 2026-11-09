"""Map supported benchmark names to workflow, environment and recorder classes.

Shared by core evaluation, baseline evaluation and the evolution container runner.
"""
from __future__ import annotations

from types import ModuleType
from typing import Dict, List, Tuple

from benchmarks.locomo import env as locomo_env
from benchmarks.locomo.env import LoCoMoRecorder
from benchmarks.locomo.workflow import LoCoMoWorkflow
from benchmarks.longmemeval import env as lme_env
from benchmarks.longmemeval.env import LongMemEvalRecorder
from benchmarks.longmemeval.workflow import LongMemEvalWorkflow

# dataset → (workflow_cls, env_module, recorder_cls).
# env_module must expose get_task_list(status, eval_n_samples).
REGISTRY: Dict[str, Tuple[type, ModuleType, type]] = {
    "locomo":        (LoCoMoWorkflow,       locomo_env, LoCoMoRecorder),
    "longmemeval_s": (LongMemEvalWorkflow,  lme_env,    LongMemEvalRecorder),
}

DATASETS: List[str] = sorted(REGISTRY)


def resolve(dataset: str) -> Tuple[type, ModuleType, type]:
    """Return (workflow_cls, env_module, recorder_cls) for `dataset`."""
    if dataset not in REGISTRY:
        raise ValueError(
            f"unknown dataset {dataset!r}; supported datasets: {DATASETS}."
        )
    return REGISTRY[dataset]
