import logging  # noqa: F401

from d2g_evaluation.metrics.interface_metrics import InterfaceMetrics
from labelstudio_snapshot.pipelines.base import BaseEval, TaskGroup  # noqa: F401


class EvalHumanVsHuman(BaseEval):
    def __init__(self) -> None:
        super().__init__()
        self.interface_metrics = InterfaceMetrics()
