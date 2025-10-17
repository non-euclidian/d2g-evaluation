import logging
import random
from abc import ABC, abstractmethod
from typing import Any, ClassVar


class BaseEvaluationReport(ABC):
    SEED: ClassVar[int] = 414242

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.rng_generator = random.Random(self.SEED)

    @abstractmethod
    def generate_report(self, *args: Any, **kwargs: Any) -> Any: ...  # noqa: ANN401
