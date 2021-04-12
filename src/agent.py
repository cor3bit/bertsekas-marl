from abc import ABC, abstractmethod
from typing import Iterable, Dict


class Agent(ABC):
    _model_based = None
    _on_policy = None
    _type = None

    @abstractmethod
    def act(
            self,
            obs: Iterable[float],
            **kwargs,
    ) -> int:
        raise NotImplementedError
