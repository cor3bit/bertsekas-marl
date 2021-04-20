from abc import ABC, abstractmethod
from typing import List, Tuple


class Agent(ABC):
    id = None
    _type = None

    @abstractmethod
    def act(
            self,
            obs: List[float],
            **kwargs,
    ) -> int:
        raise NotImplementedError


class MultiAgent(ABC):
    _type = None

    @abstractmethod
    def act_n(
            self,
            obs_n: List[List[float]],
            **kwargs,
    ) -> Tuple[int, int, int, int]:
        raise NotImplementedError
