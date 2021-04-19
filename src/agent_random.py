from typing import List

import gym

from src.agent import Agent
from src.constants import AgentType


class RandomAgent(Agent):
    def __init__(
            self,
            action_space: gym.spaces.Discrete,
    ):
        self._action_space = action_space
        self._type = AgentType.RANDOM

    def act(
            self,
            obs: List[float],
            **kwargs,
    ) -> int:
        return self._action_space.sample()
