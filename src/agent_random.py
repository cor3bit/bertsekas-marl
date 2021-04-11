from src.agent import Agent
from src.constants import AgentType


class RandomAgent(Agent):
    def __init__(self, action_space):
        self._action_space = action_space
        self._type = AgentType.RANDOM

    def act(self, obs, **kwargs):
        return self._action_space.sample()

    def train(self, env, n_episodes, save_weights=True):
        raise NotImplementedError('Not available for Random Agent!')
