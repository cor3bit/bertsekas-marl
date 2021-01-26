import numpy as np


class BaselineAgent:
    def __init__(self, agent_id, n_agents, n_preys, action_space):
        self._agent_id = agent_id
        self._n_agents = n_agents
        self._n_preys = n_preys
        self._action_space = action_space

    def act(self, action_distances):
        row_min, col_min = np.unravel_index(action_distances.argmin(), action_distances.shape)
        return row_min

    def train(self, env, n_episodes, save_weights=True):
        raise NotImplementedError('Not available for Baseline Agent!')
