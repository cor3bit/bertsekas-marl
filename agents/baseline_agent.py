import numpy as np


class BaselineAgent:
    def __init__(self, state_space, action_space, load_policy=False):
        self._state_space = state_space
        self._action_space = action_space
        self._pi = {}  # policy, state -> probability over actions

        if load_policy:
            raise NotImplementedError()
        else:
            self._fill_policy_table()

    def act(self, observation):
        if not self._pi:
            raise ValueError('Tabular Policy is not initialized!')

        possible_actions = self._pi[observation]
        return np.argmax(possible_actions)

    def train(self, env, n_episodes, save_weights=True):
        raise NotImplementedError('Not available for Baseline Agent!')

    def _fill_policy_table(self):

        b = self._state_space

        a = 1
