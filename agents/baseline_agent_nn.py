import numpy as np
import torch

from agents.qnetwork import QNetwork
from agents.constants import BaselineModelPath


class BaselineAgentNn:
    def __init__(self, agent_id, n_agents, action_space, qnet_name=None):
        self._agent_id = agent_id
        self._n_agents = n_agents
        self._action_space = action_space

        # load neural net on init
        self._nn = self._load_net(qnet_name)

    def act(self, obs):
        # 1 form 5 samples for each action
        # 2 call q-network
        # 3 arg max action

        qs = np.zeros(shape=(self._action_space,), dtype=np.float)

        for i in range(self._action_space):
            x = self._convert_to_x(obs, i)
            q = self._nn(x)
            qs[i] = q

        return np.argmax(qs)

    def train(self, env, n_episodes, save_weights=True):
        raise NotImplementedError('Not implemented for Baseline Agent!')

    def _load_net(self, qnet_name):
        net = QNetwork()
        net.load_state_dict(torch.load(BaselineModelPath))

        # set dropout and batch normalization layers to evaluation mode
        net.eval()

        return net

    def _convert_to_x(self, obs, action_id):
        raise NotImplementedError
