import numpy as np
import torch

from agents.qnetwork import QNetwork
from agents.constants import BaselineModelPath_10x10_4v2


class BaselineAgentNn:
    def __init__(self, agent_id, n_agents, n_preys, action_space, qnet_name=None):
        self.id = agent_id
        self._n_agents = n_agents
        self._n_preys = n_preys
        self._action_space = action_space

        # load neural net on init
        self._nn = self._load_net(qnet_name)

    def act(self, obs):
        # 1 form 5 samples for each action
        # 2 call q-network
        # 3 arg max action
        x = self._convert_to_x(obs)
        x = np.reshape(x, newshape=(1, -1))
        v = torch.from_numpy(x)
        qs = self._nn(v.float())

        return np.argmax(qs.data.numpy())

    def train(self, env, n_episodes, save_weights=True):
        raise NotImplementedError('Not implemented for Baseline Agent!')

    def _load_net(self, qnet_name=None):
        net = QNetwork(self._n_agents, self._n_preys)
        net.load_state_dict(torch.load(BaselineModelPath_10x10_4v2 if qnet_name is None else qnet_name))

        # set dropout and batch normalization layers to evaluation mode
        net.eval()

        return net

    def _convert_to_x(self, obs):
        obs_first = np.array(obs).flatten()

        agent_ohe = np.zeros(shape=(self._n_agents,), dtype=np.float)
        agent_ohe[self.id] = 1.

        x = np.concatenate((obs_first, agent_ohe))

        return x
