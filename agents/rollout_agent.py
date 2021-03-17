import numpy as np
import torch

from agents.qnetwork_rollout import QNetworkRollout
from agents.constants import RolloutModelPath_10x10_4v2


class RolloutAgent:
    def __init__(self, agent_id, n_agents, n_preys, action_space, qnet_name=None):
        self.id = agent_id
        self._n_agents = n_agents
        self._n_preys = n_preys
        self._action_space = action_space

        # load neural net on init
        self._nn = self._load_net(qnet_name)

    def act(self, obs, prev_actions, epsilon=0.05):
        # 1) form 5 samples for each action
        # 2) call q-network
        # 3) arg max action OR random (epsilon greedy)
        p = np.random.random()
        if p < epsilon:
            # random action -> exploration
            return self._action_space.sample()
        else:
            # argmax -> exploitation
            x = self._convert_to_x(obs, prev_actions)
            x = np.reshape(x, newshape=(1, -1))
            v = torch.from_numpy(x)
            qs = self._nn(v)
            return np.argmax(qs.data.numpy())

    def train(self, env, n_episodes, save_weights=True):
        raise NotImplementedError('Not implemented for Rollout Agent!')

    def _load_net(self, qnet_name=None):
        net = QNetworkRollout(self._n_agents, self._n_preys, self._action_space.n)
        net.load_state_dict(torch.load(RolloutModelPath_10x10_4v2 if qnet_name is None else qnet_name))

        # set dropout and batch normalization layers to evaluation mode
        net.eval()

        return net

    def _convert_to_x(self, obs, prev_actions):
        assert len(prev_actions) < self._n_agents

        # state
        obs_first = np.array(obs, dtype=np.float32).flatten()

        # agent ohe
        agent_ohe = np.zeros(shape=(self._n_agents,), dtype=np.float32)
        agent_ohe[self.id] = 1.

        # previous actions
        n_actions = self._action_space.n
        prev_actions_ohe = np.zeros(shape=(self._n_agents * n_actions,), dtype=np.float32)
        if prev_actions:
            for agent_id, prev_action in enumerate(prev_actions):
                # TODO check!!!
                pos = int(agent_id * n_actions + prev_action)
                prev_actions_ohe[pos] = 1.

        x = np.concatenate((obs_first, agent_ohe, prev_actions_ohe))

        return x