from typing import Tuple

import numpy as np
import torch
import gym

from src.qnetwork_coordinated import QNetworkCoordinated
from src.constants import RolloutModelPath_10x10_4v2
from src.agent import Agent
from src.agent_rule_based import RuleBasedAgent


class RolloutAgent(Agent):
    def __init__(
            self,
            agent_id: int,
            m_agents: int,
            p_preys: int,
            grid_shape: Tuple[int, int],
            action_space: gym.spaces.Discrete,
            n_sim_per_step: int = 10,
            qnet_name: str = None,
    ):
        self.id = agent_id
        self._m_agents = m_agents
        self._p_preys = p_preys
        self._action_space = action_space

        # load neural net on init
        self._nn = self._load_net(qnet_name)

        # for heuristic
        self._rb_agents = [RuleBasedAgent(i, m_agents, p_preys,
                                          grid_shape, action_space) for i in range(m_agents)]

    def act(self, obs, prev_actions=None, epsilon=0.0, **kwargs):
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

    def _load_net(self, qnet_name=None):
        net = QNetworkCoordinated(self._m_agents, self._p_preys, self._action_space.n)
        net.load_state_dict(torch.load(RolloutModelPath_10x10_4v2 if qnet_name is None else qnet_name))

        # set dropout and batch normalization layers to evaluation mode
        net.eval()

        return net

    def _convert_to_x(self, obs_n, prev_actions):

        # state
        obs_first = np.array(obs_n, dtype=np.float32).flatten()

        # agent ohe
        agent_ohe = np.zeros(shape=(self._m_agents,), dtype=np.float32)
        agent_ohe[self.id] = 1.

        # prev actions
        prev_actions_ohe = np.zeros(shape=(self._m_agents * self._action_space.n,), dtype=np.float32)
        for i in range(self._m_agents):
            if i == self.id:
                continue
            elif i in prev_actions:
                action_i = prev_actions[i]
                ohe_action_index = int(i * self._action_space.n) + action_i
                prev_actions_ohe[ohe_action_index] = 1.
            else:
                # TODO from RB agent OR ApproxAgent
                rb_agent_i = self._rb_agents[i]
                action_i = rb_agent_i.act(obs_first)
                ohe_action_index = int(i * self._action_space.n) + action_i
                prev_actions_ohe[ohe_action_index] = 1.

        # combine all
        x = np.concatenate((obs_first, agent_ohe, prev_actions_ohe))

        return x
