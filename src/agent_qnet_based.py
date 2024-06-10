from typing import Tuple, List, Dict

import numpy as np
import torch
import gym

from src.agent import Agent
from src.qnetwork_coordinated import QNetworkCoordinated
from src.constants import QnetType, RolloutModelPath_10x10_4v2, RepeatedRolloutModelPath_10x10_4v2


class QnetBasedAgent(Agent):
    def __init__(
            self,
            agent_id: int,
            m_agents: int,
            p_preys: int,
            grid_shape: Tuple[int, int],
            action_space: gym.spaces.Discrete,
            qnet_type: str,
    ):
        self.id = agent_id
        self._m_agents = m_agents
        self._p_preys = p_preys
        self._grid_shape = grid_shape
        self._action_space = action_space

        # load neural net on init
        qnet_name = RolloutModelPath_10x10_4v2 if qnet_type == QnetType.BASELINE else RepeatedRolloutModelPath_10x10_4v2
        self._nn = self._load_net(qnet_name)

    def act(
            self,
            obs: List[float],
            prev_actions: Dict[int, int] = None,
            epsilon: float = 0.0,
            **kwargs,
    ) -> int:
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

    def _load_net(
            self,
            qnet_name: str = None
    ) -> QNetworkCoordinated:
        net = QNetworkCoordinated(self._m_agents, self._p_preys, self._action_space.n)
        #net.load_state_dict(torch.load(qnet_name))
        net.load_state_dict(torch.load(qnet_name, map_location=torch.device('cpu')))

        # set dropout and batch normalization layers to evaluation mode
        net.eval()

        return net

    def _convert_to_x(
            self,
            obs: List[float],
            prev_actions: Dict[int, int] = None,
    ) -> np.ndarray:
        # state
        np_obs = np.array(obs, dtype=np.float32).flatten()

        # agent ohe
        agent_ohe = np.zeros(shape=(self._m_agents,), dtype=np.float32)
        agent_ohe[self.id] = 1.

        # prev actions
        prev_actions_ohe = np.zeros(shape=(self._m_agents * self._action_space.n,), dtype=np.float32)
        for agent_i, action_i in prev_actions.items():
            ohe_action_index = int(agent_i * self._action_space.n) + action_i
            prev_actions_ohe[ohe_action_index] = 1.

        # combine all
        x = np.concatenate((np_obs, agent_ohe, prev_actions_ohe))

        return x
