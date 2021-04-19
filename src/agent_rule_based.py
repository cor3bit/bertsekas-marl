from typing import List, Tuple

import numpy as np
import gym
import ma_gym

from src.constants import SpiderAndFlyEnv
from src.agent import Agent


class RuleBasedAgent(Agent):
    def __init__(
            self,
            agent_id: int,
            m_agents: int,
            p_preys: int,
            grid_shape: Tuple[int, int],
            action_space: gym.spaces.Discrete,
    ):
        self.id = agent_id
        self._m = m_agents
        self._p = p_preys
        self._action_space = action_space

        # keep env to access specific method
        self._model = gym.make(SpiderAndFlyEnv)
        assert self._model._grid_shape == grid_shape

    def act(
            self,
            obs: List[float],
            **kwargs,
    ) -> int:
        best_action, action_distances = self.act_with_info(obs)
        return best_action

    def act_with_info(
            self,
            obs,
    ) -> Tuple[int, np.ndarray]:
        curr_pos = self._get_agent_pos(obs)
        alive_prey_coords = self._get_alive_prey_coords(obs)
        action_distances = self._get_action_distances(curr_pos, alive_prey_coords)

        return action_distances.argmin(), action_distances

    def _get_action_distances(
            self,
            curr_pos: Tuple[int, int],
            alive_prey_coords: List[Tuple[float, float]],
    ):
        n_actions = self._action_space.n

        action_distances = np.full((n_actions,), fill_value=np.inf, dtype=np.float32)
        for action_id in range(n_actions):
            next_pos = self._model._apply_action(curr_pos, action_id)
            if next_pos is not None:
                min_d = np.inf
                for alive_prey_row, alive_prey_col in alive_prey_coords:
                    d = np.abs(next_pos[0] - alive_prey_row) + np.abs(next_pos[1] - alive_prey_col)
                    if d < min_d:
                        min_d = d

                action_distances[action_id] = min_d

        return action_distances

    def _convert_to_pos(
            self,
            pos_scaled: Tuple[np.float32, np.float32],
    ) -> Tuple[int, int]:
        grid_row, grid_col = self._model._grid_shape
        row_pos_scaled, col_pos_scaled = pos_scaled
        row_pos = int(np.round((grid_row - 1) * row_pos_scaled, 0))
        col_pos = int(np.round((grid_col - 1) * col_pos_scaled, 0))
        return row_pos, col_pos

    def _get_agent_pos(
            self,
            obs: np.ndarray,
    ) -> Tuple[int, int]:
        start_ind = int(self.id * 2)
        row_pos_scaled, col_pos_scaled = obs[start_ind], obs[start_ind + 1]
        row_pos, col_pos = self._convert_to_pos((row_pos_scaled, col_pos_scaled))
        return row_pos, col_pos

    def _get_alive_prey_coords(
            self,
            obs: np.array,
    ) -> List[Tuple[float, float]]:
        alive_prey_coords = []

        preys_alive = obs[-self._p:]

        for prey_alive, prey_id in zip(preys_alive, range(self._p)):
            if prey_alive:
                start_ind = int(self._m * 2 + prey_id * 2)
                row_pos_scaled, col_pos_scaled = obs[start_ind], obs[start_ind + 1]
                pos = self._convert_to_pos((row_pos_scaled, col_pos_scaled))
                alive_prey_coords.append(pos)

        assert alive_prey_coords

        return alive_prey_coords
