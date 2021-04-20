from typing import List, Dict, Iterable, Tuple
from itertools import product
from operator import itemgetter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import gym
import ma_gym

from src.agent import MultiAgent
from src.constants import SpiderAndFlyEnv
from src.agent_rule_based import RuleBasedAgent


class StdRolloutMultiAgent(MultiAgent):
    def __init__(
            self,
            m_agents: int,
            p_preys: int,
            grid_shape: Tuple[int, int],
            action_space: gym.spaces.Discrete,
            n_sim_per_step: int = 10,
            n_workers: int = 10,
    ):
        self._m_agents = m_agents
        self._p_preys = p_preys
        self._grid_shape = grid_shape
        self._action_space = action_space
        self._n_sim_per_step = n_sim_per_step
        self._n_workers = n_workers

    def act_n(
            self,
            obs_n: List[List[float]],
            **kwargs,
    ) -> Tuple[int, int, int, int]:
        n_actions = self._action_space.n
        available_moves = list(range(n_actions))
        configs = list(product(available_moves, repeat=self._m_agents))
        obs = obs_n[0]

        # parallel vars
        sim_results = []

        with ProcessPoolExecutor(max_workers=self._n_workers) as pool:
            futures = []
            for config in configs:
                futures.append(pool.submit(
                    self._simulate, obs, config, self._m_agents, self._p_preys,
                    self._grid_shape, self._action_space, self._n_sim_per_step,
                ))

            for f in as_completed(futures):
                res = f.result()
                sim_results.append(res)

        best_config = max(sim_results, key=itemgetter(1))[0]

        return best_config

    @staticmethod
    def _simulate(
            initial_obs: List[float],
            initial_step: Tuple,
            m_agents: int,
            p_preys: int,
            grid_shape: Tuple[int, int],
            action_space: gym.spaces.Discrete,
            n_sim_per_step: int,
    ) -> Tuple[Tuple, float]:
        # create env
        env = gym.make(SpiderAndFlyEnv)

        # create agents
        agents = [RuleBasedAgent(i, m_agents,
                                 p_preys, grid_shape, action_space)
                  for i in range(m_agents)]

        # run N simulations
        avg_total_reward = .0
        for _ in range(n_sim_per_step):
            obs_n = env.reset_from(initial_obs)

            # 1 step
            obs_n, reward_n, done_n, info = env.step(initial_step)
            avg_total_reward += np.sum(reward_n)

            # run an episode until all prey is caught
            while not all(done_n):
                # all agents act based on the observation
                act_n = []
                for agent, obs in zip(agents, obs_n):
                    act_n.append(agent.act(obs))

                # update step
                obs_n, reward_n, done_n, info = env.step(act_n)

                avg_total_reward += np.sum(reward_n)

        env.close()

        avg_total_reward /= m_agents
        avg_total_reward /= n_sim_per_step

        return initial_step, avg_total_reward
