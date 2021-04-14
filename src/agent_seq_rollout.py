from typing import List, Dict, Iterable, Tuple
from itertools import product
from operator import itemgetter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import gym
import ma_gym

from src.constants import SpiderAndFlyEnv
from src.agent_rule_based import RuleBasedAgent
from src.agent import Agent


class SequentialRolloutAgent(Agent):
    def __init__(
            self,
            agent_id: int,
            m_agents: int,
            p_preys: int,
            grid_shape: Tuple[int, int],
            action_space: gym.spaces.Discrete,
            n_sim_per_step=10,
            n_workers=5,
    ):
        self.id = agent_id
        self._m_agents = m_agents
        self._p_preys = p_preys
        self._grid_shape = grid_shape
        self._action_space = action_space
        self._n_sim_per_step = n_sim_per_step
        self._n_workers = n_workers

    def act(
            self,
            obs: Iterable[float],
            prev_actions: Dict[int, int] = None,
            **kwargs,
    ) -> int:
        assert prev_actions is not None

        n_actions = self._action_space.n

        # parallel vars
        sim_results = []

        with ProcessPoolExecutor(max_workers=self._n_workers) as pool:
            futures = []

            for action_id in range(n_actions):
                # 1st step - optimal actions from previous agents,
                # simulated step from current agent,
                # greedy (baseline) from undecided agents
                act_n = np.empty((self._m_agents,), dtype=np.int8)
                for i in range(self._m_agents):
                    if i in prev_actions:
                        act_n[i] = prev_actions[i]
                    elif self.id == i:
                        act_n[i] = action_id
                    else:
                        rb_agent = RuleBasedAgent(i, self._m_agents, self._p_preys,
                                                  self._grid_shape, self._action_space)
                        act_n[i] = rb_agent.act(obs)

                # run N simulations
                futures.append(pool.submit(
                    self._simulate, action_id, obs, act_n, self._m_agents, self._p_preys,
                    self._grid_shape, self._action_space, self._n_sim_per_step,
                ))

            for f in as_completed(futures):
                res = f.result()
                sim_results.append(res)

        best_action = max(sim_results, key=itemgetter(1))[0]

        return best_action

    @staticmethod
    def _simulate(
            action_id,
            initial_obs: Iterable[float],
            initial_step: np.array,
            m_agents,
            p_preys,
            grid_shape,
            action_space,
            n_sim_per_step,
    ) -> Tuple[int, float]:
        avg_total_reward = .0

        # create env
        env = gym.make(SpiderAndFlyEnv)

        # create agents
        agents = [RuleBasedAgent(i, m_agents,
                                 p_preys, grid_shape, action_space)
                  for i in range(m_agents)]

        # run N simulations
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

        return action_id, avg_total_reward
