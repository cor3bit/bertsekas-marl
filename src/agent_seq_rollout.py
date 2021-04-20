from typing import List, Dict, Iterable, Tuple
from itertools import product
from operator import itemgetter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import gym
import ma_gym

from src.constants import SpiderAndFlyEnv, AgentType
from src.agent_rule_based import RuleBasedAgent
from src.agent_qnet_based import QnetBasedAgent
from src.agent import Agent


class SeqRolloutAgent(Agent):
    def __init__(
            self,
            agent_id: int,
            m_agents: int,
            p_preys: int,
            grid_shape: Tuple[int, int],
            action_space: gym.spaces.Discrete,
            n_sim_per_step: int = 10,
            basis_agent_type: str = AgentType.RULE_BASED,
            qnet_type: str = None,
            n_workers: int = 12,
    ):
        self.id = agent_id
        self._m_agents = m_agents
        self._p_preys = p_preys
        self._grid_shape = grid_shape
        self._action_space = action_space
        self._n_sim_per_step = n_sim_per_step
        self._agents = self._create_agents(basis_agent_type, qnet_type)
        self._n_workers = n_workers

    def act(
            self,
            obs: List[float],
            prev_actions: Dict[int, int] = None,
            **kwargs,
    ) -> int:
        best_action, action_q_values = self.act_with_info(obs, prev_actions)
        return best_action

    def act_with_info(
            self,
            obs: List[float],
            prev_actions: Dict[int, int] = None,
    ) -> Tuple[int, np.ndarray]:
        assert prev_actions is not None

        n_actions = self._action_space.n

        # parallel calculations
        sim_results = []
        with ProcessPoolExecutor(max_workers=self._n_workers) as pool:
            futures = []

            # calculate first step
            for action_id in range(n_actions):
                first_step_prev_actions = dict(prev_actions)
                act_n = np.empty((self._m_agents,), dtype=np.int8)
                for i in range(self._m_agents):
                    if i in prev_actions:
                        act_n[i] = prev_actions[i]
                    elif self.id == i:
                        act_n[i] = action_id
                        first_step_prev_actions[i] = action_id
                    else:
                        underlying_agent = self._agents[i]
                        assumed_action = underlying_agent.act(obs, prev_actions=first_step_prev_actions)
                        act_n[i] = assumed_action
                        first_step_prev_actions[i] = assumed_action

                # submit simulation with config (simulated_action, simulation_id)
                for simulation_id in range(self._n_sim_per_step):
                    futures.append(pool.submit(
                        self._simulate_episode_par, action_id, simulation_id, obs, act_n, self._agents,
                    ))

            for f in as_completed(futures):
                res = f.result()
                sim_results.append(res)

        # analyze results of the simulation
        np_results = np.array(sim_results, np.float32)
        action_q_values = np.empty(shape=(n_actions,), dtype=np.float32)
        for action_id in range(n_actions):
            np_results_action = np_results[np_results[:, 0] == action_id]
            action_q_values[action_id] = np.mean(np_results_action[:, 2])

        best_action = np.argmax(action_q_values)

        return best_action, action_q_values

    def _create_agents(self, agent_type, qnet_type):
        if agent_type == AgentType.RULE_BASED:
            agents = [RuleBasedAgent(
                i, self._m_agents, self._p_preys, self._grid_shape, self._action_space,
            ) for i in range(self._m_agents)]
        elif agent_type == AgentType.QNET_BASED:
            agents = [QnetBasedAgent(
                i, self._m_agents, self._p_preys, self._grid_shape, self._action_space, qnet_type=qnet_type,
            ) for i in range(self._m_agents)]
        else:
            raise ValueError(f'Invalid agent type: {agent_type}.')

        return agents

    @staticmethod
    def _simulate_episode_par(
            action_id: int,
            simulation_id: int,
            initial_obs: List[float],
            initial_step: np.ndarray,
            agents: List[Agent],
    ):
        # create env
        env = gym.make(SpiderAndFlyEnv)

        # init env from observation
        obs_n = env.reset_from(initial_obs)

        # make prescribed first step
        obs_n, reward_n, done_n, info = env.step(initial_step)
        total_reward = np.sum(reward_n)

        # run simulation
        while not all(done_n):
            act_n = []
            prev_actions = {}
            for agent, obs in zip(agents, obs_n):
                best_action = agent.act(obs, prev_actions=prev_actions)
                act_n.append(best_action)
                prev_actions[agent.id] = best_action

            obs_n, reward_n, done_n, info = env.step(act_n)
            total_reward += np.sum(reward_n)

        env.close()

        total_reward /= len(agents)

        return action_id, simulation_id, total_reward
