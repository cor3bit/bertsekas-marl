import time
from typing import List

import numpy as np
import gym
import ma_gym  # register new envs on import
from tqdm import tqdm

from src.constants import SpiderAndFlyEnv, AgentType
from src.agent_approx_rollout import RolloutAgent
from src.agent_rule_based import RuleBasedAgent
from src.agent_exact_rollout import ExactRolloutAgent

SEED = 42
N_EPISODES = 100
N_SIMS_MC = 50


def create_agents(env: gym.Env, agent_type: str) -> List:
    # init env variables
    m_agents = env.n_agents
    p_preys = env.n_preys
    grid_shape = env._grid_shape

    if agent_type == AgentType.RULE_BASED:
        agents = [RuleBasedAgent(i, m_agents, p_preys, grid_shape, env.action_space[i])
                  for i in range(m_agents)]
    elif agent_type == AgentType.EXACT_ROLLOUT:
        agents = [ExactRolloutAgent(i, m_agents, p_preys, grid_shape, env.action_space[i],
                                    n_sim_per_step=N_SIMS_MC)
                  for i in range(m_agents)]
    elif agent_type == AgentType.APRX_ROLLOUT:
        agents = [RolloutAgent(i, m_agents, p_preys, grid_shape, env.action_space[i])
                  for i in range(m_agents)]
    else:
        raise ValueError(f'Unrecognized agent type: {agent_type}.')

    return agents


def run_agent(agent_type):
    # create Spider-and-Fly game
    env = gym.make(SpiderAndFlyEnv)
    env.seed(SEED)

    avg_reward = 0

    for _ in tqdm(range(N_EPISODES)):
        # init env
        obs_n = env.reset()
        # obs_n = env.reset_default()

        # init agents
        agents = create_agents(env, agent_type)

        # init stopping condition
        done_n = [False] * env.n_agents

        # run an episode until all prey is caught
        while not all(done_n):
            prev_actions = {}
            act_n = []
            for i, (agent, obs) in enumerate(zip(agents, obs_n)):
                action_id = agent.act(obs, prev_actions=prev_actions)

                prev_actions[i] = action_id
                act_n.append(action_id)

            # update step
            obs_n, reward_n, done_n, info = env.step(act_n)
            avg_reward += np.sum(reward_n)

    env.close()

    avg_reward /= env.n_agents
    avg_reward /= N_EPISODES

    return avg_reward


if __name__ == '__main__':
    np.random.seed(SEED)

    for agent_type in [AgentType.RULE_BASED, AgentType.EXACT_ROLLOUT]:
        print(f'Running {agent_type} Agent.')
        avg_reward = run_agent(agent_type)
        print(f'Avg reward for {agent_type} Agent on {N_EPISODES} episodes: {avg_reward}.')
