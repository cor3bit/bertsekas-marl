import time
from typing import List

import numpy as np
import gym
import ma_gym  # register new envs on import
from tqdm import tqdm

from src.constants import SpiderAndFlyEnv, AgentType, QnetType
from src.agent_qnet_based import QnetBasedAgent
from src.agent_rule_based import RuleBasedAgent
from src.agent_seq_rollout import SeqRolloutAgent
from src.agent_std_rollout import StdRolloutMultiAgent

SEED = 42
N_EPISODES = 1_000
N_SIMS_MC = 50
WITH_STD_ROLLOUT = False
BASIS_AGENT_TYPE = AgentType.QNET_BASED
QNET_TYPE = QnetType.REPEATED


def create_agents(env: gym.Env, agent_type: str) -> List:
    # init env variables
    m_agents = env.n_agents
    p_preys = env.n_preys
    grid_shape = env._grid_shape

    if agent_type == AgentType.RULE_BASED:
        return [RuleBasedAgent(
            i, m_agents, p_preys, grid_shape, env.action_space[i],
        ) for i in range(m_agents)]
    elif agent_type == AgentType.QNET_BASED:
        return [QnetBasedAgent(
            i, m_agents, p_preys, grid_shape, env.action_space[i], qnet_type=QNET_TYPE,
        ) for i in range(m_agents)]
    elif agent_type == AgentType.SEQ_MA_ROLLOUT:
        return [SeqRolloutAgent(
            i, m_agents, p_preys, grid_shape, env.action_space[i],
            n_sim_per_step=N_SIMS_MC,
            basis_agent_type=BASIS_AGENT_TYPE,
            qnet_type=QNET_TYPE,
        ) for i in range(m_agents)]
    else:
        raise ValueError(f'Unrecognized agent type: {agent_type}.')


def run_agent(agent_type: str):
    # create Spider-and-Fly game
    env = gym.make(SpiderAndFlyEnv)
    env.seed(SEED)

    avg_reward = 0.

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


def run_std_ma_agent():
    # create Spider-and-Fly game
    env = gym.make(SpiderAndFlyEnv)
    env.seed(SEED)

    # init env variables
    m_agents = env.n_agents
    p_preys = env.n_preys
    grid_shape = env._grid_shape

    avg_reward = 0.

    for _ in tqdm(range(N_EPISODES)):
        # init env
        obs_n = env.reset()
        # obs_n = env.reset_default()

        # init agents
        std_rollout_multiagent = StdRolloutMultiAgent(
            m_agents, p_preys, grid_shape, env.action_space[0], N_SIMS_MC)

        # init stopping condition
        done_n = [False] * env.n_agents

        # run an episode until all prey is caught
        while not all(done_n):
            act_n = std_rollout_multiagent.act_n(obs_n)

            # update step
            obs_n, reward_n, done_n, info = env.step(act_n)

            avg_reward += np.sum(reward_n)

    env.close()

    avg_reward /= env.n_agents
    avg_reward /= N_EPISODES

    return avg_reward


if __name__ == '__main__':
    np.random.seed(SEED)

    # AgentType.RULE_BASED, AgentType.SEQ_MA_ROLLOUT, AgentType.QNET_BASED
    for agent_type in [AgentType.QNET_BASED]:

        print(f'Running {agent_type} Agent.')

    #     avg_reward = run_agent(agent_type)
    #     print(f'Avg reward for {agent_type} Agent on {N_EPISODES} episodes: {avg_reward}.')

    # if WITH_STD_ROLLOUT:
    #     agent_type = AgentType.STD_MA_ROLLOUT
    #     print(f'Running {agent_type} Agent.')
    #     run_std_ma_agent()
    #     print(f'Avg reward for {agent_type} Agent on {N_EPISODES} episodes: {avg_reward}.')
