import time
from typing import List

import numpy as np
import gym
import ma_gym  # register new envs on import
from tqdm import tqdm

from src.constants import SpiderAndFlyEnv, AgentType
from src.agent_random import RandomAgent
from src.agent_rule_based import RuleBasedAgent
from src.agent_exact_rollout import ExactRolloutAgent

SEED = 42
N_EPISODES = 10


def create_agents(env: gym.Env, agent_type: str) -> List:
    # init env variables
    m_agents = env.n_agents
    p_preys = env.n_preys
    grid_shape = env._grid_shape

    if agent_type == AgentType.RANDOM:
        agents = [RandomAgent(env.action_space[i]) for i in range(m_agents)]
    elif agent_type == AgentType.RULE_BASED:
        agents = [RuleBasedAgent(i, m_agents, p_preys, grid_shape, env.action_space[i])
                  for i in range(m_agents)]
    elif agent_type == AgentType.EXACT_ROLLOUT:
        agents = [ExactRolloutAgent(i, m_agents, p_preys, grid_shape, env.action_space[i])
                  for i in range(m_agents)]
    else:
        raise ValueError(f'Unrecognized agent type: {agent_type}.')

    return agents


def run_agent(agent_type):
    # create Spider-and-Fly game
    env = gym.make(SpiderAndFlyEnv)
    env.seed(SEED)

    avg_steps = 0

    for _ in tqdm(range(N_EPISODES)):
        # init env
        # obs_n = env.reset()
        obs_n = env.reset()

        # init agents
        agents = create_agents(env, agent_type)

        # init stopping condition
        done_n = [False] * env.n_agents

        # run an episode until all prey is caught
        while not all(done_n):
            prev_actions = {}
            act_n = []
            for i, (agent, obs) in enumerate(zip(agents, obs_n)):
                if isinstance(agent, ExactRolloutAgent):
                    action_id = agent.act(obs, prev_actions)
                else:
                    action_id = agent.act(obs, )

                prev_actions[i] = action_id
                act_n.append(action_id)

            # update step
            obs_n, reward_n, done_n, info = env.step(act_n)

        avg_steps += env._step_count

    env.close()

    avg_steps /= N_EPISODES

    return avg_steps


if __name__ == '__main__':
    # np.random.seed(SEED)

    agent_type = AgentType.RULE_BASED
    print(f'Running {agent_type} Agent.')
    steps = run_agent(agent_type)
    print(f'Avg steps for {agent_type} Agent: {steps}.')

    agent_type = AgentType.EXACT_ROLLOUT
    print(f'Running {agent_type} Agent.')
    steps = run_agent(agent_type)
    print(f'Avg steps for {agent_type} Agent: {steps}.')
