import time
from typing import List

from tqdm import tqdm
import numpy as np
import gym
import ma_gym  # register new envs on import
from ma_gym.wrappers import Monitor

from src.constants import SpiderAndFlyEnv, AgentType
from src.agent_random import RandomAgent
from src.agent_rule_based import RuleBasedAgent
from src.agent_seq_rollout import SequentialRolloutAgent
from src.agent_qnet_based import ApproxRolloutAgent

N_EPISODES = 3
AGENT_TYPE = AgentType.QNET_BASED


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
    elif agent_type == AgentType.SEQ_MA_ROLLOUT:
        agents = [SequentialRolloutAgent(i, m_agents, p_preys, grid_shape, env.action_space[i])
                  for i in range(m_agents)]
    elif agent_type == AgentType.QNET_BASED:
        agents = [ApproxRolloutAgent(i, m_agents, p_preys, grid_shape, env.action_space[i])
                  for i in range(m_agents)]
    else:
        raise ValueError(f'Unrecognized agent type: {agent_type}.')

    return agents


if __name__ == '__main__':
    np.random.seed(42)

    # create Spider-and-Fly game
    env = gym.make(SpiderAndFlyEnv)
    env.seed(42)
    # env = Monitor(env, directory='../artifacts/recordings', force=True,)

    for i_episode in tqdm(range(N_EPISODES)):
        # init env
        # obs_n = env.reset()
        obs_n = env.reset_default()
        env.render()

        # init agents
        agents = create_agents(env, AGENT_TYPE)

        # init stopping condition
        done_n = [False] * env.n_agents

        total_reward = .0

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

            total_reward += np.sum(reward_n)

            time.sleep(0.5)
            env.render()

        print(f'Episode {i_episode}: Avg Reward is {total_reward / env.n_agents}')

    time.sleep(2.)

    env.close()
