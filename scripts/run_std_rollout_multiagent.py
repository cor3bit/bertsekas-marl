import time
from typing import List

from tqdm import tqdm
import numpy as np
import gym
import ma_gym  # register new envs on import
from ma_gym.wrappers.monitor import Monitor

from src.constants import SpiderAndFlyEnv, AgentType
from src.agent_std_rollout import StdRolloutMultiAgent

N_EPISODES = 1
N_SIMS_PER_MC = 50

if __name__ == '__main__':
    np.random.seed(42)

    # create Spider-and-Fly game
    env = gym.make(SpiderAndFlyEnv)
    env.seed(42)
    # env = Monitor(env, directory='../artifacts/recordings', force=True, )

    for i_episode in tqdm(range(N_EPISODES)):
        # init env
        # obs_n = env.reset()
        obs_n = env.reset_default()
        env.render()

        # init env variables
        m_agents = env.env.n_agents
        p_preys = env.env.n_preys
        grid_shape = env.env._grid_shape

        # init agents
        std_rollout_multiagent = StdRolloutMultiAgent(
            m_agents, p_preys, grid_shape, env.action_space[0], N_SIMS_PER_MC)

        # init stopping condition
        done_n = [False] * env.n_agents

        total_reward = .0

        # run an episode until all prey is caught
        while not all(done_n):
            act_n = std_rollout_multiagent.act_n(obs_n)

            # update step
            obs_n, reward_n, done_n, info = env.step(act_n)

            total_reward += np.sum(reward_n)

            # time.sleep(0.5)
            env.render()

        print(f'Episode {i_episode}: Avg Reward is {total_reward / env.n_agents}')

    # time.sleep(2.)

    env.close()
