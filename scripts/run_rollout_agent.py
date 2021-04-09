import time

import numpy as np
import gym
import ma_gym  # register new envs on import
from ma_gym.wrappers import Monitor

from src.constants import SpiderAndFlyEnv
from src.agent_rollout import RolloutAgent

N_EPISODES = 5

if __name__ == '__main__':
    gym.logger.set_level(gym.logger.DEBUG)

    # create Spider-and-Fly game
    env = gym.make(SpiderAndFlyEnv)
    # env = Monitor(env, directory='../artifacts/recordings', force=True)

    action_space_n = env.action_space[0].n

    for _ in range(N_EPISODES):
        # init env
        # TODO DEMO!!!
        # obs_n = env.reset_default()
        obs_n = env.reset()

        # init agents
        n_agents = env.n_agents
        n_preys = env.n_preys
        agents = [RolloutAgent(i, n_agents, n_preys, env.action_space[i]) for i in range(n_agents)]

        # init stopping condition
        done_n = [False] * n_agents
        env.render()

        # run 100 episodes for a random agent
        while not all(done_n):

            # all agents act based on the observation
            # act_n = []
            obs = obs_n[0]

            for agent in agents:
                best_action = agent.act(obs)

                # act_n.append(best_action)

                sub_obs_n, sub_reward_n, sub_done_n, sub_info = env.substep(agent.id, best_action)

                # !!
                obs = sub_obs_n[0]

            # TODO check
            obs_n = sub_obs_n
            assert done_n is not None

            reward_n = sub_reward_n
            done_n = sub_done_n
            info = sub_info

            env.render()
            time.sleep(0.5)

            # update step ->
            # obs_n, reward_n, done_n, info = env.step(act_n)

    # env.render()
    time.sleep(2.)

    env.close()
