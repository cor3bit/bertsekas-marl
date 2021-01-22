import time

import gym

import ma_gym  # register new envs on import
from agents.constants import SpiderAndFlyEnv

if __name__ == '__main__':
    # create Spider-and-Fly game
    env = gym.make(SpiderAndFlyEnv)

    # init env
    obs_n = env.reset()

    # run 100 episodes for a random agent
    for i in range(10):
        env.render()
        obs_n, reward_n, done_n, info = env.step(env.action_space.sample())

        time.sleep(1.0)

    env.close()
