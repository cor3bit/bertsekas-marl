import time

import gym
import ma_gym  # register new envs on import

from agents.constants import SpiderAndFlyEnv
from agents.random_agent import RandomAgent

if __name__ == '__main__':
    # create Spider-and-Fly game
    env = gym.make(SpiderAndFlyEnv)

    # init env
    obs_n = env.reset()

    # init agents
    n_agents = env.n_agents
    agents = [RandomAgent(env.action_space[i]) for i in range(n_agents)]

    # init stopping condition
    done_n = [False] * n_agents

    # run 100 episodes for a random agent
    while not all(done_n):
        env.render()

        # all agents act based on the observation
        act_n = []
        for agent, obs in zip(agents, obs_n):
            act_n.append(agent.act(obs))

        # update step ->
        obs_n, reward_n, done_n, info = env.step(act_n)

        time.sleep(0.5)


    env.render()
    time.sleep(2.)

    env.close()


