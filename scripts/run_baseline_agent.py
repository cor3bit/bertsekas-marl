import time

import gym
import ma_gym  # register new envs on import

from agents.constants import SpiderAndFlyEnv
from agents.baseline_agent import BaselineAgent

if __name__ == '__main__':
    # create Spider-and-Fly game
    env = gym.make(SpiderAndFlyEnv)

    # init env
    obs_n = env.reset()

    # init agents
    n_agents = env.n_agents
    n_preys = env.n_preys
    agents = [BaselineAgent(i, n_agents, n_preys, env.action_space[i]) for i in range(n_agents)]

    # init stopping condition
    done_n = [False] * n_agents

    # run 100 episodes for a random agent
    while not all(done_n):
        env.render()

        # for each agent calculates Manhattan Distance to each prey for each
        # possible action
        # O(n*m*q)
        distances = env.get_distances()

        # all agents act based on the observation
        act_n = []
        for agent, obs, action_distances in zip(agents, obs_n, distances):
            act_n.append(agent.act(action_distances))

        # update step ->
        obs_n, reward_n, done_n, info = env.step(act_n)

        time.sleep(0.2)

    env.render()
    time.sleep(2.)

    env.close()
