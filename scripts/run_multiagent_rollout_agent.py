import time

import numpy as np
import gym
import ma_gym  # register new envs on import

from agents.constants import SpiderAndFlyEnv
from agents.rollout_multiagent_agent import MultiagetRolloutAgent

if __name__ == '__main__':
    # create Spider-and-Fly game
    env = gym.make(SpiderAndFlyEnv)
    action_space_n = env.action_space[0].n

    # init env
    obs_n = env.reset()

    # init agents
    n_agents = env.n_agents
    n_preys = env.n_preys
    agents = [MultiagetRolloutAgent(i, n_agents, n_preys, env.action_space[i]) for i in range(n_agents)]

    # init stopping condition
    done_n = [False] * n_agents

    # run 100 episodes for a random agent
    while not all(done_n):
        env.render()

        # all agents act based on the observation
        act_n = []

        scaled_prev_actions = np.zeros(shape=(n_agents,), dtype=np.float32)

        for agent, obs in zip(agents, obs_n):
            best_action = agent.act(obs, scaled_prev_actions)
            act_n.append(best_action)

            scaled_prev_actions[agent.id] = (best_action + 1) / (action_space_n + 1)

        # update step ->
        obs_n, reward_n, done_n, info = env.step(act_n)

        time.sleep(0.5)

    env.render()
    time.sleep(2.)

    env.close()
