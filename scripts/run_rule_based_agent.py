import time

import gym
import ma_gym  # register new envs on import

from agents.constants import SpiderAndFlyEnv
from agents.rule_based_agent import RuleBasedAgent

N_EPISODES = 5

if __name__ == '__main__':
    # create Spider-and-Fly game
    env = gym.make(SpiderAndFlyEnv)

    for _ in range(N_EPISODES):
        # init env
        time.sleep(2.)
        # obs_n = env.reset()
        obs_n = env.reset_default()
        env.render()

        # init agents
        m_agents = env.n_agents
        p_preys = env.n_preys
        agents = [RuleBasedAgent(i, m_agents, p_preys, env._grid_shape, env.action_space[i])
                  for i in range(m_agents)]

        # init stopping condition
        done_n = [False] * m_agents

        # run an episode until all prey is caught
        while not all(done_n):

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
