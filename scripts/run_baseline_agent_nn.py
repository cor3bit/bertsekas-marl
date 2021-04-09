import time

import gym
import ma_gym  # register new envs on import
from ma_gym.wrappers import Monitor

from src.constants import SpiderAndFlyEnv
from src.baseline_agent_nn import BaselineAgentNn

N_EPISODES = 5

if __name__ == '__main__':
    # gym.logger.set_level(gym.logger.DEBUG)

    # create Spider-and-Fly game
    env = gym.make(SpiderAndFlyEnv)
    # env = Monitor(env, directory='../artifacts/recordings', force=True,)

    for _ in range(N_EPISODES):
        # init env
        # TODO DEMO!!!
        # obs_n = env.reset_default()
        obs_n = env.reset()

        # init agents
        n_agents = env.n_agents
        n_preys = env.n_preys
        agents = [BaselineAgentNn(i, n_agents, n_preys, env.action_space[i]) for i in range(n_agents)]

        # init stopping condition
        done_n = [False] * n_agents

        env.render()

        # run 100 episodes for a random agent
        while not all(done_n):

            # all agents act based on the observation
            act_n = []
            for agent, obs in zip(agents, obs_n):
                best_action = agent.act(obs)
                act_n.append(best_action)

            # update step ->
            obs_n, reward_n, done_n, info = env.step(act_n)

            time.sleep(0.5)
            env.render()

    # env.render()
    time.sleep(2.)

    env.close()
