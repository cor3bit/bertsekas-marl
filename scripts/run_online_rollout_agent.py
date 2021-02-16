import time
import random
from copy import copy, deepcopy

import numpy as np
import gym
import ma_gym  # register new envs on import

from agents.constants import SpiderAndFlyEnv
from agents.baseline_agent_nn import BaselineAgentNn

EPISODES_PER_SIM = 10


def run_simulation(
        env,
        agents,
        prev_actions,
        obs,
        agent_id,
        action_id
):
    total_reward = 0.

    for _ in range(EPISODES_PER_SIM):

        env_copy = copy(env)
        # env_copy = env.copy()

        act_n = []
        for agent, prev_action in zip(agents, prev_actions):
            # previously optimal actions
            if agent.id < agent_id:
                assert prev_action is not None
                act_n.append(prev_action)
            # simulated action
            elif agent.id == agent_id:
                act_n.append(action_id)
            # baseline policy actions
            else:
                best_action = agent.act(obs)
                act_n.append(best_action)

        # finish sub-interval
        obs_n, reward_n, done_n, info = env_copy.step(act_n)
        total_reward += reward_n[agent_id]

        # play episode till the end and collect the reward
        while not all(done_n):
            act_n = []

            for agent, obs in zip(agents, obs_n):
                best_action = agent.act(obs)
                act_n.append(best_action)

            obs_n, reward_n, done_n, info = env_copy.step(act_n)

            total_reward += reward_n[agent_id]

        env_copy.close()

    return total_reward / EPISODES_PER_SIM


def record_steps():
    random.seed(42)
    np.random.seed(42)
    # create Spider-and-Fly game
    env = gym.make(SpiderAndFlyEnv)
    # init env
    obs_n = env.reset_default()
    # init agents
    n_agents = env.n_agents
    n_preys = env.n_preys
    action_space_n = env.action_space[0].n
    agents = [BaselineAgentNn(i, n_agents, n_preys, env.action_space[i]) for i in range(n_agents)]
    # init stopping condition
    done_n = [False] * n_agents
    recorded_actions = []
    # run 100 episodes for a random agent
    while not all(done_n):
        # env.render()

        # all agents act based on the observation
        act_n = []

        prev_actions = [None] * n_agents

        for agent, obs in zip(agents, obs_n):

            best_action = None
            best_action_reward = -np.inf
            for action_id in range(action_space_n):
                sim_avg_reward = run_simulation(env, agents, prev_actions, obs, agent.id, action_id)
                if sim_avg_reward > best_action_reward:
                    best_action = action_id
                    best_action_reward = sim_avg_reward

            prev_actions[agent.id] = best_action

            act_n.append(best_action)

            recorded_actions.append(act_n)

        # update step ->
        obs_n, reward_n, done_n, info = env.step(act_n)

        # time.sleep(0.5)
    # env.render()
    # time.sleep(2.)
    env.close()

    return recorded_actions


if __name__ == '__main__':
    recorded_actions = record_steps()

    random.seed(42)
    np.random.seed(42)
    # create Spider-and-Fly game
    env = gym.make(SpiderAndFlyEnv)
    # init env
    obs_n = env.reset_default()
    # init agents
    n_agents = env.n_agents
    n_preys = env.n_preys
    action_space_n = env.action_space[0].n

    done_n = [False] * n_agents

    # run 100 episodes for a random agent
    for act_n in recorded_actions:
        env.render()



        # update step ->
        obs_n, reward_n, done_n, info = env.step(act_n)

        time.sleep(1)

    env.render()
    time.sleep(2.)
    env.close()
