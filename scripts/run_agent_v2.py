from gym.envs.registration import register
import ma_gym.envs.predator_prey.predator_prey
import time

register(
    id='PredatorPrey10x10-v4',
    entry_point='ma_gym.envs.predator_prey.predator_prey:PredatorPrey',
    max_episode_steps=1000,
    reward_threshold=1.0,
)

import gym
import numpy as np
from typing import List


from src.constants import SpiderAndFlyEnv, AgentType, QnetType
from src.agent import Agent
from src.agent_random import RandomAgent
from src.agent_rule_based import RuleBasedAgent
from src.agent_seq_rollout import SeqRolloutAgent
from src.agent_qnet_based import QnetBasedAgent

N_EPISODES = 10
AGENT_TYPE = AgentType.QNET_BASED
QNET_TYPE = QnetType.BASELINE
BASIS_AGENT_TYPE = AgentType.RULE_BASED
N_SIMS = 10



from tqdm import tqdm

def create_agents(
        env: gym.Env,
        agent_type: str,
) -> List[Agent]:
    # init env variables
    m_agents = env.n_agents
    p_preys = env.n_preys
    grid_shape = env.grid_shape

    if agent_type == AgentType.RANDOM:
        return [RandomAgent(
            env.action_space[agent_i],
        ) for agent_i in range(m_agents)]
    elif agent_type == AgentType.RULE_BASED:
        return [RuleBasedAgent(
            agent_i, m_agents, p_preys, grid_shape, env.action_space[agent_i],
        ) for agent_i in range(m_agents)]
    elif agent_type == AgentType.QNET_BASED:
        return [QnetBasedAgent(
            agent_i, m_agents, p_preys, grid_shape, env.action_space[agent_i], QNET_TYPE,
        ) for agent_i in range(m_agents)]
    elif agent_type == AgentType.SEQ_MA_ROLLOUT:
        return [SeqRolloutAgent(
            agent_i, m_agents, p_preys, grid_shape, env.action_space[i],
            n_sim_per_step=N_SIMS, basis_agent_type=BASIS_AGENT_TYPE, qnet_type=QNET_TYPE,
        ) for agent_i in range(m_agents)]
    else:
        raise ValueError(f'Unrecognized agent type: {agent_type}.')

import matplotlib.pyplot as plt
import numpy as np


def visualize_image(img: np.ndarray, pause_time: float = 0.5):

    if not isinstance(img, np.ndarray):
        raise ValueError("The provided image is not a valid NumPy array")

    plt.imshow(img)
    plt.axis('off') 
    plt.show(block=False) 
    plt.pause(pause_time)  
    plt.close()  




if __name__ == '__main__':
    np.random.seed(42)

    # create Spider-and-Fly game
    env = gym.make(SpiderAndFlyEnv)
    env.seed(42)
    # env = Monitor(env, directory='../artifacts/recordings', force=True,)

    for i_episode in tqdm(range(N_EPISODES)):
        #bs_n = env.reset_default()
        obs_n = env.reset()

        print(obs_n)
        imgs = env.render()
        visualize_image(imgs)

        agents = create_agents(env, AGENT_TYPE)
        done_n = [False] * env.n_agents
        total_reward = 0.

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
            print(total_reward)

            imgs = env.render()
            visualize_image(imgs)

            time.sleep(0.1)


