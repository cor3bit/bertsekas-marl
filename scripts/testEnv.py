from gym.envs.registration import register
import ma_gym.envs.predator_prey.predator_prey

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

N_EPISODES = 1
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

def visualize_image(img: np.ndarray):
    """
    Visualizes the image using matplotlib.
    
    Parameters:
    img (np.ndarray): The image to be visualized.
    """
    # Check if the image is a valid NumPy array
    if not isinstance(img, np.ndarray):
        raise ValueError("The provided image is not a valid NumPy array")

    # Display the image using matplotlib
    plt.imshow(img)
    plt.axis('off')  # Hide the axes
    plt.show()




if __name__ == '__main__':
    np.random.seed(42)

    # create Spider-and-Fly game
    env = gym.make(SpiderAndFlyEnv)
    env.seed(42)
    # env = Monitor(env, directory='../artifacts/recordings', force=True,)

    for i_episode in tqdm(range(N_EPISODES)):
        obs_n = env.reset_default()

        print(obs_n)
        imgs = env.render()
        visualize_image(imgs)

        agents = create_agents(env, AGENT_TYPE)


