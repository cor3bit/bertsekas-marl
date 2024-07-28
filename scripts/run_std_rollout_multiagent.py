import time
from typing import List
import cv2
import warnings
import logging

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import gym
import ma_gym  # register new envs on import
from ma_gym.wrappers.monitor import Monitor

from src.constants import SpiderAndFlyEnv, AgentType
from src.agent_std_rollout import StdRolloutMultiAgent

N_EPISODES = 10
N_SIMS_PER_MC = 50

from gym.envs.registration import register

register(
    id='PredatorPrey10x10-v4',
    entry_point='ma_gym.envs.predator_prey.predator_prey:PredatorPrey',
    max_episode_steps=1000,
    reward_threshold=1.0,
)
gym.logger.set_level(logging.ERROR)  # Set gym logger level to ERROR

warnings.filterwarnings("ignore", category=UserWarning, module="gym")  # Ignore UserWarnings from gym


def create_movie_clip(frames: list, output_file: str, fps: int = 10):
    # Assuming all frames have the same shape
    height, width, layers = frames[0].shape
    size = (width, height)
    
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    out.release()

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
    # env = Monitor(env, directory='../artifacts/recordings', force=True, )

    for i_episode in tqdm(range(N_EPISODES)):
        frames = []
        epi_steps = 0
        # init env
        # obs_n = env.reset()
        obs_n = env.reset()
        obs_n = env.reset_default()
        env.render()

        # init env variables
        m_agents = env.env.n_agents
        p_preys = env.env.n_preys
        grid_shape = env.env._grid_shape

        # init agents
        std_rollout_multiagent = StdRolloutMultiAgent(
            m_agents, p_preys, grid_shape, env.action_space[0], N_SIMS_PER_MC)

        # init stopping condition
        done_n = [False] * env.n_agents

        total_reward = .0

        # run an episode until all prey is caught
        while not all(done_n):
            act_n = std_rollout_multiagent.act_n(obs_n)

            # update step
            obs_n, reward_n, done_n, info = env.step(act_n)
            epi_steps += 1

            total_reward += np.sum(reward_n)
            # visualize_image(imgs)
            frames.append(env.render())

            # time.sleep(0.5)
            env.render()

        print(f'Episode {i_episode}: Reward is {total_reward}, with steps {epi_steps}')
        # create_movie_clip(frames, 'standardMARollout.mp4', fps=10)

    # time.sleep(2.)

    env.close()
