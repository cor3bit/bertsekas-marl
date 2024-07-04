import time
from typing import List, Iterable

from tqdm import tqdm
import numpy as np
import gym
import ma_gym  # register new envs on import
import torch
import torch.nn as nn
import torch.optim as optim

from src.constants import SpiderAndFlyEnv, RolloutModelPath_10x10_4v2
from src.qnetwork import QNetwork
from src.qnetwork_coordinated import QNetworkCoordinated
from src.agent_rule_based import RuleBasedAgent
from src.agent_qnet_based import QnetBasedAgent

N_EPISODES = 20
N_SIMS_PER_STEP = 10
BATCH_SIZE = 512
EPOCHS = 10

import warnings
import logging
import cv2
import matplotlib.pyplot as plt


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



def simulate(
        initial_obs: np.array,
        initial_step: np.array,
        m_agents: int,
        qnet,
        fake_env,
        action_space,
        epsilon,
) -> float:
    avg_total_reward = .0

    # create env
    env = gym.make(SpiderAndFlyEnv)
   

    # run N simulations
    for _ in range(N_SIMS_PER_STEP):
        obs_n = env.reset()
        obs_n = env.reset_from(initial_obs)

        # 1 step
        obs_n, reward_n, done_n, info = env.step(initial_step)
        avg_total_reward += np.sum(reward_n)

        # run an episode until all prey is caught
        while not all(done_n):

            # all agents act based on the observation
            act_n = []

            prev_actions = {}

            for agent_id, obs in enumerate(obs_n):
                # obs_after_coordination = update_obs(fake_env, obs, prev_actions)

                action_taken = epsilon_greedy_step(
                    obs, m_agents, agent_id, qnet, action_space, prev_actions, epsilon)

                prev_actions[agent_id] = action_taken
                act_n.append(action_taken)

            # update step
            obs_n, reward_n, done_n, info = env.step(act_n)

            avg_total_reward += np.sum(reward_n)

    env.close()

    avg_total_reward /= m_agents
    avg_total_reward /= N_SIMS_PER_STEP

    return avg_total_reward


def convert_to_x(obs, m_agents, agent_id, action_space, prev_actions):
    # state
    obs_first = np.array(obs, dtype=np.float32).flatten()

    # agent ohe
    agent_ohe = np.zeros(shape=(m_agents,), dtype=np.float32)
    agent_ohe[agent_id] = 1.

    # prev actions
    prev_actions_ohe = np.zeros(shape=(m_agents * action_space.n,), dtype=np.float32)
    for agent_i, action_i in prev_actions.items():
        ohe_action_index = int(agent_i * action_space.n) + action_i
        prev_actions_ohe[ohe_action_index] = 1.

    # combine all
    x = np.concatenate((obs_first, agent_ohe, prev_actions_ohe))

    return x


def epsilon_greedy_step(
        obs,
        m_agents,
        agent_id,
        qnet,
        action_space,
        prev_actions,
        epsilon,
) -> int:
    p = np.random.random()
    if p < epsilon:
        # random action -> exploration
        return action_space.sample()
    else:
        qnet.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # argmax -> exploitation
        x = convert_to_x(obs, m_agents, agent_id, action_space, prev_actions)
        x = np.reshape(x, newshape=(1, -1))
        # v = torch.from_numpy(x)
        v = torch.tensor(x, device=device)
        qs = qnet(v)
        return np.argmax(qs.data.cpu().numpy())


def epsilon_greedy_step_from_array(
        qvalues,
        action_space,
        epsilon,
):
    p = np.random.random()
    if p < epsilon:
        # random action -> exploration
        return action_space.sample()
    else:
        # argmax -> exploitation
        return np.argmax(qvalues)


def update_obs(env, obs, prev_actions):
    if prev_actions:
        obs_new = env.reset_from(obs)
        for agent_id, action in prev_actions.items():
            obs_new = env.apply_move(agent_id, action)

        return obs_new[0]
    else:
        return obs


def get_epsilon(episode_i: int, episodes_n: int) -> float:
    # TODO

    # First 10%
    if episode_i < episodes_n * 0.1:
        return 0.2

    #

    return 0.05


def train_qnet(qnet, samples):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    qnet.train()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(qnet.parameters(), lr=0.01)
    data_loader = torch.utils.data.DataLoader(samples,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True)

    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        running_loss = .0
        n_batches = 0

        for data in data_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = qnet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # logging
            running_loss += loss.item()
            n_batches += 1

        print(f'[{epoch}] {running_loss / n_batches:.3f}.')

    return qnet


def learn_policy():
    frames = []
    np.random.seed(42)

    # create Spiders-and-Flies game
    env = gym.make(SpiderAndFlyEnv)
    env.seed(42)

    # used only for specific env methods
    fake_env = gym.make(SpiderAndFlyEnv)
    fake_env.seed(1)

    # init env variables
    m_agents = env.n_agents
    p_preys = env.n_preys
    #grid_shape = env._grid_shape
    action_space = env.action_space[0]

    # rollout net
    qnet = QNetworkCoordinated(m_agents, p_preys, action_space.n)
    # qnet.load_state_dict(torch.load(RolloutModelPath_10x10_4v2))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    qnet.to(device)

    for episode_i in tqdm(range(N_EPISODES)):
        eps = get_epsilon(episode_i, N_EPISODES)

        # init env
        obs_n = env.reset()

        # init stopping condition
        done_n = [False] * env.n_agents

        # run an episode until all prey is caught
        while not all(done_n):
            prev_actions = {}
            act_n = []

            samples = []

            for agent_id, obs in enumerate(obs_n):
                n_actions = action_space.n
                # create the same action space with infinity values filled.
                q_values = np.full((n_actions,), fill_value=-np.inf, dtype=np.float32)

                new_actions = {}

                for action_id in range(n_actions):
                    # 1st step - optimal actions from previous agents,
                    # simulated step from current agent,
                    # greedy (baseline) from undecided agents
                    initial_step = np.empty((m_agents,), dtype=np.int8)

                    for i in range(m_agents):
                        if i in prev_actions:
                            initial_step[i] = prev_actions[i]
                        elif agent_id == i:
                            initial_step[i] = action_id

                            new_actions[i] = action_id
                        else:
                            # update obs with info about prev steps
                            # obs_after_coordination = update_obs(fake_env, obs, {**prev_actions, **new_actions})

                            best_action = epsilon_greedy_step(
                                obs,
                                m_agents,
                                i,
                                qnet,
                                action_space,
                                {**prev_actions, **new_actions},
                                epsilon=eps,
                            )

                            initial_step[i] = best_action

                            new_actions[i] = best_action

                    # run N simulations
                    avg_total_reward = simulate(obs, initial_step, m_agents, qnet, fake_env, action_space, eps)

                    q_values[action_id] = avg_total_reward

                # adds sample to the dataset
                # agent_ohe = np.zeros(shape=(m_agents,), dtype=np.float32)
                # agent_ohe[agent_id] = 1.
                # # obs_after_coordination = np.array(update_obs(fake_env, obs, prev_actions), dtype=np.float32)
                # prev_actions_ohe = np.zeros(shape=(m_agents * action_space.n,), dtype=np.float32)
                #
                # x = np.concatenate((obs, agent_ohe, prev_actions_ohe))

                x = convert_to_x(obs, m_agents, agent_id, action_space, prev_actions)
                samples.append((x, q_values))

                # TODO sanity check
                # print(f'Qnet: {q_values}')
                # print(f'MC: {}')

                # current policy
                action_taken = epsilon_greedy_step_from_array(q_values, action_space, epsilon=eps)

                prev_actions[agent_id] = action_taken
                act_n.append(action_taken)

            # update step
            obs_n, reward_n, done_n, info = env.step(act_n)
            imgs = env.render()
            visualize_image(imgs)
            frames.append(imgs)

            # update rollout policy with samples
            qnet = train_qnet(qnet, samples)

    env.close()

    # save updated qnet
    torch.save(qnet.state_dict(), RolloutModelPath_10x10_4v2)

    return frames
 

# ------------- Runner -------------

if __name__ == '__main__':
    frames = learn_policy()
    create_movie_clip(frames, 'onlinePolicyIteration.mp4', fps=10) 
