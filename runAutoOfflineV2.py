from time import perf_counter
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple

from tqdm import tqdm
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import ma_gym  # register new envs on import

from src.constants import SpiderAndFlyEnv, RepeatedRolloutModelPath_10x10_4v3, AgentType, \
    QnetType
from src.qnetwork_coordinated import QNetworkCoordinated
from src.agent_seq_rollout import SeqRolloutAgent
from src.agent_rule_based import RuleBasedAgent

import time
import wandb
from src.agent import Agent


import warnings

# Suppress the specific gym warning
warnings.filterwarnings("ignore", category=UserWarning)

from learnRolloutV4 import Model, select_actions_from_output

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = gym.make(SpiderAndFlyEnv)
    obs_n = env.reset()
    m_agents = env.n_agents
    action_space = env.action_space[0]

    
    model_path = "models/221640.pth" 
    trained_model = Model(tuple(torch.Tensor(obs_n[0]).shape), np.int64(env.action_space[0].n * m_agents))
    trained_model.load_state_dict(torch.load(model_path))
    trained_model.eval()  

    # Testing the trained model
    epi = 0
    epiFlag = False
    while epiFlag == False:
        obs_n = env.reset()
        done_n = [False] * env.n_agents
        rolling_reward = 0
        epi_steps = 0
        while not all(done_n):
            with torch.no_grad():
                act_n = select_actions_from_output(trained_model(torch.Tensor(obs_n[0])), m_agents, action_space.n)
            
            obs_n, reward_n, done_n, info = env.step(act_n)
            epi_steps += 1
            rolling_reward += np.mean(reward_n)
            
            # Render the environment (optional)
            env.render()
        
        if rolling_reward > -12:
            print(f'Episode {epi + 1}: Reward is {rolling_reward} with step count {epi_steps}')
            epi += 1
        if epi>30:
            epiFlag = True

    env.close()
        
