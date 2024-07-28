from time import perf_counter
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import ma_gym  # register new envs on import

from src.constants import SpiderAndFlyEnv, RepeatedRolloutModelPath_10x10_4v3, RepeatedRolloutModelPath_10x10_4v2, AgentType, \
    QnetType
from src.qnetwork_coordinated import QNetworkCoordinated
from src.agent_seq_rollout import SeqRolloutAgent

import wandb
import warnings
from random import sample, random
from dataclasses import dataclass
from typing import Any

# Suppress the specific gym warning
warnings.filterwarnings("ignore", category=UserWarning)


EED = 42

M_AGENTS = 4
P_PREY = 2

N_SAMPLES = 50
BATCH_SIZE = 1024
EPOCHS = 500
N_SIMS_MC = 50
FROM_SCRATCH = False
INPUT_QNET_NAME = RepeatedRolloutModelPath_10x10_4v3
OUTPUT_QNET_NAME = RepeatedRolloutModelPath_10x10_4v3
BASIS_POLICY_AGENT = AgentType.QNET_BASED
QNET_TYPE = QnetType.BASELINE

'''
Attempting to create a better signaling policy by
approximating  the sequential rollout policy
'''

@dataclass
class Sarsd:
    state: Any
    action : Any
    reward : int
    next_state : Any
    done : bool


# Improve this with python deque
# also can be a database?
# deque can be dropped and made faster
class ReplayBuffer:
    def __init__(self,buffer_size = 100):
        self.buffer_size = buffer_size
        self.buffer = [None]*buffer_size # fixed size array
        self.idx = 0

    def insert(self,sars):
        self.buffer[self.idx % self.buffer_size] = sars
        self.idx += 1

    def sample(self, num_samples):
        
        assert num_samples < min(self.idx,self.buffer_size)

        if self.idx < self.buffer_size:
        # until we reach the buffer size we cant  sample
        # from the entire array but sample upto idx only
            return sample(self.buffer[:self.idx],num_samples)
        return sample(self.buffer,num_samples)
            

        # return sample(self.buffer, num_samples)




 
class Model(nn.Module):
    def __init__(self,obs_shape,num_actions):
        super(Model,self).__init__()
        assert len(obs_shape) == 1 # This only works for flat observations
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_shape[0],256),
            torch.nn.ReLU(),
            torch.nn.Linear(256,num_actions),
            # No activations after this because we 
            # need to represent a real value for the rewards
            # if not we wont be able to represent negative rewards
        )
        self.opt = optim.Adam(self.net.parameters(),lr = 0.0001)

    
    def forward(self,x):
        return self.net(x)
    
def update_tgt_model(m,tgt):
    '''
    THis is to copy the weights from one to another
    '''
    tgt.load_state_dict(m.state_dict())


def select_actions_from_output(output, n_agents, n_actions_per_agent):
    actions = []
    for i in range(n_agents):
        agent_action_values = output[i * n_actions_per_agent: (i + 1) * n_actions_per_agent]
        best_action = torch.argmax(agent_action_values).item()
        actions.append(best_action)
    return actions

def train_step(model,state_transitions,tgt,num_actions, gamma=0.99):
        # import ipdb; ipdb.set_trace()
        # need to create the state vector
        # that is a stacked

        cur_states = torch.stack([torch.Tensor(s.state) for s in state_transitions])
        rewards = torch.stack([torch.Tensor([s.reward]) for s in state_transitions])
        
        # we need to make the future rewards to zero
        # if the episode is done. So 1 if weren't done
        # 0 if were done
        mask = torch.stack([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in state_transitions])

        next_states = torch.stack([torch.Tensor(s.next_state) for s in state_transitions])
        actions = [s.action for s in state_transitions]


        with torch.no_grad():
            # we dont do backprop on target model
            qval_next = tgt(next_states).max(-1)[0] # max of soemthig shaped (N, num_actions)

        model.opt.zero_grad()
       
        # we need to pick only the q values for the actions that we chose
        qvals = model(cur_states) # (N, num_actions)

        one_hot_actions = F.one_hot(torch.LongTensor(actions),num_actions)
        
        
        # ignoring the discount factor for now

        # check deep rl tutorial david silver for this refrence 
        loss = ((rewards +  mask[:,0]*qval_next*gamma - torch.sum(qvals*one_hot_actions,-1))**2).mean()
        loss.backward()
        
        model.opt.step()

        return loss



def main():
    min_rb_size = 20
    sample_size = 10
    
    eps_decay = 0.9999

    env_steps_before_train = 5
    tgt_model_update = 5


    env = gym.make(SpiderAndFlyEnv)
    obs_n = env.reset()
    m_agents = env.n_agents
    p_preys = env.n_preys
    grid_shape = env._grid_shape
    action_space = env.action_space[0]


    


    m = Model(tuple(torch.Tensor(obs_n[0]).shape),np.int64(env.action_space[0].n*m_agents)) # model that we train
    tgt = Model(tuple(torch.Tensor(obs_n[0]).shape),np.int64(env.action_space[0].n*m_agents)) # fixed model
    update_tgt_model(m,tgt)

    
    

    agents = [SeqRolloutAgent(
                i, m_agents, p_preys, grid_shape, env.action_space[i],
                n_sim_per_step=N_SIMS_MC,
                basis_agent_type=BASIS_POLICY_AGENT,
                qnet_type=QNET_TYPE,
            ) for i in range(m_agents)]

    rb = ReplayBuffer()

    steps_since_train = 0
    epochs_since_tgt = 0
    steps_num = -1 * min_rb_size

    episode_rewards = []
    rolling_reward = 0
    tq = tqdm()
    try:
        while True:
            tq.update(1)
            eps =eps_decay**(steps_num)

            act_n = []

            if random() < eps:
                
                # use sequential actions
                prev_actions = {}
                
                for i, (agent, obs) in enumerate(zip(agents, obs_n)):
                    best_action, action_q_values = agent.act_with_info(
                        obs, prev_actions=prev_actions)
                    
                    prev_actions[i] = best_action
                    act_n.append(best_action)

            else:
                # use the agent to get the action
                act_n = select_actions_from_output(m(torch.Tensor(obs_n[0])),m_agents,action_space.n)


            # do one hot encoding for agent actions
            actions_ohe = np.zeros(shape=(m_agents * action_space.n,), dtype=np.float32)
            for agntID in range(m_agents):
                actionIndex = act_n[agntID] + int(agntID * action_space.n)
                actions_ohe[actionIndex] = 1


            obs_n_Next, reward_n, done_n, info = env.step(act_n)
            done = all(done_n)

            rolling_reward += np.sum(reward_n)

            rb.insert(Sarsd(obs_n[0], actions_ohe,np.sum(reward_n),obs_n_Next[0],done))
            obs_n = obs_n_Next

            # end of an episode
            if done:
                episode_rewards.append(rolling_reward)
                rolling_reward = 0

                obs_n_Next = env.reset()

            steps_since_train += 1
            steps_num += 1

            if rb.idx > min_rb_size and steps_since_train > env_steps_before_train:
                loss=  train_step(m,rb.sample(sample_size),tgt,env.action_space[0].n*m_agents)

                print(loss)
                

                episode_rewards = []
                epochs_since_tgt += 1
                if epochs_since_tgt > tgt_model_update:
                    print("Updating the Target Model")
                    update_tgt_model(m,tgt)
                    epochs_since_tgt = 0
                    torch.save(tgt.state_dict(),f"models/{steps_num}.pth")

                steps_since_train = 0

    except KeyboardInterrupt:
        pass

    env.clos()






if __name__ == '__main__':
    main()