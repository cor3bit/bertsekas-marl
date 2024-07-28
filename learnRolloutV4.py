from gym.envs.registration import register
import ma_gym.envs.predator_prey.predator_prey
import time
import cv2

register(
    id='PredatorPrey10x10-v4',
    entry_point='ma_gym.envs.predator_prey.predator_prey:PredatorPrey',
    max_episode_steps=1000,
    reward_threshold=1.0,
)

import gym
import numpy as np
from typing import List
from random import sample, random
from dataclasses import dataclass
from typing import Any


from src.constants import SpiderAndFlyEnv, AgentType, QnetType
from src.agent import Agent
from src.agent_random import RandomAgent
from src.agent_rule_based import RuleBasedAgent
from src.agent_seq_rollout import SeqRolloutAgent
from src.agent_qnet_based import QnetBasedAgent
from src.agent_std_rollout import StdRolloutMultiAgent

import warnings

# Suppress the specific gym warning
warnings.filterwarnings("ignore", category=UserWarning)
import wandb

from tqdm import tqdm
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



AGENT_TYPE = AgentType.SEQ_MA_ROLLOUT
N_EPISODES = 300000
#GENT_TYPE = AgentType.SEQ_MA_ROLLOUT
QNET_TYPE = QnetType.REPEATED
BASIS_AGENT_TYPE = AgentType.RULE_BASED
N_SIMS = 10
SEED = 42
N_SIMS_MC = 50


def create_agents(
        env: gym.Env,
        agent_type: str,
) -> List[Agent]:
    # init env variables
    m_agents = env.n_agents
    p_preys = env.n_preys
    #grid_shape = env.grid_shape
    grid_shape = env.grid_shape

    return [SeqRolloutAgent(
        agent_i, m_agents, p_preys, grid_shape, env.action_space[agent_i],
        n_sim_per_step=N_SIMS, basis_agent_type=BASIS_AGENT_TYPE, qnet_type=QNET_TYPE,
    ) for agent_i in range(m_agents)]
   

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
    def __init__(self,buffer_size = 100000):
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


def train_step(model,state_transitions,tgt, gamma=0.99):
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

        # one_hot_actions = F.one_hot(torch.LongTensor(actions),num_actions)
        one_hot_actions = torch.LongTensor(actions)
        
        
        # ignoring the discount factor for now
        # check deep rl tutorial david silver for this refrence 
        loss = ((rewards +  mask[:,0]*qval_next*gamma - torch.sum(qvals*one_hot_actions,-1))**2).mean()
        loss.backward()
        
        model.opt.step()

        return loss



if __name__ == '__main__':
    
    env = gym.make(SpiderAndFlyEnv)
    obs_n = env.reset()
    m_agents = env.n_agents
    p_preys = env.n_preys
    action_space = env.action_space[0]

    wandb.init(project="Training_SecurityAndSurveillance",name="Sequential Rollout RB")
    m = Model(tuple(torch.Tensor(obs_n[0]).shape),np.int64(env.action_space[0].n*m_agents)) # model that we train
    tgt = Model(tuple(torch.Tensor(obs_n[0]).shape),np.int64(env.action_space[0].n*m_agents)) # fixed model

    rb = ReplayBuffer()

    update_tgt_model(m,tgt)
    

    eps_decay = 0.99999995

    
    env_steps_before_train = 1000
    min_rb_size = 1000
    sample_size = 100
    tgt_model_update = 10

    epochs_since_tgt = 0
    steps_num = 0
    steps_since_train = 0
    rolling_reward = 0
    episode_rewards = []
    episode_steps = []

    tq = tqdm()
    for epi in range (N_EPISODES):
        tq.update(1)
        frames = []
        startTime = time.time()
        obs_n = env.reset()
        agents = create_agents(env, AGENT_TYPE)
        
        done_n = [False] * env.n_agents
        rolling_reward = 0
        epi_steps = 0
        while not all(done_n):
            prev_actions = {}
            
            eps =eps_decay**(steps_num)

            if random() < eps:
                act_n = []
                for i, (agent, obs) in enumerate(zip(agents, obs_n)):
                    # each agent acts based on the same observation
                    action_id = agent.act(obs, prev_actions=prev_actions)
                    prev_actions[i] = action_id

                    act_n.append(action_id)
            else:
                act_n = select_actions_from_output(m(torch.Tensor(obs_n[0])),m_agents,action_space.n)


            # find one hot encoding of action
            actions_ohe = np.zeros(shape=(m_agents * action_space.n,), dtype=np.int8)
            for agntID in range(m_agents):
                actionIndex = act_n[agntID] + int(agntID * action_space.n)
                actions_ohe[actionIndex] = 1



            obs_n_Next, reward_n, done_n, info = env.step(act_n)
            steps_num += 1
            epi_steps += 1
            steps_since_train += 1

            done = all(done_n)


            rb.insert(Sarsd(obs_n[0], actions_ohe,np.mean(reward_n),obs_n_Next[0],done))
            obs_n = obs_n_Next

            rolling_reward += np.mean(reward_n)
            frames.append(env.render())

        endTime = time.time()
        episode_rewards.append(rolling_reward)
        episode_steps.append(epi_steps)

        # print(f'Episode {epi}: Reward is {rolling_reward}, with steps {epi_steps} exeTime{endTime-startTime}')
        
        if rb.idx > min_rb_size and steps_since_train > env_steps_before_train:
            loss=  train_step(m,rb.sample(sample_size),tgt)
            wandb.log({'loss':loss.detach().item(), 'eps':epi, 'avg_reward' : np.mean(episode_rewards), 'avg_steps' : np.mean(episode_steps)},step=steps_num) 
            
            episode_rewards = []
            episode_steps = []

            steps_since_train = 0
            
            epochs_since_tgt += 1

            print(loss,epochs_since_tgt,tgt_model_update)


            if epochs_since_tgt > tgt_model_update:
                print("Updating the Target Model")
                update_tgt_model(m,tgt)
                epochs_since_tgt = 0
                torch.save(tgt.state_dict(),f"models/{steps_num}.pth")


        # if (epi+1) % 100 ==0:
        #     print("Updating the Target Model")
        #     update_tgt_model(m,tgt)
        #     torch.save(tgt.state_dict(),f"models/{steps_num}.pth")

        #     print("Checkpoint passed")
        #     import ipdb; ipdb.set_trace()
            
            
            # axes are (time, channel, height, width)
            # create_movie_clip(frames, f"ManhattanRuleBased_2_agents_{epi+1}.mp4", fps=10)
            # wandb.log({"video": wandb.Video(np.stack(frames,0).transpose(0,3,1,2), fps=20,format="mp4")})



    wandb.finish()
    env.close()