import numpy as np
import gym
from gym.envs.registration import register
from src.agent_rule_based import RuleBasedAgent

register(
    id='PredatorPrey10x10-v4',
    entry_point='ma_gym.envs.predator_prey.predator_prey:PredatorPrey',
    max_episode_steps=1000,
    reward_threshold=1.0,
)



from src.constants import SpiderAndFlyEnv, BaselineModelPath_10x10_4v2


m_agents = 4
p_preys= 2 
grid_shape=(10, 10)

env = gym.make(SpiderAndFlyEnv)
obs_n = env.reset()

print("Printing Observations")
print(obs_n)

print("Printing action space")
acts = env.action_space


## Same observation for all agents -> duplicate lists for all agents
# first 4 are agent positions
# next 2 are preys positions
# final are if preys are alive or not.
# [   [0.4444444444444444, 0.0, 
#      0.3333333333333333, 0.4444444444444444, 
#      0.8888888888888888, 0.5555555555555556, 
#      0.3333333333333333, 0.3333333333333333, 
#      0.5555555555555556, 0.1111111111111111, 
#      0.8888888888888888, 0.3333333333333333, 
#      1.0, 1.0], 

obs_first = np.array(obs_n[0], dtype=np.float32).flatten()

agents = [RuleBasedAgent(
            agent_i, m_agents, p_preys, grid_shape, env.action_space[agent_i],
        ) for agent_i in range(m_agents)]

done_n = [False] * m_agents
prev_actions = {}
act_n = []
for i, (agent, obs) in enumerate(zip(agents, obs_n)):
    action_id = agent.act(obs, prev_actions=prev_actions)
    print("action_id",action_id)

    prev_actions[i] = action_id
    act_n.append(action_id)

print(act_n)