import gym
import ma_gym  # registers new envs on import
import time

env = gym.make('PredatorPrey10x10-v4')

done_n = [False for _ in range(env.n_agents)]
ep_reward = 0

obs_n = env.reset()

# while not all(done_n):
for i in range(100):
    env.render()
    obs_n, reward_n, done_n, info = env.step(env.action_space.sample())
    ep_reward += sum(reward_n)

    time.sleep(0.1)

env.close()
