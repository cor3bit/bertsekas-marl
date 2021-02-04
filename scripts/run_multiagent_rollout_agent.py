import gym

from agents.constants import SpiderAndFlyEnv
from agents.rollout_multiagent_agent import RolloutAgent


def train_robot():
    # initializes env
    env = gym.make(SpiderAndFlyEnv)

    # initializes an RL agent with pre-trained weights
    agent = RolloutAgent()
    agent.train(env, n_episodes=100000, save_weights=True)

    # post-processing
    env.close()


def test_robot():
    # initializes env
    env = gym.make(SpiderAndFlyEnv)

    # initializes an RL agent with pre-trained weights
    agent = RolloutAgent()
    agent.run(env, n_episodes=5, render=True)

    # post-processing
    env.close()


# ----------------- Script -----------------


if __name__ == '__main__':
    # train
    train_robot()

    # test
    test_robot()
