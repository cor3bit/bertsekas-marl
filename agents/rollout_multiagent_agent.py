from joblib import dump, load


class RolloutAgent:
    def __init__(self, method):
        pass

    def act(self):
        pass

    def train(self, env, n_episodes, save_weights=True):
        pass

    def run(self, env, n_episodes=10, render=False, weight_path=None):
        pass


def rnd_predict(env, n_episodes, render, weight_path=None):
    for i_episode in range(n_episodes):

        observation = env.reset()
        done = False
        while not done:
            if render:
                env.render()

            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
