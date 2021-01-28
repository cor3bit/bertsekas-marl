import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import ma_gym  # register new envs on import

from agents.constants import SpiderAndFlyEnv, BaselineModelPath
from agents.baseline_agent import BaselineAgent
from agents.qnetwork import QNetwork

N_SAMPLES = 100


# BATCH_SIZE = 16
# data_train = DataLoader(dataset = data, batch_size = BATCH_SIZE, shuffle =False)
# criterion = nn.MSELoss()


def generate_samples():
    print('Started sample generation.')

    samples = []

    # create Spider-and-Fly game
    env = gym.make(SpiderAndFlyEnv)

    while len(samples) < N_SAMPLES:
        # init env
        obs_n = env.reset()

        # init agents
        n_agents = env.n_agents
        n_preys = env.n_preys
        agents = [BaselineAgent(i, n_agents, n_preys, env.action_space[i]) for i in range(n_agents)]

        # init stopping condition
        done_n = [False] * n_agents

        # run 100 episodes for a random agent
        while not all(done_n):
            # for each agent calculates Manhattan Distance to each prey for each
            # possible action
            # O(n*m*q)
            distances = env.get_distances()

            # transform into samples
            obs_first = np.array(obs_n[0]).flatten()  # same for all agent
            for i, a_dist in enumerate(distances):
                agent_ohe = np.zeros(shape=(n_agents,), dtype=np.float)
                agent_ohe[i] = 1.

                min_prey = a_dist.min(axis=1)
                size_action_space = min_prey.size
                for j, d in enumerate(min_prey):
                    if d != np.inf:
                        action_ohe = np.zeros(shape=(size_action_space,), dtype=np.float)
                        action_ohe[j] = 1.

                        x = np.concatenate((obs_first, agent_ohe, action_ohe))
                        y = -d

                        samples.append((x, y))

            # all agents act based on the observation
            act_n = []
            for agent, obs, action_distances in zip(agents, obs_n, distances):
                max_action = agent.act(action_distances)
                act_n.append(max_action)

            # update step ->
            obs_n, reward_n, done_n, info = env.step(act_n)

        env.close()

    print('Finished sample generation.')

    return samples[:N_SAMPLES]


def train_qnetwork():
    print('Started Training.')

    net = QNetwork()

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training.')

    return net


if __name__ == '__main__':
    samples = generate_samples()

    net = train_qnetwork()

    torch.save(net.state_dict(), BaselineModelPath)
