from time import perf_counter

import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import ma_gym  # register new envs on import

from src.constants import SpiderAndFlyEnv, BaselineModelPath_10x10_4v2
from src.baseline_agent import BaselineAgent
from src.qnetwork import QNetwork

N_AGENTS = 4
N_PREY = 2

N_SAMPLES = 2_000_000

BATCH_SIZE = 256

EPOCHS = 20

SEED = 42


# data_train = DataLoader(dataset = data, batch_size = BATCH_SIZE, shuffle =False)
# criterion = nn.MSELoss()


def generate_samples(n_samples, seed):
    print('Started sample generation.')

    samples = []

    # create Spider-and-Fly game
    env = gym.make(SpiderAndFlyEnv)

    env.seed(seed)

    while len(samples) < n_samples:
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
            obs_first = np.array(obs_n[0], dtype=np.float32).flatten()  # same for all agent
            for i, a_dist in enumerate(distances):
                agent_ohe = np.zeros(shape=(n_agents,), dtype=np.float32)
                agent_ohe[i] = 1.

                min_prey = a_dist.min(axis=1).astype(np.float32)

                x = np.concatenate((obs_first, agent_ohe))
                y = -min_prey

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

    return samples[:n_samples]


def train_qnetwork(samples):
    print('Started Training.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Found device: {device}.')

    net = QNetwork(N_AGENTS, N_PREY)

    net.to(device)

    net.train()  # check

    criterion = nn.MSELoss()

    optimizer = optim.Adam(net.parameters(), lr=0.01)

    data_loader = torch.utils.data.DataLoader(samples,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True)

    for epoch in range(EPOCHS):  # loop over the dataset multiple times

        running_loss = 0.0
        n_batches = 0

        for i, data in enumerate(data_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            # print statistics
            running_loss += loss.item()
            n_batches += 1

            # if i % 200 == 199:  # print every 2000 mini-batches
            #     print('[%d, %5d] loss: %.3f' %
            #           (epoch + 1, i + 1, running_loss / 200))
            #
            #     running_loss = 0.0

        if epoch % 10 == 0:
            torch.save(net.state_dict(), BaselineModelPath_10x10_4v2)

        print(f'[{epoch}] {running_loss / n_batches:.3f}.')

    print('Finished Training.')

    return net


def test_qnetwork(net, samples):
    net.eval()

    data_loader = torch.utils.data.DataLoader(samples,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True)

    # with torch.no_grad():
    #     for x, y in test_samples:
    #         net


if __name__ == '__main__':
    # fix seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # run experiment
    t1 = perf_counter()
    train_samples = generate_samples(N_SAMPLES, SEED)

    t2 = perf_counter()
    net = train_qnetwork(train_samples)
    t3 = perf_counter()

    # test
    # test_samples = generate_samples(1000, 1)
    # test_qnetwork(net, test_samples)

    # save
    torch.save(net.state_dict(), BaselineModelPath_10x10_4v2)

    print(f'Generated samples in {(t2 - t1) / 60.:.2f} min.')
    print(f'Trained in {(t3 - t2) / 60.:.2f} min.')
