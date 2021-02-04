from time import perf_counter
from copy import copy

import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import ma_gym  # register new envs on import

from agents.constants import SpiderAndFlyEnv, RolloutModelPath_10x10_4v2
from agents.baseline_agent_nn import BaselineAgentNn
from agents.qnetwork_rollout import QNetworkRollout

N_AGENTS = 4
N_PREY = 2

N_SAMPLES = 1_000

BATCH_SIZE = 128

EPOCHS = 10

SEED = 42


def run_simulation(env, agent, obs, action_id):
    state = copy(env)



    # run 100 episodes for a random agent
    while not all(done_n):

        # all agents act based on the observation
        act_n = []
        for agent, obs in zip(agents, obs_n):
            best_action = agent.act(obs)
            act_n.append(best_action)

        # update step ->
        obs_n, reward_n, done_n, info = env.step(act_n)

    state.close()


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
        agents = [BaselineAgentNn(i, n_agents, n_preys, env.action_space[i]) for i in range(n_agents)]

        # init stopping condition
        done_n = [False] * n_agents

        # run 100 episodes for a random agent
        while not all(done_n):
            #

            # all agents act based on the observation
            act_n = []
            for agent, obs in zip(agents, obs_n):

                # TODO simulate 5 actions, select the best one, assume Baseline policy

                for action_id in range(env.action_space[0].n):
                    # copy env
                    total_return = run_simulation(env, agent, obs, action_id)

                    # run simulation until the end

                best_action = agent.act(obs)
                act_n.append(best_action)

            # update step ->
            obs_n, reward_n, done_n, info = env.step(act_n)

        env.close()

    print('Finished sample generation.')

    return samples[:N_SAMPLES]


def train_qnetwork(samples):
    print('Started Training.')

    net = QNetworkRollout(N_AGENTS, N_PREY)

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
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.float())

            loss = criterion(outputs.float(), labels.float())

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        if epoch % 10 == 0:
            torch.save(net.state_dict(), RolloutModelPath_10x10_4v2)

        print(f'[{epoch}] {running_loss / n_batches:.3f}.')

    print('Finished Training.')

    return net


if __name__ == '__main__':
    # fix seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # do simulations -> obtain samples
    t1 = perf_counter()
    train_samples = generate_samples(N_SAMPLES, SEED)

    # train rollout network
    t2 = perf_counter()
    net = train_qnetwork(train_samples)
    t3 = perf_counter()

    # save
    torch.save(net.state_dict(), RolloutModelPath_10x10_4v2)

    print(f'Generated samples in {(t2 - t1) / 60.:.2f} min.')
    print(f'Trained in {(t3 - t2) / 60.:.2f} min.')
