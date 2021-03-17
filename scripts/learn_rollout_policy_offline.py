from time import perf_counter
from copy import copy

from tqdm import tqdm
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

N_SAMPLES = 5_000
EPOCHS = 100
BATCH_SIZE = 512

N_SIMS_PER_ACTION = 10
EPSILON = .05

SEED = 42


def run_simulation(env, agents, prev_actions, obs, agent_id, action_id):
    env_copy = copy(env)

    total_reward = 0.

    act_n = []
    for agent, prev_action in zip(agents, prev_actions):
        # previously optimal actions
        if agent.id < agent_id:
            act_n.append(prev_action)
        # simulated action
        elif agent.id == agent_id:
            act_n.append(action_id)
        # baseline policy actions
        else:
            best_action = agent.act(obs)
            act_n.append(best_action)

    # finish sub-interval
    obs_n, reward_n, done_n, info = env_copy.step(act_n)
    total_reward += reward_n[agent_id]

    # play episode till the end and collect the reward
    while not all(done_n):
        act_n = []

        for agent, obs in zip(agents, obs_n):
            best_action = agent.act(obs)
            act_n.append(best_action)

        obs_n, reward_n, done_n, info = env_copy.step(act_n)

        total_reward += reward_n[agent_id]

    env_copy.close()

    return total_reward


def generate_samples(n_samples, seed):
    print('Started sample generation.')

    samples = []

    # create Spider-and-Fly game
    env = gym.make(SpiderAndFlyEnv)
    env.seed(seed)
    action_space = env.action_space[0]
    action_space_n = action_space.n

    with tqdm(total=n_samples) as pbar:
        while len(samples) < n_samples:

            # init env
            obs_n = env.reset()

            # init agents
            n_agents = env.n_agents
            n_preys = env.n_preys

            agents = [BaselineAgentNn(i, n_agents, n_preys, env.action_space[i]) for i in range(n_agents)]

            # init stopping condition
            done_n = [False] * n_agents

            while not all(done_n):

                # TODO (not urgent) shuffle agents before each move

                prev_actions = []

                for agent, obs in zip(agents, obs_n):
                    agent_id = agent.id

                    agent_ohe = np.zeros(shape=(n_agents,), dtype=np.float32)
                    agent_ohe[agent_id] = 1.0

                    actions_total_returns = np.zeros(shape=(action_space_n,), dtype=np.float32)

                    for action_id in range(action_space_n):
                        # copy env
                        total_return = .0
                        for _ in range(N_SIMS_PER_ACTION):
                            total_return += run_simulation(env, agents, prev_actions, obs, agent_id,
                                                           action_id) / N_SIMS_PER_ACTION

                        actions_total_returns[action_id] = total_return

                        # run simulation until the end

                    # TODO select the best action
                    p = np.random.random()
                    if p < EPSILON:
                        # random action -> exploration
                        best_action = action_space.sample()
                    else:
                        best_action = np.argmax(actions_total_returns)

                    prev_actions.append(best_action)

                    # previous actions
                    prev_actions_ohe = np.zeros(shape=(N_AGENTS * action_space_n,), dtype=np.float32)
                    if prev_actions:
                        for agent_id, prev_action in enumerate(prev_actions):
                            # TODO check!!!
                            pos = int(agent_id * action_space_n + prev_action)
                            prev_actions_ohe[pos] = 1.

                    x = np.concatenate((np.array(obs, dtype=np.float32), agent_ohe, prev_actions_ohe))

                    samples.append((x, actions_total_returns))
                    pbar.update(1)

                    # best_action = agent.act(obs)

                    # act_n.append(best_action)

                    # add to prev_actions only after
                    # prev_actions[agent_id] = best_action
                    # scaled_prev_actions[agent_id] = (best_action + 1) / (action_space_n + 1)

                    sub_obs_n, sub_reward_n, sub_done_n, sub_info = env.substep(agent_id, best_action)

                # TODO
                obs_n = sub_obs_n
                assert done_n is not None

                reward_n = sub_reward_n
                done_n = sub_done_n
                info = sub_info

            env.close()

    print('Finished sample generation.')

    return samples[:N_SAMPLES]


def train_qnetwork(samples):
    print('Started Training.')

    net = QNetworkRollout(N_AGENTS, N_PREY, 5)

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
            outputs = net(inputs)

            loss = criterion(outputs, labels)

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
