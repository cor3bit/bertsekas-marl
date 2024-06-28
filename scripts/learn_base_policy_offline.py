from time import perf_counter
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import ma_gym  # register new envs on import

from src.constants import SpiderAndFlyEnv, BaselineModelPath_10x10_4v2
from src.qnetwork import QNetwork
from src.agent_rule_based import RuleBasedAgent

SEED = 42

M_AGENTS = 4
P_PREY = 2

N_SAMPLES = 2_000_000
BATCH_SIZE = 512
EPOCHS = 20


def generate_samples(n_samples, seed):
    print('Started sample generation.')

    samples = []

    # create Spider-and-Fly game
    env = gym.make(SpiderAndFlyEnv)

    env.seed(seed)

    with tqdm(total=n_samples) as pbar:
        while len(samples) < n_samples:
            # init env
            obs_n = env.reset()

            # init agents
            m_agents = env.n_agents
            p_preys = env.n_preys
            grid_shape = env._grid_shape

            agents = [RuleBasedAgent(i, m_agents, p_preys, grid_shape, env.action_space[i]) for i in range(m_agents)]

            # init stopping condition
            done_n = [False] * m_agents

            # run 100 episodes for a random agent
            # run while all agents are done (done_n = True)
            while not all(done_n):
                # for each agent calculates Manhattan Distance to each prey for each

                # transform into samples
                obs_first = np.array(obs_n[0], dtype=np.float32).flatten()  # same for all agent (Fully observable)

                act_n = []
                for i, (agent, obs) in enumerate(zip(agents, obs_n)):
                    agent_ohe = np.zeros(shape=(m_agents,), dtype=np.float32)
                    agent_ohe[i] = 1.

                    action_id, action_distances = agent.act_with_info(obs)

                    if np.inf in action_distances:
                        max_without_inf = np.max(action_distances[~np.isinf(action_distances)])
                        action_distances[action_distances == np.inf] = max_without_inf

                    x = np.concatenate((obs_first, agent_ohe))
                    y = -action_distances
                    samples.append((x, y))
                    pbar.update(1)

                    act_n.append(action_id)

                # update step
                obs_n, reward_n, done_n, info = env.step(act_n)

    env.close()

    print('Finished sample generation.')

    return samples[:n_samples]


def train_qnetwork(samples):
    print('Started Training.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Found device: {device}.')

    # agents, preys, action size
    net = QNetwork(M_AGENTS, P_PREY, 5)

    net.to(device)

    net.train()  # check

    criterion = nn.MSELoss()

    optimizer = optim.Adam(net.parameters(), lr=0.01)

    data_loader = torch.utils.data.DataLoader(samples,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True)

    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        running_loss = .0
        n_batches = 0

        for data in data_loader:
            # TODO optimize
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            # logging
            running_loss += loss.item()
            n_batches += 1

        print(f'[{epoch}] {running_loss / n_batches:.3f}.')

        # save interim results
        if epoch % 10 == 0:
            torch.save(net.state_dict(), BaselineModelPath_10x10_4v2)

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
    n_workers = 5
    chunk = int(N_SAMPLES / n_workers)
    train_samples = []

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = []
        for _ in range(n_workers):
            futures.append(pool.submit(generate_samples, chunk, SEED))

        for f in as_completed(futures):
            samples_part = f.result()
            train_samples += samples_part

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
