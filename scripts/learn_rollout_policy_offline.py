from time import perf_counter
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import ma_gym  # register new envs on import

from src.constants import SpiderAndFlyEnv, RolloutModelPath_10x10_4v2
from src.qnetwork_coordinated import QNetworkCoordinated
from src.agent_seq_rollout import SequentialRolloutAgent

SEED = 42

M_AGENTS = 4
P_PREY = 2

N_SAMPLES = 200_000
BATCH_SIZE = 1024
EPOCHS = 500
FROM_SCRATCH = False
N_SIMS_MC = 50


def generate_samples(n_samples, seed):
    print('Started sample generation.')

    samples = []

    # create Spider-and-Fly game
    env = gym.make(SpiderAndFlyEnv)

    # TODO Switched off for re-training
    # env.seed(seed)

    with tqdm(total=n_samples) as pbar:
        while len(samples) < n_samples:
            # init env
            obs_n = env.reset()

            # init agents
            m_agents = env.n_agents
            p_preys = env.n_preys
            grid_shape = env._grid_shape
            action_space = env.action_space[0]

            agents = [SequentialRolloutAgent(i, m_agents, p_preys, grid_shape,
                                             env.action_space[i], n_sim_per_step=N_SIMS_MC)
                      for i in range(m_agents)]

            # init stopping condition
            done_n = [False] * m_agents

            # run 100 episodes for a random agent
            while not all(done_n):
                # for each agent calculates Manhattan Distance to each prey for each

                # transform into samples
                obs_first = np.array(obs_n[0], dtype=np.float32).flatten()  # same for all agent

                prev_actions = {}
                act_n = []
                for i, (agent, obs) in enumerate(zip(agents, obs_n)):
                    best_action, sim_results, actions_other_agents = agent.act_with_info(
                        obs, prev_actions=prev_actions)

                    # create an (x,y) sample for QNet
                    agent_ohe = np.zeros(shape=(m_agents,), dtype=np.float32)
                    agent_ohe[i] = 1.

                    prev_actions_ohe = np.zeros(shape=(m_agents * action_space.n,), dtype=np.float32)

                    for agent_i, action_i in enumerate(actions_other_agents):
                        # current agent's actions is the output of QNet, not input
                        if agent_i == i:
                            continue
                        elif agent_i in prev_actions:
                            ohe_action_index = int(agent_i * action_space.n) + prev_actions[agent_i]
                            prev_actions_ohe[ohe_action_index] = 1.
                        # TODO
                        # else:
                        #     ohe_action_index = int(agent_i * action_space.n) + action_i
                        #     prev_actions_ohe[ohe_action_index] = 1.

                    x = np.concatenate((obs_first, agent_ohe, prev_actions_ohe))

                    np_sim_results = np.array(sim_results, dtype=np.float32)
                    np_sim_results_sorted = np_sim_results[np.argsort(np_sim_results[:, 0])]
                    y = np_sim_results_sorted[:, 1]

                    samples.append((x, y))
                    pbar.update(1)

                    # best action taken for the agent i
                    prev_actions[i] = best_action
                    act_n.append(best_action)

                # update step
                obs_n, reward_n, done_n, info = env.step(act_n)

    env.close()

    print('Finished sample generation.')

    return samples[:n_samples]


def train_qnetwork(samples):
    print('Started Training.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Found device: {device}.')

    net = QNetworkCoordinated(M_AGENTS, P_PREY, 5)
    if not FROM_SCRATCH:
        net.load_state_dict(torch.load(RolloutModelPath_10x10_4v2))

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
            torch.save(net.state_dict(), RolloutModelPath_10x10_4v2)

    print('Finished Training.')

    return net


if __name__ == '__main__':
    # fix seed
    # TODO fix seed
    # np.random.seed(SEED)
    # torch.manual_seed(SEED)

    # collect samples
    t1 = perf_counter()
    train_samples = generate_samples(N_SAMPLES, SEED)

    # train net
    t2 = perf_counter()
    net = train_qnetwork(train_samples)
    t3 = perf_counter()

    # save
    torch.save(net.state_dict(), RolloutModelPath_10x10_4v2)

    print(f'Generated samples in {(t2 - t1) / 60.:.2f} min.')
    print(f'Trained in {(t3 - t2) / 60.:.2f} min.')
