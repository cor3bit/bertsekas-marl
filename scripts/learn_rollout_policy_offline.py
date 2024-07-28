from time import perf_counter
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import ma_gym  # register new envs on import

from src.constants import SpiderAndFlyEnv, RolloutModelPath_10x10_4v2, RepeatedRolloutModelPath_10x10_4v2, AgentType, \
    QnetType
from src.qnetwork_coordinated import QNetworkCoordinated
from src.agent_seq_rollout import SeqRolloutAgent


import wandb

SEED = 42

M_AGENTS = 4
P_PREY = 2

N_SAMPLES = 50_000
BATCH_SIZE = 1024
EPOCHS = 500
N_SIMS_MC = 50
FROM_SCRATCH = False
INPUT_QNET_NAME = RepeatedRolloutModelPath_10x10_4v2
OUTPUT_QNET_NAME = RepeatedRolloutModelPath_10x10_4v2
BASIS_POLICY_AGENT = AgentType.QNET_BASED
QNET_TYPE = QnetType.BASELINE


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

            agents = [SeqRolloutAgent(
                i, m_agents, p_preys, grid_shape, env.action_space[i],
                n_sim_per_step=N_SIMS_MC,
                basis_agent_type=BASIS_POLICY_AGENT,
                qnet_type=QNET_TYPE,
            ) for i in range(m_agents)]

            # init stopping condition
            done_n = [False] * m_agents

            while not all(done_n):
                prev_actions = {}
                act_n = []
                for i, (agent, obs) in enumerate(zip(agents, obs_n)):
                    best_action, action_q_values = agent.act_with_info(
                        obs, prev_actions=prev_actions)

                    # create an (x,y) sample for QNet
                    agent_ohe = np.zeros(shape=(m_agents,), dtype=np.float32)
                    agent_ohe[i] = 1.

                    prev_actions_ohe = np.zeros(shape=(m_agents * action_space.n,), dtype=np.float32)
                    for agent_i, action_i in prev_actions.items():
                        ohe_action_index = int(agent_i * action_space.n) + prev_actions[agent_i]
                        prev_actions_ohe[ohe_action_index] = 1.

                    obs_first = np.array(obs, dtype=np.float32).flatten()  # same for all agent
                    x = np.concatenate((obs_first, agent_ohe, prev_actions_ohe))
                    

                    samples.append((x, action_q_values))
                    pbar.update(1)
                    if len(samples) == N_SAMPLES:
                        env.close()
                        return samples

                    # best action taken for the agent i
                    prev_actions[i] = best_action
                    act_n.append(best_action)

                # update step
                obs_n, reward_n, done_n, info = env.step(act_n)

    env.close()

    print('Finished sample generation.')

    return samples[:n_samples]


def train_qnetwork(samples):
    steps_num = 0
    print('Started Training.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Found device: {device}.')

    net = QNetworkCoordinated(M_AGENTS, P_PREY, 5)
    if not FROM_SCRATCH:
        net.load_state_dict(torch.load(INPUT_QNET_NAME))

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
            wandb.log({'loss':running_loss},step=epoch) 
            n_batches += 1

        print(f'[{epoch}] {running_loss / n_batches:.3f}.')

        # save interim results
        # if epoch % 10 == 0:
        #     torch.save(net.state_dict(), OUTPUT_QNET_NAME)

    print('Finished Training.')

    return net


if __name__ == '__main__':
    # fix seed
    # TODO fix seed
    # np.random.seed(SEED)
    # torch.manual_seed(SEED)
    wandb.init(project="Training_SecurityAndSurveillance",name="Sequential Rollout")
    # collect samples
    t1 = perf_counter()
    train_samples = generate_samples(N_SAMPLES, SEED)

    # train net
    t2 = perf_counter()
    net = train_qnetwork(train_samples)
    t3 = perf_counter()

    # save
    torch.save(net.state_dict(), OUTPUT_QNET_NAME)

    print(f'Generated samples in {(t2 - t1) / 60.:.2f} min.')
    print(f'Trained in {(t3 - t2) / 60.:.2f} min.')

    
