import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, n_agents, m_preys):
        super(QNetwork, self).__init__()

        # coords of agents & preys + alive status + OHE vector of agents
        input_dims = (n_agents + m_preys) * 2 + m_preys + n_agents

        self.net = nn.Sequential(
            nn.Linear(input_dims, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 5),
        )

    def forward(self, x):
        x = self.net(x)
        return x
