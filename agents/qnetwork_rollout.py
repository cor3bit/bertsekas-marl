import torch.nn as nn


class QNetworkRollout(nn.Module):
    def __init__(self, n_agents, m_preys):
        super(QNetworkRollout, self).__init__()

        # coords of agents & preys + alive status + OHE vector of agents (active agent) +
        # + OHE vector of agents (taken action)
        input_dims = (n_agents + m_preys) * 2 + m_preys + n_agents + n_agents

        self.net = nn.Sequential(
            nn.Linear(input_dims, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
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
