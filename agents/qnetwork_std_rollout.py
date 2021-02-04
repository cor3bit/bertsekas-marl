import torch.nn as nn


class QNetworkStdRollout(nn.Module):
    def __init__(self, n_agents, m_preys, action_space_size):
        super(QNetworkStdRollout, self).__init__()

        # coords of agents & preys + alive status
        input_dims = int((n_agents + m_preys) * 2 + m_preys)

        output_dims = int(action_space_size * n_agents)

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

            nn.Linear(64, output_dims),
        )

    def forward(self, x):
        x = self.net(x)
        return x
