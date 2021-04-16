import torch.nn as nn


class QNetworkCoordinated(nn.Module):
    def __init__(self, m_agents, p_preys, action_size):
        super(QNetworkCoordinated, self).__init__()

        # coords of agents & preys + alive status + OHE vector of agents (active agent)
        state_dims = (m_agents + p_preys) * 2 + p_preys
        input_dims = state_dims + m_agents + m_agents * action_size

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

            nn.Linear(64, action_size),
        )

    def forward(self, x):
        x = self.net(x)
        return x
