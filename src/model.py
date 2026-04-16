import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class IsotopeGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout_rate=0.1):
        super(IsotopeGNN, self).__init__()
        self.dropout_rate = dropout_rate

        # SAGEConv is memory efficient.
        # Output dim is exactly hidden_dim (no multiplication by heads)
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

        # Mean prediction head
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # Must be hidden_dim
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Uncertainty head
        self.sigma_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # Must be hidden_dim
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, edge_index, force_dropout=False):
        training_mode = self.training or force_dropout

        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_rate, training=training_mode)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_rate, training=training_mode)

        # Predictions
        mu = self.mu_head(x)
        log_var = self.sigma_head(x)

        # Clamp log_var to prevent exp() from exploding
        # -10 to 10 allows variance between 0.00004 and 22026
        log_var = torch.clamp(log_var, min=-10, max=10)

        return mu, log_var
