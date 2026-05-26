import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class HybridIsotopologueGATv2(nn.Module):
    """
    A physics-informed Graph Attention Network tailored for diatomic isotopic shifts.

    The architecture splits the prediction into two pathways:
    1. A Graph Topology Trunk (GATv2) that learns state-dependent corrections
       (e.g., local perturbations, centrifugal distortion) using the full feature set.
    2. A Strictly Linear Physics Bypass that explicitly scales independent Born-Oppenheimer
       breakdown mass ratios without activation functions to ensure stable out-of-bounds extrapolation.

    Args:
        config (dict): Configuration dictionary containing model hyperparameters.
        num_node_features (int): Total number of input features in the node tensor 'x'.
    """

    def __init__(self, config, num_node_features):
        super(HybridIsotopologueGATv2, self).__init__()

        # 1. Extract hyperparameters
        hidden_dim = config["model"]["hidden_dim"]
        heads = config["model"]["heads"]
        dropout = config["model"]["dropout"]
        edge_emb_dim = config["model"]["edge_embedding_dim"]
        num_edge_types = config["model"]["num_edge_types"]

        self.dropout_rate = dropout

        # Determine how many features belong to the strictly linear bypass
        # (These are always appended to the end of the feature tensor)
        self.num_bypass_features = len(config["data"]["physics_features"])

        # 2. Edge Embedding Layer
        # (Converts discrete edge types [0, 1, 2] into dense continuous vectors to prevent attention zeroing)
        self.edge_emb = nn.Embedding(
            num_embeddings=num_edge_types, embedding_dim=edge_emb_dim
        )

        # 3. GNN Trunk (Topology Pathway)
        # (in_channels uses the full feature set so the network can differentiate isotopologues)
        self.conv1 = GATv2Conv(
            in_channels=num_node_features,
            out_channels=hidden_dim,
            heads=heads,
            concat=True,
            edge_dim=edge_emb_dim,
        )

        # The output of conv1 is concatenated across all heads (hidden_dim * heads)
        conv1_out_dim = hidden_dim * heads

        self.conv2 = GATv2Conv(
            in_channels=conv1_out_dim,
            out_channels=hidden_dim * 2,
            heads=1,
            concat=False,
            edge_dim=edge_emb_dim,
        )

        # 4. Topology MLP Head
        # (Processes the graph representation into a non-linear state correction)
        self.head_topology = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_dim, 1),
        )

        # 5. Explicit Physics Bypass Head
        # (Strictly linear mapping of mass ratios; bias=False prevents parent baseline shifting)
        self.head_physics = nn.Linear(self.num_bypass_features, 1, bias=False)

    def forward(self, x, edge_index, edge_attr):
        """
        Executes the forward pass of the model.

        Args:
            x (torch.Tensor): Node feature matrix of shape [num_nodes, num_node_features].
            edge_index (torch.Tensor): Graph connectivity matrix of shape [2, num_edges].
            edge_attr (torch.Tensor): Discrete edge types of shape [num_edges] or [num_edges, 1].

        Returns:
            torch.Tensor: Predicted energy shifts for each node, shape [num_nodes].
        """
        # 1. Format and Embed Edge Attributes
        if edge_attr.dim() == 2:
            edge_attr = edge_attr.squeeze(-1)

        edge_attr_discrete = edge_attr.long()
        edge_attr_dense = self.edge_emb(edge_attr_discrete)

        # 2. Extract and Zero-Center the Physics Features
        # (Slices the exact number of physics features from the end of the tensor)
        # (Subtracting 1.0 ensures the parent isotope baseline is exactly 0.0)
        bypass_features = x[:, -self.num_bypass_features :] - 1.0

        # 3. Execute Graph Topology Pathway
        # (The full 'x' is passed so the network recognizes specific isotopes during message passing)
        graph_x = self.conv1(x, edge_index, edge_attr_dense)
        graph_x = F.gelu(graph_x)
        graph_x = F.dropout(graph_x, p=self.dropout_rate, training=self.training)

        graph_x = self.conv2(graph_x, edge_index, edge_attr_dense)
        graph_x = F.gelu(graph_x)

        topology_correction = self.head_topology(graph_x)

        # 4. Execute Pure Linear Physics Pathway
        physics_correction = self.head_physics(bypass_features)

        # 5. Aggregate Predictions
        out = topology_correction + physics_correction

        return out.squeeze(-1)
