import torch

from src.model import HybridIsotopologueGATv2

CONFIG = {
    "model": {"hidden_dim": 8, "heads": 2, "dropout": 0.1, "edge_embedding_dim": 4, "num_edge_types": 2},
    "data": {"physics_features": ["mass_ratio_A", "mass_ratio_B", "mu_vib_ratio", "mu_rot_ratio"]},
}
NUM_NODE_FEATURES = 3 + 4 + 1  # v, J, ECalc + 4 ratio terms + j_ext_j1


def test_bypass_always_includes_j_term():
    model = HybridIsotopologueGATv2(CONFIG, NUM_NODE_FEATURES)
    assert model.num_ratio_features == 4
    assert model.num_bypass_features == 5
    assert model.head_physics.in_features == 5


def test_forward_runs_on_tiny_graph():
    model = HybridIsotopologueGATv2(CONFIG, NUM_NODE_FEATURES)
    x = torch.randn(6, NUM_NODE_FEATURES)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    edge_attr = torch.tensor([0, 1, 0, 1])

    out = model(x, edge_index, edge_attr)

    assert out.shape == (6,)


def test_j_term_is_not_zero_centered_but_ratios_are():
    # Ratio features are centered by -1.0 (parent isotopologue baseline);
    # j_ext_j1 is left untouched since it's naturally zero at J=0.
    model = HybridIsotopologueGATv2(CONFIG, NUM_NODE_FEATURES)
    model.eval()

    # x columns: v, J, ECalc, 4 ratio terms (=1.0, i.e. parent baseline), j_ext_j1
    x = torch.zeros(1, NUM_NODE_FEATURES)
    x[0, 3:7] = 1.0  # ratio terms at parent baseline
    x[0, 7] = 42.0  # j_ext_j1

    bypass_block = x[:, -model.num_bypass_features:]
    ratio_features = bypass_block[:, : model.num_ratio_features] - 1.0
    j_feature = bypass_block[:, model.num_ratio_features:]

    assert torch.allclose(ratio_features, torch.zeros(1, 4))
    assert torch.allclose(j_feature, torch.tensor([[42.0]]))
