import logging

import pandas as pd

from src.dataset import generate_graph_edges

logger = logging.getLogger("test")


def _states_df(rows):
    df = pd.DataFrame(rows, columns=["molecule", "iso_id", "v", "J"])
    df["node_idx"] = df.index
    return df


def test_single_isotopologue_produces_only_physical_edges():
    # PS-mode: one isotopologue, states (v, J) = (0,0),(0,1),(0,2),(1,0),(1,1)
    df = _states_df(
        [
            ("SiO", 28, 0, 0),
            ("SiO", 28, 0, 1),
            ("SiO", 28, 0, 2),
            ("SiO", 28, 1, 0),
            ("SiO", 28, 1, 1),
        ]
    )

    edge_index, edge_attr = generate_graph_edges(df, logger)

    assert edge_index.shape[1] > 0
    # No isotope-shift edges possible with only one isotopologue present.
    assert (edge_attr == 1).sum().item() == 0
    assert (edge_attr == 0).sum().item() == edge_index.shape[1]


def test_multi_isotopologue_produces_both_edge_types():
    # IE-mode: two isotopologues sharing (v=0, J=0) and (v=0, J=1)
    df = _states_df(
        [
            ("CO", 26, 0, 0),
            ("CO", 26, 0, 1),
            ("CO", 36, 0, 0),
            ("CO", 36, 0, 1),
        ]
    )

    edge_index, edge_attr = generate_graph_edges(df, logger)

    physical_edges = (edge_attr == 0).sum().item()
    isotopic_edges = (edge_attr == 1).sum().item()

    # Physical: (26,0,0)<->(26,0,1) and (36,0,0)<->(36,0,1), each bidirectional
    assert physical_edges == 4
    # Isotopic: (26,0,0)<->(36,0,0) and (26,0,1)<->(36,0,1), each bidirectional
    assert isotopic_edges == 4


def test_isotope_edges_never_cross_molecules():
    # Same iso_id/v/J happening to coincide across two different molecules
    # must never be linked -- generate_graph_edges groups by molecule first.
    df = _states_df(
        [
            ("CO", 26, 0, 0),
            ("CN", 26, 0, 0),
        ]
    )

    edge_index, edge_attr = generate_graph_edges(df, logger)

    assert edge_index.shape[1] == 0
