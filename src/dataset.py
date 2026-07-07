import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

from src.registry import get_molecule

# Columns an ingested CSV should carry for PS-band grouping. Older CSVs
# (ingested before this was required) may lack them -- fall back to a
# placeholder so the pipeline still runs, but the PS baseline's band split
# degrades to (v) alone for that molecule until it's re-ingested.
BAND_KEY_COLS = ["ElecState", "Omega", "parity"]
BAND_KEY_DEFAULTS = {"ElecState": "UNKNOWN", "Omega": np.nan, "parity": ""}


def _read_isotopologue_csv(input_dir, iso_name, logger):
    file_path = os.path.join(input_dir, f"{iso_name}.csv")
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path)

    missing_band_cols = [c for c in BAND_KEY_COLS if c not in df.columns]
    if missing_band_cols:
        logger.warning(
            f"  - {iso_name}: missing band-key columns {missing_band_cols} "
            "(legacy ingestion) -- PS baseline will group by v only for this molecule."
        )
        for col in missing_band_cols:
            df[col] = BAND_KEY_DEFAULTS[col]

    return df


def load_and_preprocess_data(config, logger):
    """
    Iterates through the configured molecules (by registry name), loads CSV
    files, applies energy cutoffs, calculates physical ratios, and
    concatenates the processed data into a global dataframe.

    Args:
        config (dict): Configuration dictionary containing data and molecule parameters.
        logger (logging.Logger): Logger instance for recording execution steps.

    Returns:
        pd.DataFrame: A single concatenated dataframe containing all valid nodes
        across all specified molecules and isotopes.
    """
    input_dir = config["data"]["input_dir"]
    max_energy = config["data"]["max_energy_cutoff"]
    physics_features = config["data"]["physics_features"]
    optional_features = config["data"]["optional_quantum_features"]
    registry_path = config["data"].get("registry", "configs/molecules.yaml")

    all_nodes = []

    for mol_name in config["molecules"]:
        logger.info(f"Processing Molecule: {mol_name}")
        mol_entry = get_molecule(mol_name, registry_path)

        # Parent reference masses come from the parent isotopologue's own CSV
        # (already resolved at ingestion time via src/masses.py) rather than
        # being re-typed in the experiment config.
        parent_df = _read_isotopologue_csv(input_dir, mol_entry["parent_isotope"], logger)
        if parent_df is None:
            raise FileNotFoundError(
                f"{mol_name}: parent isotopologue CSV {mol_entry['parent_isotope']}.csv not found in {input_dir}"
            )
        parent_mass_A = parent_df["mass_A"].iloc[0]
        parent_mass_B = parent_df["mass_B"].iloc[0]
        parent_mu = parent_df["reduced_mass"].iloc[0]

        for iso_name, iso_id in zip(mol_entry["isotopes"], mol_entry["isotope_ids"]):
            df = _read_isotopologue_csv(input_dir, iso_name, logger)
            if df is None:
                logger.warning(f"File {iso_name}.csv not found in {input_dir}. Skipping.")
                continue

            # 1. Apply Energy Cutoff
            initial_count = len(df)
            df = df[df["ECalc"] <= max_energy].copy()
            logger.info(
                f"  - Iso {iso_id}: Dropped {initial_count - len(df)} high-energy states. {len(df)} remaining."
            )

            if config["data"]["max_diff_cutoff"] is not None:
                diff_cutoff = config["data"]["max_diff_cutoff"]
                initial_count = len(df)
                df = df[df["Ediff"].abs() <= diff_cutoff].copy()
                logger.info(
                    f"  - Iso {iso_id}: Dropped {initial_count - len(df)} states exceeding diff cutoff. {len(df)} remaining."
                )

            # 2. Tag with metadata for graph grouping
            df["molecule"] = mol_name
            df["iso_id"] = iso_id

            # 3. Calculate Base Dunham & Mass Ratios
            df["mass_ratio_A"] = df["mass_A"] / parent_mass_A
            df["mass_ratio_B"] = df["mass_B"] / parent_mass_B
            df["mu_vib_ratio"] = (df["reduced_mass"] / parent_mu) ** -0.5
            df["mu_rot_ratio"] = (df["reduced_mass"] / parent_mu) ** -1.0

            # (If higher order terms are specified in the configuration, they are applied here)
            if "mu_anharmonic_ratio" in physics_features:
                df["mu_anharmonic_ratio"] = (df["reduced_mass"] / parent_mu) ** -1.5
            if "mu_centrifugal_ratio" in physics_features:
                df["mu_centrifugal_ratio"] = (df["reduced_mass"] / parent_mu) ** -2.0

            # (Always available to the model's physics bypass, mirroring PS's own
            # a*J_ext(J_ext+1)+sigma functional form -- see model.py)
            df["j_ext_j1"] = df["J"] * (df["J"] + 1)

            # 4. Handle Optional Quantum Features
            # (Padding for closed-shell molecules to maintain consistent tensor dimensions)
            for opt_feat in optional_features:
                if opt_feat not in df.columns:
                    df[opt_feat] = 0.0

            # 5. Calculate Target Residual
            # (MARVEL states possess an EMarv value; synthetic states evaluate to NaN)
            if "EMarv" in df.columns:
                df["is_known_marvel"] = df["EMarv"].notna()
            else:
                df["is_known_marvel"] = False

            all_nodes.append(df)

    # Combine into one global dataframe
    global_df = pd.concat(all_nodes, ignore_index=True)

    # Reset the index to establish a clean 0 to N contiguous ID for PyTorch Geometric
    global_df = global_df.reset_index(drop=True)
    global_df["node_idx"] = global_df.index

    return global_df


def generate_graph_edges(nodes_df, logger):
    """
    Generates Physical (Type 0) and Isotopic (Type 1) edges for the graph representation.
    Strictly prevents cross-molecule connections.

    Args:
        nodes_df (pd.DataFrame): The preprocessed global dataframe containing all valid nodes.
        logger (logging.Logger): Logger instance for recording execution steps.

    Returns:
        tuple: A tuple containing:
            - edge_index (torch.Tensor): A 2xN tensor of source and destination indices.
            - edge_attr (torch.Tensor): A 1D tensor mapping each edge to its categorical type (0 or 1).
    """
    logger.info("Generating Graph Edges...")
    edges_src = []
    edges_dst = []
    edge_types = []

    # Group by molecule to ensure isolated subgraphs
    for mol_name, mol_df in nodes_df.groupby("molecule"):

        # Fast lookup dictionary: (iso_id, v, J) -> node_idx
        state_lookup = mol_df.set_index(["iso_id", "v", "J"])["node_idx"].to_dict()

        # Group identical (v, J) states across different isotopologues
        v_j_groups = {}
        for (iso, v, J), idx in state_lookup.items():
            if (v, J) not in v_j_groups:
                v_j_groups[(v, J)] = []
            v_j_groups[(v, J)].append((iso, idx))

        for (iso, v, J), src_idx in state_lookup.items():

            # 1. Physical Rovibrational Transitions (Within SAME isotope)
            allowed_neighbors = [
                (iso, v, J - 1),
                (iso, v, J + 1),
                (iso, v - 1, J - 1),
                (iso, v - 1, J + 1),
            ]
            for target_key in allowed_neighbors:
                if target_key in state_lookup:
                    edges_src.append(src_idx)
                    edges_dst.append(state_lookup[target_key])
                    edge_types.append(0)

            # 2. Isotope Shift Edges (Across DIFFERENT isotopes, SAME molecule)
            # (Single-isotopologue molecules naturally produce zero edges of
            # this type -- the model then trains on physical edges alone,
            # which is exactly the PS-style J-extrapolation regime.)
            for other_iso, dst_idx in v_j_groups[(v, J)]:
                if other_iso != iso:
                    edges_src.append(src_idx)
                    edges_dst.append(dst_idx)
                    edge_types.append(1)

    # Convert to PyTorch tensors
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_types, dtype=torch.long)

    logger.info(
        f"Graph constructed with {len(nodes_df)} nodes and {edge_index.shape[1]} edges."
    )
    return edge_index, edge_attr


def prepare_molecule_graph(config, logger):
    """
    Master pipeline function that generates the final PyTorch Geometric graph object
    and the corresponding processed dataframe.

    Args:
        config (dict): Configuration dictionary containing data and molecule parameters.
        logger (logging.Logger): Logger instance for recording execution steps.

    Returns:
        tuple: A tuple containing:
            - pyg_graph (torch_geometric.data.Data): The un-scaled PyG graph object containing x, y, edge_index, and edge_attr.
            - nodes_df (pd.DataFrame): The processed dataframe matching the nodes in the graph.
            - feature_cols (list): The list of column names used to build the feature tensor.
    """
    nodes_df = load_and_preprocess_data(config, logger)
    edge_index, edge_attr = generate_graph_edges(nodes_df, logger)

    # Fill missing targets with 0.0 for stable tensor conversion
    # (Masks hide these targets during training)
    nodes_df["Ediff"] = nodes_df["Ediff"].fillna(0.0)

    # Define the base feature list
    # (Topology + Optional + Physics ratios + J(J+1), in this exact order --
    # model.py slices the bypass features off the end of this tensor)
    feature_cols = (
        ["v", "J", "ECalc"]
        + config["data"]["optional_quantum_features"]
        + config["data"]["physics_features"]
        + ["j_ext_j1"]
    )

    # Create the raw feature tensor
    # (Scaling occurs dynamically per fold within the training loop)
    x = torch.tensor(nodes_df[feature_cols].values, dtype=torch.float32)
    y = torch.tensor(nodes_df["Ediff"].values, dtype=torch.float32)

    # Build base PyG Data object
    pyg_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    return pyg_graph, nodes_df, feature_cols
