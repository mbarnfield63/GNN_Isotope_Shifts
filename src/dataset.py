import os
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from src.config import ISOTOPOLOGUE_CONFIGS


class IsotopeDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        # Allow the PyG classes to be unpickled safely
        torch.serialization.add_safe_globals([Data])
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_dir(self):
        return os.path.join(self.root, "preprocessed")

    @property
    def processed_dir(self):
        return os.path.join(self.root, "processed")

    @property
    def raw_file_names(self):
        # Look for the 6 CSV files defined in your config
        return [f"{iso}.csv" for iso in ISOTOPOLOGUE_CONFIGS.keys()]

    @property
    def processed_file_names(self):
        return ["multi_isotope_graph.pt"]

    def process(self):
        all_nodes = []
        all_targets = []
        all_uncertainties = []
        all_masks = []

        # To track node indices across files for inter-isotope edges
        # key: (v, J), value: list of (global_node_index, iso_name)
        quantum_map = {}
        current_idx = 0

        edge_indices = []

        for iso_name, config in ISOTOPOLOGUE_CONFIGS.items():
            path = os.path.join(self.raw_dir, rf"{iso_name}.csv")
            df = pd.read_csv(path)

            iso_start_idx = current_idx

            for i, row in df.iterrows():
                v, j = row["v"], row["J"]

                # 1. Features: [v, J, J*(J+1), E_calc, reduced_mass]
                node_feat = [v, j, j * (j + 1), row["ECalc"], config["mu"]]
                all_nodes.append(node_feat)

                # 2. Targets & Masks
                # We predict the residual. If Marvel is NaN, residual is 0 (masked anyway)
                is_marvel = not pd.isna(row["EMarv"])
                residual = row["EMarv"] - row["ECalc"] if is_marvel else 0.0
                unc = row["unc"] if is_marvel else 1.0  # 1.0 is dummy for masked nodes

                all_targets.append(residual)
                all_uncertainties.append(unc)
                all_masks.append(is_marvel)

                # 3. Build Quantum Map for Inter-Isotope Edges
                q_key = (v, j)
                if q_key not in quantum_map:
                    quantum_map[q_key] = []
                quantum_map[q_key].append(current_idx)

                current_idx += 1

            # 4. Intra-Isotope Edges (Selection Rules: ΔJ = ±1)
            # This is a simplification; for diatomic, we connect J to J+1 within same v
            for j_val in df["J"].unique():
                j_nodes = df[df["J"] == j_val].index + iso_start_idx
                jp1_nodes = df[df["J"] == j_val + 1].index + iso_start_idx
                for n1 in j_nodes:
                    for n2 in jp1_nodes:
                        edge_indices.append([n1, n2])
                        edge_indices.append([n2, n1])

        # 5. Inter-Isotope Edges (Isotopic Bridges)
        for q_key, indices in quantum_map.items():
            if len(indices) > 1:
                # Connect all molecules that share this (v, J) in a clique
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        edge_indices.append([indices[i], indices[j]])
                        edge_indices.append([indices[j], indices[i]])

        # Convert to Tensors
        x = torch.tensor(all_nodes, dtype=torch.float)
        y = torch.tensor(all_targets, dtype=torch.float).view(-1, 1)
        unc = torch.tensor(all_uncertainties, dtype=torch.float).view(-1, 1)
        mask = torch.tensor(all_masks, dtype=torch.bool)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

        data = Data(x=x, y=y, edge_index=edge_index)
        data.unc = unc
        data.train_mask = mask

        # Save processed data
        torch.save(self.collate([data]), self.processed_paths[0])
