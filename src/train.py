import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.model import HybridIsotopologueGATv2


def apply_dynamic_scaling(pyg_graph, nodes_df, config, feature_cols, train_mask):
    """
    Applies StandardScaler strictly to the training pool and ONLY to the
    non-physics features. Physics mass-ratios are preserved exactly.
    """
    scaler = StandardScaler()

    # 1. Isolate the columns that should be scaled
    physics_cols = config["data"]["physics_features"]
    scale_cols = [col for col in feature_cols if col not in physics_cols]

    # 2. Extract training indices
    train_indices = torch.where(train_mask)[0].numpy()

    # 3. Create a safe copy of the dataframe
    scaled_df = nodes_df.copy()

    # 4. Fit scaler ONLY on the training subset for the safe columns
    scaler.fit(scaled_df.loc[train_indices, scale_cols])

    # 5. Transform the target columns (leaving physics columns completely untouched)
    scaled_df[scale_cols] = scaler.transform(scaled_df[scale_cols])

    return torch.tensor(scaled_df[feature_cols].values, dtype=torch.float32)


def compute_sample_weights(nodes_df, train_mask, device):
    """
    Calculates inverse frequency weights for training samples.
    """
    iso_tensor = torch.tensor(nodes_df["iso_id"].values, dtype=torch.long)
    train_iso_ids = iso_tensor[train_mask]

    unique_train_isos, iso_counts = torch.unique(train_iso_ids, return_counts=True)
    total_train_samples = len(train_iso_ids)

    class_weights = {
        iso.item(): total_train_samples / (len(unique_train_isos) * count.item())
        for iso, count in zip(unique_train_isos, iso_counts)
    }

    sample_weights = torch.tensor(
        [class_weights.get(iso.item(), 1.0) for iso in iso_tensor],
        dtype=torch.float32,
        device=device,
    )

    return sample_weights


def setup_optimizer(model, config):
    """
    Configures Adam optimizer. Safely exempts the Physics Bypass from weight decay.
    """
    lr = config["training"]["learning_rate"]
    wd_gnn = config["training"]["weight_decay_gnn"]
    wd_physics = config["training"]["weight_decay_physics"]

    optimizer = optim.Adam(
        [
            {
                "params": model.edge_emb.parameters(),
                "weight_decay": wd_gnn,
            },  # [FIXED] Edge embeddings now train
            {"params": model.conv1.parameters(), "weight_decay": wd_gnn},
            {"params": model.conv2.parameters(), "weight_decay": wd_gnn},
            {"params": model.head_topology.parameters(), "weight_decay": wd_gnn},
            {"params": model.head_physics.parameters(), "weight_decay": wd_physics},
        ],
        lr=lr,
    )

    return optimizer


def train_single_fold(
    pyg_graph, nodes_df, train_mask, val_mask, test_mask, config, fold_dir, logger
):
    """
    Executes the training loop for a single cross-validation fold.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pyg_graph = pyg_graph.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    num_node_features = pyg_graph.x.shape[1]
    model = HybridIsotopologueGATv2(config, num_node_features).to(device)
    optimizer = setup_optimizer(model, config)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=100
    )

    # Multiply targets to scale them UP to GNN-friendly range
    scale_factor = config["training"]["scale_factor"]
    scaled_targets = pyg_graph.y * scale_factor

    sample_weights = compute_sample_weights(nodes_df, train_mask.cpu(), device)

    epochs = config["training"]["epochs"]
    patience_limit = config["training"]["patience"]

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        preds = model(pyg_graph.x, pyg_graph.edge_index, pyg_graph.edge_attr)

        raw_errors = torch.abs(preds[train_mask] - scaled_targets[train_mask])
        # loss = torch.mean(raw_errors * sample_weights[train_mask])
        loss = torch.mean(raw_errors)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(pyg_graph.x, pyg_graph.edge_index, pyg_graph.edge_attr)
            val_loss = torch.mean(
                torch.abs(val_preds[val_mask] - scaled_targets[val_mask])
            ).item()
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

        if patience_counter >= patience_limit:
            logger.info(f"    Early stopping triggered at epoch {epoch}")
            break

    # Testing Phase
    model.load_state_dict(best_model_state)
    torch.save(best_model_state, os.path.join(fold_dir, "best_model.pt"))

    model.eval()
    with torch.no_grad():
        final_preds = model(pyg_graph.x, pyg_graph.edge_index, pyg_graph.edge_attr)

        # Divide predictions to scale them DOWN back to cm^-1
        unscaled_preds = final_preds[test_mask] / scale_factor
        unscaled_targets = pyg_graph.y[test_mask]

        test_errors = torch.abs(unscaled_preds - unscaled_targets)
        test_mae = test_errors.mean().item()

    return test_mae, unscaled_preds.cpu().numpy(), unscaled_targets.cpu().numpy()


def run_loio_cross_validation(
    config, pyg_graph, nodes_df, feature_cols, output_dir, logger
):
    logger.info("Initializing LOIO Cross-Validation...")

    parent_isotopes = [
        str(mol_cfg["parent_isotope"]) for mol_cfg in config["molecules"].values()
    ]
    all_iso_ids = nodes_df["iso_id"].astype(str).unique()
    minor_isos = [iso for iso in all_iso_ids if iso not in parent_isotopes]

    loio_results = []

    iso_tensor_str = torch.tensor(nodes_df["iso_id"].astype(str).values)
    has_marvel = torch.tensor(nodes_df["is_known_marvel"].values, dtype=torch.bool)

    for unseen_iso in minor_isos:
        logger.info(
            f"\n{'='*50}\nEXTRAPOLATING TO UNSEEN ISOTOPOLOGUE: {unseen_iso}\n{'='*50}"
        )

        fold_dir = os.path.join(output_dir, "checkpoints", f"fold_iso_{unseen_iso}")
        os.makedirs(fold_dir, exist_ok=True)

        test_mask = (iso_tensor_str == unseen_iso) & has_marvel
        train_val_pool = torch.where((iso_tensor_str != unseen_iso) & has_marvel)[0]

        train_val_pool = train_val_pool[torch.randperm(len(train_val_pool))]
        val_size = max(1, int(0.1 * len(train_val_pool)))

        train_mask = torch.zeros(len(nodes_df), dtype=torch.bool)
        val_mask = torch.zeros(len(nodes_df), dtype=torch.bool)

        val_mask[train_val_pool[:val_size]] = True
        train_mask[train_val_pool[val_size:]] = True

        # Passed config to scaling function so it can identify physics features
        scaled_x = apply_dynamic_scaling(
            pyg_graph, nodes_df, config, feature_cols, train_mask
        )

        fold_graph = pyg_graph.clone()
        fold_graph.x = scaled_x

        test_mae, preds, targets = train_single_fold(
            fold_graph,
            nodes_df,
            train_mask,
            val_mask,
            test_mask,
            config,
            fold_dir,
            logger,
        )

        orig_mae = torch.abs(torch.tensor(targets)).mean().item()
        improvement = ((orig_mae - test_mae) / orig_mae) * 100

        loio_results.append(
            {
                "Held-Out Isotopologue": unseen_iso,
                "Original MAE (cm-1)": orig_mae,
                "Extrapolated MAE (cm-1)": test_mae,
                "Improvement (%)": improvement,
            }
        )

        logger.info(
            f"Result for {unseen_iso} | Original: {orig_mae:.5f} | Extrapolated: {test_mae:.5f} | Improv: {improvement:.2f}%"
        )

    return pd.DataFrame(loio_results)


def run_standard_split(config, pyg_graph, nodes_df, feature_cols, output_dir, logger):
    logger.info("Initializing Standard Random Split...")

    has_marvel = torch.tensor(nodes_df["is_known_marvel"].values, dtype=torch.bool)
    valid_indices = torch.where(has_marvel)[0]

    shuffled_indices = valid_indices[torch.randperm(len(valid_indices))]

    total_valid = len(shuffled_indices)
    train_size = int(total_valid * config["training"]["train_split"])
    val_size = int(total_valid * config["training"]["val_split"])

    train_mask = torch.zeros(len(nodes_df), dtype=torch.bool)
    val_mask = torch.zeros(len(nodes_df), dtype=torch.bool)
    test_mask = torch.zeros(len(nodes_df), dtype=torch.bool)

    train_mask[shuffled_indices[:train_size]] = True
    val_mask[shuffled_indices[train_size : train_size + val_size]] = True
    test_mask[shuffled_indices[train_size + val_size :]] = True

    scaled_x = apply_dynamic_scaling(
        pyg_graph, nodes_df, config, feature_cols, train_mask
    )

    fold_graph = pyg_graph.clone()
    fold_graph.x = scaled_x

    fold_dir = os.path.join(output_dir, "checkpoints", "standard_split")
    os.makedirs(fold_dir, exist_ok=True)

    test_mae, preds, targets = train_single_fold(
        fold_graph, nodes_df, train_mask, val_mask, test_mask, config, fold_dir, logger
    )

    # 1. Calculate Errors and Noise Floor Masks
    ml_absolute_errors = np.abs(targets - preds)
    original_absolute_errors = np.abs(targets)

    # (Extract uncertainties for the test set and fill missing with 0.0)
    test_unc = nodes_df["unc"].fillna(0.0).values[test_mask.numpy()]

    improved_levels_mask = ml_absolute_errors < original_absolute_errors
    degraded_levels_mask = ml_absolute_errors > original_absolute_errors
    within_noise_floor_mask = ml_absolute_errors <= test_unc
    harmless_degradation_mask = degraded_levels_mask & within_noise_floor_mask

    orig_mae = original_absolute_errors.mean()
    improvement = ((orig_mae - test_mae) / orig_mae) * 100

    overall_pct_improved = improved_levels_mask.mean() * 100
    overall_pct_harmless = harmless_degradation_mask.mean() * 100
    true_success_rate = overall_pct_improved + overall_pct_harmless

    # 2. Log the detailed performance metrics
    logger.info(f"\nOriginal Baseline MAE: {orig_mae:.5f} cm^-1")
    logger.info(f"ML Corrected MAE:      {test_mae:.5f} cm^-1")
    logger.info(f"Overall Improvement:   {improvement:.2f}%\n")
    logger.info(f"Levels Strictly Improved:            {overall_pct_improved:.2f}%")
    logger.info(f"Harmless Degradation (Inside Noise): {overall_pct_harmless:.2f}%")
    logger.info(f"Effective Success Rate:              {true_success_rate:.2f}%\n")

    # 3. Record Metrics to DataFrame
    result_df = pd.DataFrame(
        [
            {
                "Run Type": "Standard Random Split",
                "Original MAE (cm-1)": orig_mae,
                "Extrapolated MAE (cm-1)": test_mae,
                "Improvement (%)": improvement,
            }
        ]
    )

    test_energies = nodes_df["ECalc"].values[test_mask.numpy()]

    return result_df, targets, preds, test_energies
