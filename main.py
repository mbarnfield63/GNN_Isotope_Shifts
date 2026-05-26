import argparse
import pandas as pd
import os

from tabulate import tabulate

from src.dataset import prepare_molecule_graph
from src.plotting import plot_extrapolation_improvements, plot_standard_split_residuals
from src.train import run_loio_cross_validation, run_standard_split
from src.utils import setup_experiment, set_seed


def parse_args():
    """
    Parses command-line arguments for the execution of the pipeline.

    Returns:
        argparse.Namespace: Parsed arguments containing the path to the config file.
    """
    parser = argparse.ArgumentParser(description="Diatomic GNN Isotope Shifts Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the YAML configuration file",
    )
    return parser.parse_args()


def execute_ensemble(
    config, pyg_graph, nodes_df, feature_cols, paths, logger, run_func, group_col
):
    """
    Executes the designated training function multiple times across different
    random seeds and aggregates the results to ensure statistical robustness.

    Args:
        config (dict): Configuration dictionary.
        pyg_graph (torch_geometric.data.Data): Base unscaled PyG data object.
        nodes_df (pd.DataFrame): Base nodes dataframe.
        feature_cols (list): List of feature column names.
        paths (dict): Dictionary of output directory paths.
        logger (logging.Logger): Logger instance.
        run_func (callable): The specific training function to execute (LOIO or Standard).
        group_col (str): The dataframe column name used to group results for aggregation.

    Returns:
        pd.DataFrame: Aggregated results showing the mean and standard deviation across all seeds.
    """
    num_seeds = config["execution"]["num_seeds"]
    base_seed = config["execution"]["base_seed"]

    all_results = []

    # 1. Iterate through seeds
    for i in range(num_seeds):
        current_seed = base_seed + i
        logger.info(
            f"\n{'*'*60}\nSTARTING ENSEMBLE RUN {i+1}/{num_seeds} (SEED: {current_seed})\n{'*'*60}"
        )

        # Lock the seed for this specific run
        set_seed(current_seed, logger)

        # Execute the pipeline for this seed
        # (For standard split ensembles, the return signature unpacking may require modification based on train.py)
        results_df = run_func(
            config, pyg_graph, nodes_df, feature_cols, paths["root"], logger
        )

        # (If run_func returns a tuple containing the dataframe and arrays, extract the dataframe here)
        if isinstance(results_df, tuple):
            results_df = results_df[0]

        results_df["Seed"] = current_seed
        all_results.append(results_df)

    # 2. Aggregate Results
    combined_df = pd.concat(all_results)

    summary_df = (
        combined_df.groupby(group_col)
        .agg(
            {
                "Original MAE (cm-1)": "mean",
                "Extrapolated MAE (cm-1)": ["mean", "std"],
                "Improvement (%)": ["mean", "std"],
            }
        )
        .reset_index()
    )

    # 3. Flatten the multi-level column names created by pandas aggregation
    summary_df.columns = [
        group_col,
        "Original MAE",
        "Extrapolated MAE Mean",
        "Extrapolated MAE Std",
        "Improvement (%)",
        "Improvement Std",
    ]

    # 4. Format strings for terminal output
    # (Combines mean and standard deviation into a single readable string)
    display_df = pd.DataFrame()
    display_df["Category"] = summary_df[group_col]
    display_df["Original MAE"] = summary_df["Original MAE"].map("{:.5f}".format)
    display_df["Extrapolated MAE"] = summary_df.apply(
        lambda row: f"{row['Extrapolated MAE Mean']:.5f} ± {row['Extrapolated MAE Std']:.5f}",
        axis=1,
    )
    display_df["Improvement (%)"] = summary_df.apply(
        lambda row: f"{row['Improvement (%)']:.2f} ± {row['Improvement Std']:.2f}",
        axis=1,
    )

    logger.info(f"\nFINAL ENSEMBLE RESULTS ({num_seeds} SEEDS)")
    logger.info(
        "\n" + tabulate(display_df, headers="keys", tablefmt="grid", stralign="center")
    )

    # Save the aggregated summary to CSV
    summary_df.to_csv(
        os.path.join(paths["root"], "ensemble_summary_results.csv"), index=False
    )

    return summary_df


def main():
    """
    Main orchestrator function. Reads configuration, initializes the dataset,
    triggers the appropriate training loops, and generates final plots.
    """
    # 1. Initialization and Configuration
    args = parse_args()
    config, paths, logger = setup_experiment(args.config)

    # 2. Data Preparation
    logger.info("Building Universal PyG Graph from configuration...")
    pyg_graph, nodes_df, feature_cols = prepare_molecule_graph(config, logger)

    # 3. Execution Routing
    mode = config["execution"].get("mode", "loio")
    is_ensemble = config["execution"].get("ensemble_run", False)

    if mode == "standard":
        group_column = "Run Type"
        if is_ensemble:
            logger.info("Mode: Standard Split (Ensemble Enabled)")
            final_results = execute_ensemble(
                config,
                pyg_graph,
                nodes_df,
                feature_cols,
                paths,
                logger,
                run_standard_split,
                group_column,
            )
        else:
            logger.info("Mode: Standard Split (Single Seed)")
            set_seed(config["execution"]["base_seed"], logger)

            # Execute single standard split run
            final_results, targets, preds, test_energies = run_standard_split(
                config, pyg_graph, nodes_df, feature_cols, paths["root"], logger
            )

            logger.info("\nFINAL RESULTS")
            logger.info(
                "\n"
                + tabulate(
                    final_results, headers="keys", tablefmt="grid", stralign="center"
                )
            )
            final_results.to_csv(
                os.path.join(paths["root"], "standard_summary_results.csv"), index=False
            )

            # Generate performance plots
            logger.info("Generating standard split performance plots...")
            ml_corrected_error = targets - preds
            plot_standard_split_residuals(
                targets, ml_corrected_error, test_energies, paths["root"]
            )

    elif mode == "loio":
        group_column = "Held-Out Isotopologue"
        if is_ensemble:
            logger.info("Mode: LOIO Cross-Validation (Ensemble Enabled)")
            final_results = execute_ensemble(
                config,
                pyg_graph,
                nodes_df,
                feature_cols,
                paths,
                logger,
                run_loio_cross_validation,
                group_column,
            )
        else:
            logger.info("Mode: LOIO Cross-Validation (Single Seed)")
            set_seed(config["execution"]["base_seed"], logger)
            final_results = run_loio_cross_validation(
                config, pyg_graph, nodes_df, feature_cols, paths["root"], logger
            )

            logger.info("\nFINAL RESULTS")
            logger.info(
                "\n"
                + tabulate(
                    final_results, headers="keys", tablefmt="grid", stralign="center"
                )
            )
            final_results.to_csv(
                os.path.join(paths["root"], "loio_summary_results.csv"), index=False
            )

        # Generate performance plots
        logger.info("Generating extrapolation performance plots...")
        plot_extrapolation_improvements(final_results, paths["root"])

    else:
        logger.error(f"Unknown execution mode: {mode}")

    logger.info(f"Pipeline complete. All artifacts saved to: {paths['root']}")


if __name__ == "__main__":
    main()
