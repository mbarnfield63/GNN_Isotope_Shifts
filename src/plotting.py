import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns


def plot_standard_split_residuals(
    true_residuals: np.ndarray,
    ml_residuals: np.ndarray,
    energies: np.ndarray,
    output_dir: str,
) -> None:
    """
    Generates a two-panel plot for standard random splits. The left panel displays
    an overlaid histogram of the residual distributions, and the right panel plots
    the residuals as a function of the state energy.

    Args:
        true_residuals (np.ndarray): Original Obs-Calc residuals (cm^-1).
        ml_residuals (np.ndarray): Machine learning corrected residuals (cm^-1).
        energies (np.ndarray): Calculated energies of the evaluated states (cm^-1).
        output_dir (str): Root directory for saving the plot.

    Returns:
        None
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Left Panel: Residual Histogram
    # (Determine universal binning boundaries to ensure both distributions share the exact same scale)
    min_val = min(np.min(true_residuals), np.min(ml_residuals))
    max_val = max(np.max(true_residuals), np.max(ml_residuals))
    bins = np.linspace(min_val, max_val, 50)

    axes[0].hist(
        true_residuals,
        bins=bins,
        alpha=0.5,
        color="purple",
        label="Original Obs-Calc",
        edgecolor="black",
    )
    axes[0].hist(
        ml_residuals,
        bins=bins,
        alpha=0.7,
        color="green",
        label="ML Corrected",
        edgecolor="black",
    )

    axes[0].axvline(0, color="black", linestyle="dashed", linewidth=1)
    axes[0].set_title("Residual Error Distribution", fontsize=14)
    axes[0].set_xlabel("Error (cm$^{-1}$)", fontsize=12)
    axes[0].set_ylabel("State Count", fontsize=12)
    axes[0].legend(loc="upper right")

    # 2. Right Panel: Residuals vs. Energy Scatter
    axes[1].scatter(
        energies,
        true_residuals,
        alpha=0.5,
        color="purple",
        s=15,
        label="Original Obs-Calc",
        edgecolor="none",
    )
    axes[1].scatter(
        energies,
        ml_residuals,
        alpha=0.7,
        color="green",
        s=15,
        label="ML Corrected",
        edgecolor="none",
    )

    axes[1].axhline(0, color="black", linestyle="dashed", linewidth=1)
    axes[1].set_title("Residuals vs. Energy", fontsize=14)
    axes[1].set_xlabel("Energy (cm$^{-1}$)", fontsize=12)
    axes[1].set_ylabel("Residual (cm$^{-1}$)", fontsize=12)
    axes[1].legend(loc="upper right")

    # 3. Synchronize scatter y-axis limits symmetrically
    max_abs_res = max(np.max(np.abs(true_residuals)), np.max(np.abs(ml_residuals)))
    axes[1].set_ylim(-max_abs_res * 1.1, max_abs_res * 1.1)

    # 4. Save and close
    save_path = os.path.join(output_dir, "plots", "standard_split_residuals.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_extrapolation_improvements(results_df: pd.DataFrame, output_dir: str) -> None:
    """
    Generates a bar chart visualizing the percentage improvement of the
    machine learning extrapolation compared to the original theoretical MAE.

    Args:
        results_df (pd.DataFrame): Dataframe containing 'Held-Out Isotopologue'
                                   and 'Improvement (%)' columns.
        output_dir (str): Root directory for saving the plot.

    Returns:
        None
    """
    # Set plot aesthetics
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # 1. Create the bar plot
    ax = sns.barplot(
        x="Held-Out Isotopologue",
        y="Improvement (%)",
        data=results_df,
        palette="viridis",
    )

    # 2. Add horizontal zero line for reference
    plt.axhline(0, color="black", linewidth=1.2, linestyle="--")

    # 3. Format titles and labels
    plt.title("Physics-Informed GNN Extrapolation Improvement", fontsize=14, pad=15)
    plt.ylabel("Improvement over Base Hamiltonian (%)", fontsize=12)
    plt.xlabel("Held-Out Isotopologue", fontsize=12)

    # 4. Annotate bars with exact percentages
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{height:.1f}%",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom" if height > 0 else "top",
            xytext=(0, 5 if height > 0 else -15),
            textcoords="offset points",
            fontsize=10,
        )

    # 5. Save and close
    # (Plots are routed specifically to the 'plots' subdirectory)
    save_path = os.path.join(output_dir, "plots", "extrapolation_improvements.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_energy_dependent_residuals(
    true_residuals: np.ndarray,
    ml_residuals: np.ndarray,
    energies: np.ndarray,
    iso_name: str,
    output_dir: str,
) -> None:
    """
    Generates a two-panel plot comparing the original Hamiltonian residuals
    against the ML-corrected residuals as a function of state energy.

    Args:
        true_residuals (np.ndarray): Original Obs-Calc residuals (cm^-1).
        ml_residuals (np.ndarray): Machine learning corrected residuals (cm^-1).
        energies (np.ndarray): Calculated energies of the states (cm^-1).
        iso_name (str): Identifier of the isotopologue being plotted.
        output_dir (str): Root directory for saving the plot.

    Returns:
        None
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # 1. Plot Original Residuals
    axes[0].scatter(
        energies,
        true_residuals,
        alpha=0.6,
        color="purple",
        s=15,
        edgecolor="k",
        linewidth=0.2,
    )
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title(f"Original Obs-Calc: {iso_name}", fontsize=14)
    axes[0].set_xlabel("Energy (cm$^{-1}$)", fontsize=12)
    axes[0].set_ylabel("Residual (cm$^{-1}$)", fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # 2. Plot ML Corrected Residuals
    axes[1].scatter(
        energies,
        ml_residuals,
        alpha=0.6,
        color="green",
        s=15,
        edgecolor="k",
        linewidth=0.2,
    )
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title(f"ML Corrected: {iso_name}", fontsize=14)
    axes[1].set_xlabel("Energy (cm$^{-1}$)", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # 3. Synchronize axis limits for direct visual comparison
    max_abs_res = max(np.max(np.abs(true_residuals)), np.max(np.abs(ml_residuals)))
    axes[0].set_ylim(-max_abs_res * 1.1, max_abs_res * 1.1)

    # 4. Save and close
    save_path = os.path.join(output_dir, "plots", f"energy_residuals_{iso_name}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_residual_distribution(
    true_residuals: np.ndarray, ml_residuals: np.ndarray, iso_name: str, output_dir: str
) -> None:
    """
    Generates overlaid histograms comparing the error distribution of the
    base Hamiltonian against the ML-corrected predictions.

    Args:
        true_residuals (np.ndarray): Original Obs-Calc residuals (cm^-1).
        ml_residuals (np.ndarray): Machine learning corrected residuals (cm^-1).
        iso_name (str): Identifier of the isotopologue being plotted.
        output_dir (str): Root directory for saving the plot.

    Returns:
        None
    """
    plt.figure(figsize=(8, 6))

    # 1. Determine universal binning boundaries
    # (Ensures both histograms share the exact same scale)
    min_val = min(np.min(true_residuals), np.min(ml_residuals))
    max_val = max(np.max(true_residuals), np.max(ml_residuals))
    bins = np.linspace(min_val, max_val, 50)

    # 2. Plot distributions
    plt.hist(
        true_residuals,
        bins=bins,
        alpha=0.5,
        color="purple",
        label="Original Obs-Calc",
        edgecolor="black",
    )
    plt.hist(
        ml_residuals,
        bins=bins,
        alpha=0.7,
        color="green",
        label="ML Corrected",
        edgecolor="black",
    )

    # 3. Format titles and labels
    plt.axvline(0, color="black", linestyle="dashed", linewidth=1)
    plt.title(f"Residual Error Distribution: {iso_name}", fontsize=14)
    plt.xlabel("Error (cm$^{-1}$)", fontsize=12)
    plt.ylabel("State Count", fontsize=12)
    plt.legend(loc="upper right")

    # 4. Save and close
    save_path = os.path.join(output_dir, "plots", f"residual_histogram_{iso_name}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
