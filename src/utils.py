import os
import datetime
import logging
import yaml
import torch
import numpy as np
import random


def setup_experiment(config_path="configs/default.yaml"):
    """
    Parses the config, creates timestamped output directories, and sets up logging.
    """
    # 1. Load configuration
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # 2. Extract base settings
    exp_name = config["experiment"]["name"]
    base_dir = config["experiment"]["output_base_dir"]

    # 3. Create a unique timestamped directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"{exp_name}_{timestamp}")

    # 4. Create necessary subdirectories
    paths = {
        "root": run_dir,
        "checkpoints": os.path.join(run_dir, "checkpoints"),
        "plots": os.path.join(run_dir, "plots"),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    # 5. Save a backup of the config to the output dir for reproducibility
    with open(os.path.join(run_dir, "config.yaml"), "w") as file:
        yaml.dump(config, file, default_flow_style=False)

    # 6. Setup Logging
    log_file = os.path.join(run_dir, "experiment.log")

    # Remove existing handlers if the logger is re-initialized
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Experiment '{exp_name}' initialized.")
    logger.info(f"Outputs will be saved to: {run_dir}")

    return config, paths, logger


def set_seed(seed: int, logger=None):
    """
    Locks all random seeds for reproducibility.
    Absolutely critical for the 5-seed ensemble tests.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Force deterministic operations (slight performance hit, but guarantees reproducibility)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if logger:
        logger.info(f"Random seed locked to: {seed}")
