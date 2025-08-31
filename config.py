# Simple configuration file for paths and settings
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Model paths
MODEL_PATH = BASE_DIR / "checkpoints" / "baseline_nanogpt" / "baseline_nanogpt.pt"
META_PATH = BASE_DIR / "checkpoints" / "baseline_nanogpt" / "nanogpt_meta.pkl"
CONFIG_PATH = BASE_DIR / "baselines" / "nanogpt" / "shakespeare-char" / "config.yaml"
DATASET_PATH = BASE_DIR / "baselines" / "nanogpt" / "dataset.txt"

# Device
DEVICE = "cuda" if __name__ == "__main__" else "cpu"
