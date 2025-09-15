# Simple configuration file for paths and settings
from pathlib import Path
import torch


# Base directory
BASE_DIR = Path(__file__).parent

# Model paths
MODEL_PATH = BASE_DIR / "baselines" / "nanogpt" / "shakespeare-char" / "models" / "baseline_model_2500.pt" # "baseline_nanogpt.pt"
META_PATH = BASE_DIR / "baselines" / "nanogpt" / "shakespeare-char" / "models" / "meta.pkl"
CONFIG_PATH = BASE_DIR / "baselines" / "nanogpt" / "shakespeare-char" / "models" / "config.yaml"
DATASET_PATH = BASE_DIR / "baselines" / "nanogpt" / "dataset.txt"

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

START_PROMPT = "to be o"