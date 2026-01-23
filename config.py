from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# Paths from environment
BASE_DIR = Path(os.getenv('BASE_DIR', Path.cwd()))
MODEL_PATH = Path(os.getenv('MODEL_PATH'))
META_PATH = Path(os.getenv('META_PATH'))
DATA_DIR = Path(os.getenv('DATA_DIR'))
DEVICE = os.getenv('DEVICE', 'cpu')
WANDB_AVAILABLE = os.getenv('WANDB_AVAILABLE', 'false').lower() == 'true'
BNN_MODEL_PATH = Path(os.getenv('BNN_MODEL_PATH'))

CONFIG = {
    # === Data Configuration ===
    'batch_size': 16,
    'train_samples': 10000,
    'max_seq_length': 128,

    # === Training Configuration ===
    'learning_rate': 0.0001,
    'temperature': 1,

    # === Prior Configuration ===
    'prior_std': 1,
    'prior_beta': 0.0001,
    'prior_center': 'pretrained',  # 'pretrained' (default) or 'zero' for standard Bayesian baseline

    # === Evaluation Configuration ===
    'num_samples': 10,
    'max_new_tokens': 100,
    'generation_temperature': 0.8,

    # === Saving Configuration ===
    'save_dir': 'checkpoints/samplers',
    'wandb_project': 'bayesian-nanogpt',
}

# === SGMCMC Sampler Configurations ===

CONFIG_SGLD = {
    **CONFIG,
    'learning_rate': 1e-6,
    'sgld_beta': 0.0,  # Gradient noise correction
    'temperature': 1.0,
    'prior_beta': 0.0001,  # Reduced prior influence

    # Warm-up and sampling schedule
    'warmup_steps': 200,
    'sampling_steps': 1000,
    'thinning': 10,  # Collect every 10th sample
}

CONFIG_SGHMC = {
    **CONFIG,
    'learning_rate': 1e-5,      
    'sghmc_alpha': 0.1,      
    'sghmc_beta': 0.0,          
    'sghmc_sigma': 1.0,         
    'temperature': 1.0,         

    # Warm-up and sampling schedule
    'warmup_steps': 200,
    'sampling_steps': 1000,
    'thinning': 10,  # Collect every 10th sample
}

CONFIG_BAOA = {
    **CONFIG,
    'learning_rate': 1e-6,
    'baoa_alpha': 0.01,  # Momentum decay
    'baoa_sigma': 1.0,  # Prior std for momenta
    'temperature': 1.0,
    'prior_beta': 0.0001,  # Reduced prior influence

    # Warm-up and sampling schedule
    'warmup_steps': 200,
    'sampling_steps': 1000,
    'thinning': 10,  # Collect every 10th sample
}