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

CONFIG = {
    # === Data Configuration ===
    'batch_size': 16,  
    'train_samples': 10000, 
    'max_seq_length': 128,
    
    # === Training Configuration ===
    'num_epochs': 15,  
    'learning_rate': 0.0001,
    
    # === Variational Inference Configuration ===
    'temperature': 1,  
    'vi_n_samples': 1, 
    
    # === Prior Configuration ===
    # TIGHTER PRIOR - This is key for preventing variance explosion
    'prior_std': 1, 
    'prior_strength': 10.0,  # TODO: check if this is needed
    
    # === Variance Initialization ===
    'init_log_scale': -5, 
    'max_log_scale': 2, 
    "prior_beta": 5e-4,  # Scale the influence of the prior in the posterior

    # === Evaluation Configuration ===
    'num_samples': 10,
    'max_new_tokens': 100,
    'generation_temperature': 0.8,
    
    # === Saving Configuration ===
    'save_dir': 'checkpoints/samplers',
    'wandb_project': 'bayesian-nanogpt',
    
    # === Regularization ===
    'weight_decay': 0.01,  # TODO: check where to use it: L2 regularization on mean
    'kl_weight': 1.0,  # TODO: check where to use it: Weight for KL term in ELBO
}

# === Alternative Configurations for Different Samplers ===

CONFIG_EKF = {
    **CONFIG,
    'learning_rate': 5e-6, 
    'ekf_damping': 0.1,  
}

CONFIG_LAPLACE = {
    **CONFIG,
    'learning_rate': 1e-5
}

CONFIG_SGMCMC = {
    **CONFIG,
    'learning_rate': 1e-7,  #
    'sghmc_alpha': 0.2,    
    'sghmc_beta': 0.0,      
    'temperature': 1.0,
}