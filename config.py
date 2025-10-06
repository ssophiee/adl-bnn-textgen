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

# Model configuration
# CONFIG = {
#     'batch_size': 12, # was 8
#     'num_epochs': 3, # was 4
#     'learning_rate': 1e-5,
#     'max_new_tokens': 100,
#     'generation_temperature': 0.8,
#     'num_samples': 5,
#     'max_seq_length': 128,
#     'train_samples': 200,
#     'save_dir': "checkpoints/samplers",
#     'wandb_project': 'bayesian-nanogpt',  # W&B project name
# }
# CONFIG['temperature'] = 1 / CONFIG['train_samples']

# Conservative VI configuration
CONFIG = {
    'batch_size': 16,  # Larger batches for more stable gradients
    'num_epochs': 3, #5,   # More epochs for convergence
    'learning_rate': 5e-6,  # Lower LR for fine-tuning pre-trained model
    'max_new_tokens': 100,
    'generation_temperature': 0.8,
    'num_samples': 10,  # More samples for better uncertainty estimates
    'max_seq_length': 128,
    'train_samples': 500,  # More training data
    'save_dir': 'checkpoints/samplers/vi_conservative',
    'wandb_project': 'bayesian-nanogpt',
    
    # VI-specific: CRITICAL for stability
    'temperature': 0.001,  # Much lower! Reduces noise in VI
    'vi_n_samples': 1,     # Keep at 1 for stability
    'prior_std': 0.1,      # Tighter prior around pre-trained weights
}


