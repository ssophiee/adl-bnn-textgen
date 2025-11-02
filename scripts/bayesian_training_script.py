"""
Bayesian NanoGPT Training Script
Run with: python train_bayesian_nanogpt.py --sampler vi --epochs 10
"""
import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import torch
import datetime

# Add paths for importing utilities and models
current_dir = Path.cwd()
sys.path.append(str(current_dir))
sys.path.append(str(current_dir.parent))
sys.path.append(str(current_dir.parent / "baselines"))

from src.nanogpt_utils import load_model, load_tokenizer
from src.bayesian_utils import create_training_batches, run_bayesian_pipeline
from config import CONFIG, MODEL_PATH, META_PATH, DATA_DIR


def setup_logging(log_dir: Path, sampler_type: str):
    """Setup logging to both file and console"""
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{sampler_type}_training_{ts}.log"
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train Bayesian NanoGPT with different inference methods'
    )
    # TODO: remove laplace option if not implemented
    parser.add_argument(
        '--sampler',
        type=str,
        default='vi',
        choices=['vi', 'ekf', 'laplace', 'sgmcmc'],
        help='Bayesian inference method to use'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (overrides config)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )
    
    parser.add_argument(
        '--train-samples',
        type=int,
        default=None,
        help='Number of training samples (overrides config)'
    )
    
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable Weights & Biases logging'
    )
    
    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs',
        help='Directory for log files'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    """Main training function"""
    args = parse_args()
    
    # Setup logging
    log_dir = Path(args.log_dir)
    logger, log_file = setup_logging(log_dir, args.sampler)
    
    logger.info("="*70)
    logger.info("BAYESIAN NANOGPT TRAINING")
    logger.info("="*70)
    logger.info(f"Sampler: {args.sampler.upper()}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Using W&B: {not args.no_wandb}")
    logger.info("="*70)
    
    # Set seed
    set_seed(args.seed)
    logger.info(f"Random seed set to: {args.seed}")
    
    # Override config if command line args provided
    config = CONFIG.copy()
    if args.epochs:
        config['num_epochs'] = args.epochs
        logger.info(f"Overriding epochs: {args.epochs}")
    if args.batch_size:
        config['batch_size'] = args.batch_size
        logger.info(f"Overriding batch_size: {args.batch_size}")
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
        logger.info(f"Overriding learning_rate: {args.learning_rate}")
    if args.train_samples:
        config['train_samples'] = args.train_samples
        logger.info(f"Overriding train_samples: {args.train_samples}")
    
    try:
        # Load model
        logger.info("\n" + "="*70)
        logger.info("Loading pre-trained model...")
        logger.info("="*70)
        
        model, checkpoint = load_model(Path(MODEL_PATH))
        stoi, itos = load_tokenizer(Path(META_PATH))
        vocab_size = len(itos)
        
        logger.info(f"Model loaded successfully!")
        logger.info(f"Vocabulary size: {vocab_size}")
        
        # Prepare training data
        logger.info("\n" + "="*70)
        logger.info("Preparing training data...")
        logger.info("="*70)
        
        train_data_path = Path(DATA_DIR / 'train.bin')
        data = np.memmap(str(train_data_path), dtype=np.uint16, mode='r')
        
        training_batches = create_training_batches(
            data,
            config['batch_size'],
            config['max_seq_length'],
            config['train_samples']
        )
        
        logger.info(f"Created {len(training_batches)} training batches")
        logger.info(f"Batch shape: {training_batches[0][0].shape}")
        logger.info(f"Target shape: {training_batches[0][1].shape}")
        logger.info(f"Total training samples: {config['train_samples']}")
        
        # Run Bayesian training pipeline
        logger.info("\n" + "="*70)
        logger.info("Starting Bayesian training pipeline...")
        logger.info("="*70)
        
        state, metrics, eval_results = run_bayesian_pipeline(
            training_batches,
            sampler_type=args.sampler,
            use_wandb=not args.no_wandb
        )
        
        # Print final results
        logger.info("\n" + "="*70)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*70)
        
        if metrics['training_losses']:
            logger.info(f"Final Training Loss: {metrics['training_losses'][-1]:.4f}")
            logger.info(f"Final Log Posterior: {metrics['log_posterior_values'][-1]:.4f}")
        
        if eval_results and 'posterior_mean' in eval_results:
            logger.info(f"\nEvaluation Results:")
            logger.info(f"  Deterministic Loss: {eval_results['deterministic']['loss']:.4f}")
            logger.info(f"  Bayesian Loss: {eval_results['posterior_mean']['loss']:.4f}")
            logger.info(f"  Improvement: {eval_results['posterior_mean']['improvement_over_deterministic']:+.4f}")
            logger.info(f"  Better than deterministic: {eval_results['posterior_mean']['better_than_deterministic']}")
        
        logger.info("="*70)
        logger.info(f"Log file saved to: {log_file}")
        logger.info("="*70)
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\n" + "="*70)
        logger.warning("Training interrupted by user (Ctrl+C)")
        logger.warning("="*70)
        return 130
        
    except Exception as e:
        logger.error("\n" + "="*70)
        logger.error("TRAINING FAILED!")
        logger.error("="*70)
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
