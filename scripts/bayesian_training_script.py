"""
Bayesian NanoGPT Training Script

Supported samplers:
- vi: Variational Inference (Diagonal Gaussian)
- ekf: Extended Kalman Filter
- laplace: Laplace Approximation
- sgld: Stochastic Gradient Langevin Dynamics (MCMC)
- sghmc: Stochastic Gradient Hamiltonian Monte Carlo (MCMC)
- baoa: Bayesian Adaptive Optimization Algorithm (MCMC)

Examples:
    python scripts/bayesian_training_script.py --sampler vi --epochs 15
    python scripts/bayesian_training_script.py --sampler sgld --eval
    python scripts/bayesian_training_script.py --sampler sghmc --eval
"""
import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import torch
import datetime
import json

# Ensure project root and relevant subdirs are on sys.path BEFORE imports
current_dir = Path.cwd()
project_root = current_dir if (current_dir / 'src').exists() else current_dir.parent
paths_to_add = [
    project_root,
    project_root / 'src',
    project_root / 'baselines',
]
for p in paths_to_add:
    sp = str(p)
    if sp not in sys.path:
        sys.path.append(sp)

# Module imports (now safe)
from src.nanogpt_utils import load_model, load_tokenizer
from src.bayesian_utils import create_training_batches, run_bayesian_pipeline
from config import CONFIG, MODEL_PATH, META_PATH, DATA_DIR, CONFIG_EKF, CONFIG_SGLD, CONFIG_SGHMC, CONFIG_BAOA
from src.evaluation.bayesian_evaluator import BayesianNanoGPTEvaluator, evaluate_bayesian_splits

"""Bayesian NanoGPT training script with optional external evaluation.

Usage examples:
    # Variational Inference
    python scripts/bayesian_training_script.py --sampler vi --epochs 15 --eval --eval-splits val train

    # Extended Kalman Filter
    python scripts/bayesian_training_script.py --sampler ekf --epochs 15 --eval

    # SGMCMC Samplers (with automatic warmup and sampling schedule)
    python scripts/bayesian_training_script.py --sampler sgld --eval
    python scripts/bayesian_training_script.py --sampler sghmc --eval
    python scripts/bayesian_training_script.py --sampler baoa --eval

    # Note: For SGMCMC samplers (sgld, sghmc, baoa), the script will automatically:
    #   - Run 200 warmup steps
    #   - Run 1000 sampling steps
    #   - Collect every 10th sample (~100 samples total)
    #   - Stop after completing the schedule (ignoring --epochs if provided)
"""


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
    parser.add_argument(
        '--sampler',
        type=str,
        default='vi',
        choices=['vi', 'ekf', 'laplace', 'sgld', 'sghmc', 'baoa'],
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

    # Evaluation options
    parser.add_argument(
        '--eval',
        action='store_true',
        help='Run Bayesian evaluation after training (samples from posterior for generation)'
    )
    parser.add_argument(
        '--eval-splits',
        nargs='*',
        default=['val','train'],
        help='Dataset splits to evaluate'
    )
    parser.add_argument(
        '--eval-num-posterior-samples',
        type=int,
        default=10,
        help='Number of posterior samples to use for Bayesian evaluation'
    )
    parser.add_argument(
        '--eval-max-samples',
        type=int,
        default=200,
        help='Max samples (controls perplexity batch iterations)'
    )
    parser.add_argument(
        '--eval-num-text-samples',
        type=int,
        default=20,
        help='Number of text samples for BLEU/ROUGE'
    )
    parser.add_argument(
        '--eval-prompt-length',
        type=int,
        default=20,
        help='Prompt length for generation'
    )
    parser.add_argument(
        '--eval-generation-length',
        type=int,
        default=30,
        help='Generation length'
    )
    parser.add_argument(
        '--eval-max-tokens',
        type=int,
        default=None,
        help='Optional cap on tokens loaded from each split (fast debug)'
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

    # Note for MCMC samplers
    if args.sampler in ['sgld', 'sghmc', 'baoa']:
        logger.info("\nMCMC Sampler Configuration:")
        logger.info("  - This sampler uses warmup and sampling schedule")
        logger.info("  - Training will automatically stop after completing the schedule")
        logger.info("  - Check config.py for warmup_steps, sampling_steps, and thinning settings")

    logger.info("="*70)
    
    # Set seed
    set_seed(args.seed)
    logger.info(f"Random seed set to: {args.seed}")
    
    # Select appropriate config based on sampler type
    if args.sampler == 'vi':
        config = CONFIG.copy()
    elif args.sampler == 'ekf':
        config = CONFIG_EKF.copy()
    elif args.sampler == 'sgld':
        config = CONFIG_SGLD.copy()
    elif args.sampler == 'sghmc':
        config = CONFIG_SGHMC.copy()
    elif args.sampler == 'baoa':
        config = CONFIG_BAOA.copy()
    elif args.sampler == 'laplace':
        config = CONFIG.copy()  # Use base config for Laplace
    else:
        raise ValueError(f"Unknown sampler type: {args.sampler}")

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
        logger.info("Model loaded successfully!")
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
        
        state, metrics, eval_results, collected_samples = run_bayesian_pipeline(
            training_batches,
            sampler_type=args.sampler,
            config=config,
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
            logger.info("\nEvaluation Results:")
            logger.info(f"  Deterministic Loss: {eval_results['deterministic']['loss']:.4f}")
            logger.info(f"  Bayesian Loss: {eval_results['posterior_mean']['loss']:.4f}")
            logger.info(f"  Improvement: {eval_results['posterior_mean']['improvement_over_deterministic']:+.4f}")
            logger.info(f"  Better than deterministic: {eval_results['posterior_mean']['better_than_deterministic']}")

        if collected_samples:
            logger.info(f"\nCollected {len(collected_samples)} SGMCMC samples for generation")

        # Optional external evaluation using BayesianNanoGPTEvaluator
        if args.eval:
            logger.info("\n" + "="*70)
            logger.info("Running Bayesian evaluation with posterior sampling...")
            logger.info("="*70)
            try:
                eval_cfg = {
                    'data_dir': str(DATA_DIR),
                    'splits': args.eval_splits,
                    'batch_size': config['batch_size'],
                    'max_eval_samples': args.eval_max_samples,
                    'num_text_samples': args.eval_num_text_samples,
                    'prompt_length': args.eval_prompt_length,
                    'generation_length': args.eval_generation_length,
                    'max_tokens': args.eval_max_tokens,
                }
                
                # Create Bayesian evaluator with appropriate sampler state
                bayesian_evaluator = BayesianNanoGPTEvaluator(
                    model=model,
                    stoi=stoi,
                    itos=itos,
                    sampler_type=args.sampler,
                    state=state if args.sampler in ['vi', 'ekf', 'laplace'] else None,
                    collected_samples=collected_samples if args.sampler in ['sgld', 'sghmc', 'baoa'] else None,
                    device='auto',
                    num_posterior_samples=10,
                )
                
                split_results = evaluate_bayesian_splits(bayesian_evaluator, eval_cfg)
                
                logger.info("\nBayesian Evaluation Results:")
                for split, res in split_results.items():
                    if 'error' in res:
                        logger.error(f"[{split}] Evaluation error: {res['error']}")
                    else:
                        logger.info(f"[{split}] sampler={res.get('sampler_type')} posterior_samples={res.get('num_posterior_samples')}")
                        logger.info(f"        tokens={res.get('total_tokens')} ppl={res.get('perplexity'):.4f} bleu={res.get('bleu',0.0):.4f} rouge1={res.get('rouge1',0.0):.4f}")
                
                # Persist evaluation JSON
                out_dir = Path('checkpoints') / 'samplers' / f"{args.sampler}_sampler"
                out_dir.mkdir(parents=True, exist_ok=True)
                ts_eval = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                out_file = out_dir / f"bayesian_eval_{ts_eval}.json"
                payload = {
                    'config': eval_cfg,
                    'sampler_type': args.sampler,
                    'num_posterior_samples': 10,
                    'results': split_results,
                    'vocab_size': bayesian_evaluator.vocab_size,
                }
                with open(out_file, 'w') as f:
                    json.dump(payload, f, indent=2)
                logger.info(f"Bayesian evaluation saved to: {out_file}")
            except Exception:
                logger.exception("Bayesian evaluation failed")
        
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
