"""
Utility functions for Bayesian inference with deterministic model comparison.
"""
import torch
import wandb
import datetime
import numpy as np
from torch import func
import gc
import posteriors
import torch.nn.functional as F
from pathlib import Path
from torch.optim.lr_scheduler import CosineAnnealingLR
from config import DEVICE, CONFIG, MODEL_PATH, WANDB_AVAILABLE, CONFIG_SGLD, CONFIG_SGHMC, CONFIG_BAOA
# Note: This module supports SGMCMC samplers (SGLD, SGHMC, BAOA)
# These are step-based methods that collect samples during training
from src.nanogpt_utils import load_model
MODEL, checkpoint = load_model(Path(MODEL_PATH), DEVICE) # TODO: pass device
INITIAL_PARAMS = {k: v.clone().to(DEVICE) for k, v in MODEL.named_parameters()}
# Zero-centered prior mean (standard Bayesian approach)
ZERO_PARAMS = {k: torch.zeros_like(v) for k, v in MODEL.named_parameters()}

def create_training_batches(data, batch_size, seq_length, num_samples):
    """Create training batches from the data for next-token prediction"""
    batches = []
    max_start = len(data) - seq_length - 1
    
    # Sample random starting positions
    start_indices = np.random.choice(max_start, size=num_samples, replace=False)
    
    for i in range(0, len(start_indices), batch_size):
        batch_starts = start_indices[i:i+batch_size]
        x_batch = []
        y_batch = []
        
        for start in batch_starts:
            x_seq = data[start:start+seq_length].astype(np.int64)
            y_seq = data[start+seq_length:start+seq_length+1].astype(np.int64)
            x_batch.append(x_seq)
            y_batch.append(y_seq)
        
        x_tensor = torch.tensor(x_batch, dtype=torch.long, device=DEVICE)
        y_tensor = torch.tensor(y_batch, dtype=torch.long, device=DEVICE)

        batches.append((x_tensor, y_tensor))
    
    return batches


def single_batch_loss(params, batch):
    """Compute loss for a single batch using functional_call"""
    x, y = batch
    logits, _ = func.functional_call(MODEL, params, (x,))
    batch_size, seq_length, vocab_size = logits.shape
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = y.view(-1)
    loss = F.cross_entropy(logits_flat, targets_flat, reduction='mean')
    return loss

def log_posterior_fn(params, batch):
    """
    Compute log posterior with character-based model.

    Each training sample is a sequence that predicts 1 next character.

    Prior centering (CONFIG['prior_center']):
        - 'pretrained': Center at pretrained weights (default). Localizes sampling
          around the pretrained solution, similar to warm-start approaches.
        - 'zero': Standard zero-centered prior (traditional Bayesian baseline).
    """
    x, y = batch  # x: (batch_size, seq_length), y: (batch_size, 1)
    MODEL.eval()
    # Compute negative log-likelihood for the batch
    nll = single_batch_loss(params, batch)

    # Select prior center: 'pretrained' (default) or 'zero'
    prior_mean = ZERO_PARAMS if CONFIG.get('prior_center') == 'zero' else INITIAL_PARAMS

    # Compute log prior
    log_prior = posteriors.diag_normal_log_prob(
        params,
        mean=prior_mean,
        sd_diag=CONFIG.get('prior_std', 0.01)
    )

    # Total number of predictions in training set
    total_samples = CONFIG['train_samples']  # Total sequences = total predictions

    # Scale likelihood to full dataset
    # nll is mean loss per prediction in batch
    log_likelihood = -nll * total_samples

    beta = CONFIG.get('prior_beta', 1.0 / total_samples)
    log_posterior = log_likelihood + beta * log_prior

    return log_posterior, {}


def log_likelihood_fn(params, batch):
    """
    Compute log-likelihood for a single batch.
    """
    x, y = batch
    
    # CRITICAL: Set model to eval mode to disable dropout
    MODEL.eval()
    
    logits, aux_model = func.functional_call(MODEL, params, (x,))
    
    batch_size, seq_length, vocab_size = logits.shape
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = y.view(-1)
    
    loss_sum = F.cross_entropy(logits_flat, targets_flat, reduction='sum')
    log_likelihood = -loss_sum
    
    aux = {'logits': logits, 'model_aux': aux_model}
    return log_likelihood, aux

def evaluate_deterministic_model(model, test_batch):
    """
    Evaluate the deterministic (original) model on test data
    
    Args:
        model: The original deterministic model
        test_batch: Test batch (x, y)
    
    Returns:
        dict with loss and perplexity
    """
    x_test, y_test = test_batch
    model.eval()
    
    with torch.no_grad():
        logits, _ = model(x_test)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y_test.view(-1)
        )
        perplexity = torch.exp(loss)
    
    return {
        'loss': loss.item(),
        'perplexity': perplexity.item()
    }


class BayesianSamplerPipeline:
    """Unified pipeline for different Bayesian inference methods with W&B tracking"""

    AVAILABLE_SAMPLERS = ['sgld', 'sghmc', 'baoa']

    def __init__(self, sampler_type: str, config: dict, use_wandb: bool = True):
        """
        Initialize the Bayesian sampler pipeline

        Args:
            sampler_type: One of 'sgld', 'sghmc', 'baoa'
            config: Configuration dictionary with training parameters
            use_wandb: Whether to use Weights & Biases for tracking
        """
        if sampler_type.lower() not in self.AVAILABLE_SAMPLERS:
            raise ValueError(f"Sampler must be one of {self.AVAILABLE_SAMPLERS}, got {sampler_type}")
        
        self.sampler_type = sampler_type.lower()
        self.config = config
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_run = None
        
        self.metrics = {
            'training_losses': [],
            'log_posterior_values': [],
            'epoch_times': [],
            'sampler_specific_metrics': []
        }

        # For SGMCMC samplers: store collected samples
        self.collected_samples = []

        if self.use_wandb:
            self._init_wandb()
    
    def _init_wandb(self):
        """Initialize Weights & Biases"""
        try:
            self.wandb_run = wandb.init(
                project=self.config.get('wandb_project', 'bayesian-nanogpt'),
                name=f"{self.sampler_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    'sampler_type': self.sampler_type,
                    **self.config
                },
                tags=[self.sampler_type, 'bayesian_inference'],
                reinit=True
            )
            print(f"W&B initialized: {self.wandb_run.url}")
        except Exception as e:
            print(f"Warning: Could not initialize W&B: {e}")
            self.use_wandb = False
        
    def setup_sampler(self, log_posterior_fn, params):
        """Setup the appropriate sampler based on type"""
        print(f"\n{'='*60}")
        print(f"Setting up {self.sampler_type.upper()} sampler")
        print(f"{'='*60}")

        if self.sampler_type == 'sgld':
            return self._setup_sgld(log_posterior_fn, params)
        elif self.sampler_type == 'sghmc':
            return self._setup_sghmc(log_posterior_fn, params)
        elif self.sampler_type == 'baoa':
            return self._setup_baoa(log_posterior_fn, params)
    
    def _setup_sgld(self, log_posterior_fn, params):
        """Setup Stochastic Gradient Langevin Dynamics (SGLD)"""

        transform = posteriors.sgmcmc.sgld.build(
            log_posterior=log_posterior_fn,
            lr=self.config.get('learning_rate', 1e-6),
            beta=self.config.get('sgld_beta', 0.0),        # Gradient noise correction
            temperature=self.config.get('temperature', 1.0)
        )

        state = transform.init(params)

        print(f"SGLD configured with:")
        print(f"- Learning rate: {self.config.get('learning_rate', 1e-6)}")
        print(f"- Beta (noise correction): {self.config.get('sgld_beta', 0.0)}")
        print(f"- Temperature: {self.config.get('temperature', 1.0)}")
        print(f"- Warmup steps: {self.config.get('warmup_steps', 200)}")
        print(f"- Sampling steps: {self.config.get('sampling_steps', 1000)}")
        print(f"- Thinning (collect every Nth): {self.config.get('thinning', 10)}")

        return transform, state

    def _setup_sghmc(self, log_posterior_fn, params):
        """Setup Stochastic Gradient Hamiltonian Monte Carlo (SGHMC)"""

        transform = posteriors.sgmcmc.sghmc.build(
            log_posterior=log_posterior_fn,
            lr=self.config.get('learning_rate', 1e-7),
            alpha=self.config.get('sghmc_alpha', 0.01),      # Friction coefficient
            beta=self.config.get('sghmc_beta', 0.0),         # Noise estimate (0 = no correction)
            sigma=self.config.get('sghmc_sigma', 1.0),       # Prior std for momenta
            temperature=self.config.get('temperature', 1.0),  # Posterior tempering
            momenta=None  # Will be initialized automatically
        )

        state = transform.init(params)

        print(f"SGHMC configured with:")
        print(f"- Learning rate: {self.config.get('learning_rate', 1e-7)}")
        print(f"- Alpha (friction): {self.config.get('sghmc_alpha', 0.01)}")
        print(f"- Beta (noise estimate): {self.config.get('sghmc_beta', 0.0)}")
        print(f"- Sigma (momenta prior std): {self.config.get('sghmc_sigma', 1.0)}")
        print(f"- Temperature: {self.config.get('temperature', 1.0)}")
        print(f"- Warmup steps: {self.config.get('warmup_steps', 200)}")
        print(f"- Sampling steps: {self.config.get('sampling_steps', 1000)}")
        print(f"- Thinning (collect every Nth): {self.config.get('thinning', 10)}")

        return transform, state

    def _setup_baoa(self, log_posterior_fn, params):
        """Setup Bayesian Adaptive Optimization Algorithm (BAOA)"""

        transform = posteriors.sgmcmc.baoa.build(
            log_posterior=log_posterior_fn,
            lr=self.config.get('learning_rate', 1e-6),
            alpha=self.config.get('baoa_alpha', 0.01),       # Momentum decay
            sigma=self.config.get('baoa_sigma', 1.0),        # Prior std for momenta
            temperature=self.config.get('temperature', 1.0),  # Posterior tempering
            momenta=None  # Will be initialized automatically
        )

        state = transform.init(params)

        print(f"BAOA configured with:")
        print(f"- Learning rate: {self.config.get('learning_rate', 1e-6)}")
        print(f"- Alpha (momentum decay): {self.config.get('baoa_alpha', 0.01)}")
        print(f"- Sigma (momenta prior std): {self.config.get('baoa_sigma', 1.0)}")
        print(f"- Temperature: {self.config.get('temperature', 1.0)}")
        print(f"- Warmup steps: {self.config.get('warmup_steps', 200)}")
        print(f"- Sampling steps: {self.config.get('sampling_steps', 1000)}")
        print(f"- Thinning (collect every Nth): {self.config.get('thinning', 10)}")

        return transform, state

    def run_training(self, transform, state, training_batches,
                     single_batch_loss, log_posterior_fn):
        """Run SGMCMC training with step-based approach"""
        return self._run_mcmc_training(transform, state, training_batches,
                                      single_batch_loss, log_posterior_fn)

    def _run_mcmc_training(self, transform, state, training_batches,
                           single_batch_loss, log_posterior_fn):
        """Step-based training for SGMCMC samplers"""
        import datetime

        warmup_steps = self.config.get('warmup_steps', 200)
        sampling_steps = self.config.get('sampling_steps', 1000)
        thinning = self.config.get('thinning', 10)
        total_steps = warmup_steps + sampling_steps

        print(f"\n{'='*60}")
        print(f"Starting {self.sampler_type.upper()} Sampling")
        print(f"{'='*60}")
        print(f"  Warmup steps: {warmup_steps}")
        print(f"  Sampling steps: {sampling_steps}")
        print(f"  Total steps: {total_steps}")
        print(f"  Thinning: {thinning}")
        print(f"  Expected samples: {sampling_steps // thinning}")
        print(f"{'='*60}\n")

        collected_samples = []
        start_time = datetime.datetime.now()

        # Lists to track losses during training
        step_losses = []
        step_log_posts = []

        # Create infinite batch iterator (cycles through batches)
        def batch_generator():
            while True:
                for batch in training_batches:
                    yield batch

        batch_iter = batch_generator()

        for step in range(total_steps):
            batch = next(batch_iter)

            # Update
            state, aux = transform.update(state, batch)

            # Collect samples after warmup with thinning
            if step >= warmup_steps:
                steps_after_warmup = step - warmup_steps
                if steps_after_warmup % thinning == 0:
                    sample = {k: v.clone().detach() for k, v in state.params.items()}
                    collected_samples.append(sample)

                    if len(collected_samples) % 10 == 0:
                        print(f"  Step {step}/{total_steps} - "
                              f"Samples collected: {len(collected_samples)}")

            # Log metrics periodically
            if step % 50 == 0:
                with torch.no_grad():
                    current_loss = single_batch_loss(state.params, batch)
                    current_log_post, _ = log_posterior_fn(state.params, batch)

                    # Store losses for summary
                    step_losses.append(current_loss.item())
                    step_log_posts.append(current_log_post.item())

                    print(f"  [{step:4d}/{total_steps}] "
                          f"Loss: {current_loss.item():.4f} | "
                          f"Log Post: {current_log_post.item():.4f}")

                    if self.use_wandb:
                        wandb.log({
                            'step': step,
                            'loss': current_loss.item(),
                            'log_posterior': current_log_post.item(),
                            'phase': 'warmup' if step < warmup_steps else 'sampling',
                            'samples_collected': len(collected_samples)
                        }, step=step)

            # Cleanup
            if step % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        total_time = (datetime.datetime.now() - start_time).total_seconds()

        print(f"\n{'='*60}")
        print(f"MCMC Sampling Complete!")
        print(f"  Total samples collected: {len(collected_samples)}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"{'='*60}\n")

        # Store samples
        self.collected_samples = [
            {k: v.to(DEVICE) for k, v in sample.items()}
            for sample in collected_samples
        ]

        # Store metrics
        self.metrics['total_steps'] = total_steps
        self.metrics['samples_collected'] = len(collected_samples)
        self.metrics['training_losses'] = step_losses
        self.metrics['log_posterior_values'] = step_log_posts
        # Create dummy epoch_times for compatibility with summary report
        self.metrics['epoch_times'] = [total_time]

        return state, self.metrics

    def _extract_sampler_metrics(self, state, epoch):
        """Extract sampler-specific metrics from state"""
        metrics = {'epoch': epoch}

        if hasattr(state, 'momentum') or hasattr(state, 'momenta'):
            momentum_dict = getattr(state, 'momentum', None) or getattr(state, 'momenta', None)
            if momentum_dict:
                avg_momentum = torch.mean(torch.stack([
                    torch.abs(v).mean() for v in momentum_dict.values()
                ])).item()
                metrics['avg_momentum'] = avg_momentum

        return metrics
    
    def evaluate_predictions(self, state, model, test_batch, num_samples=5):
        """
        Evaluate model predictions with uncertainty quantification, W&B logging,
        and comparison against deterministic model

        Args:
            state: Bayesian state from training
            model: Original model for deterministic comparison
            test_batch: Test data (x, y)
            num_samples: Number of posterior samples

        Returns:
            dict with evaluation results including deterministic comparison
        """
        print(f"\n{'='*60}")
        print(f"Evaluating {self.sampler_type.upper()} Predictions")
        print(f"{'='*60}")

        x_test, y_test = test_batch
        results = {}

        current_params = state.params

        # Multiple posterior samples
        print(f"\nPredictive Uncertainty ({num_samples} samples):")
        sample_losses = []
        posterior_samples = []

        if len(self.collected_samples) > 0:
            # Use last N samples from training
            samples_to_use = self.collected_samples[-num_samples:]

            for sample_params in samples_to_use:
                with torch.no_grad():
                    sample_logits, _ = func.functional_call(model, sample_params, (x_test,))
                    sample_loss = F.cross_entropy(
                        sample_logits.view(-1, sample_logits.size(-1)),
                        y_test.view(-1)
                    )
                    sample_losses.append(sample_loss.item())
                    posterior_samples.append(sample_logits)
        else:
            print(f"No {self.sampler_type.upper()} samples collected, using final params only")
            # Fallback to just using final params
            sample_params = state.params
            with torch.no_grad():
                sample_logits, _ = func.functional_call(model, sample_params, (x_test,))
                sample_loss = F.cross_entropy(
                    sample_logits.view(-1, sample_logits.size(-1)),
                    y_test.view(-1)
                )
                sample_losses.append(sample_loss.item())
                posterior_samples.append(sample_logits)

        # Save posterior_sampling metrics
        if sample_losses:
            results['posterior_sampling'] = {
                'mean_loss': np.mean(sample_losses),
                'std_loss': np.std(sample_losses),
                'min_loss': np.min(sample_losses),
                'max_loss': np.max(sample_losses),
                'num_samples': num_samples
            }
            print(f"  Mean Loss: {np.mean(sample_losses):.4f} ± {np.std(sample_losses):.4f}")
            print(f"  Range: [{np.min(sample_losses):.4f}, {np.max(sample_losses):.4f}]")

        # Predictive uncertainty
        print(f"\nCalculating Uncertainty:")
        stacked_logits = torch.stack(posterior_samples)
        pred_probs = F.softmax(stacked_logits, dim=-1)
        mean_probs = pred_probs.mean(dim=0)

        pred_entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=-1)
        avg_uncertainty = pred_entropy.mean().item()

        results['uncertainty'] = {
            'avg_predictive_entropy': avg_uncertainty,
            'max_predictive_entropy': pred_entropy.max().item(),
            'min_predictive_entropy': pred_entropy.min().item()
        }

        print(f"  Average Entropy: {avg_uncertainty:.4f}")
        print(f"  Range: [{pred_entropy.min().item():.4f}, {pred_entropy.max().item():.4f}]")

        # Log evaluation results to W&B
        if self.use_wandb:
            wandb.log({
                'eval/avg_predictive_entropy': results['uncertainty']['avg_predictive_entropy'],
                'eval/max_predictive_entropy': results['uncertainty']['max_predictive_entropy'],
                'eval/min_predictive_entropy': results['uncertainty']['min_predictive_entropy']
            })
            wandb.summary.update({
                'eval_uncertainty': results['uncertainty']['avg_predictive_entropy']
            })
        
        print(f"{'='*60}\n")
        
        return results
    
    def save_model(self, state, model, evaluation_results=None):
        """Save model, state, and all metrics (with W&B artifact logging)"""
        import datetime
        import json
        
        project_root = Path() #.resolve().parents[0]
        print("PROJECT ROOT:", project_root)  # Debugging line
        save_dir = project_root / self.config.get('save_dir', 'checkpoints') / f"{self.sampler_type}_sampler"
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        version_dir = save_dir / f"run_{timestamp}"
        version_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Saving Model and Metrics")
        print(f"{'='*60}")
        
        # Prepare complete metrics
        # For MCMC samplers, only save relevant metrics
        if self.sampler_type in ['sgld', 'sghmc', 'baoa']:
            # Extract MCMC-specific training metrics
            mcmc_training_metrics = {
                'total_steps': self.metrics.get('total_steps', 0),
                'samples_collected': self.metrics.get('samples_collected', 0)
            }

            # Extract momentum statistics if available
            if hasattr(state, 'momentum') or hasattr(state, 'momenta'):
                momentum_dict = getattr(state, 'momentum', None) or getattr(state, 'momenta', None)
                if momentum_dict:
                    avg_momentum = torch.mean(torch.stack([
                        torch.abs(v).mean() for v in momentum_dict.values()
                    ])).item()
                    mcmc_training_metrics['avg_momentum'] = avg_momentum

            complete_metrics = {
                'sampler_type': self.sampler_type,
                'config': self.config,
                'training_metrics': mcmc_training_metrics,
                'timestamp': datetime.datetime.now().isoformat()
            }
            # Only include uncertainty metrics from evaluation
            if evaluation_results and 'uncertainty' in evaluation_results:
                complete_metrics['uncertainty'] = evaluation_results['uncertainty']

        current_params = state.params
        
        # Save model state
        model_path = version_dir / f'{self.sampler_type}_model.pt'

        # Build saved state incrementally
        saved_state = {
            'model_state_dict': model.state_dict(),
            'sampler_state_params': {k: v.cpu() for k, v in current_params.items()},
            'complete_metrics': complete_metrics
        }

        # Save opt_state if present
        if hasattr(state, 'opt_state'):
            # Handle nested opt_state structure with named tuples
            def move_to_cpu_recursive(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.cpu()
                elif isinstance(obj, dict):
                    return {k: move_to_cpu_recursive(v) for k, v in obj.items()}
                elif isinstance(obj, tuple) and hasattr(obj, '_fields'):  # Named tuple
                    # Reconstruct named tuple with CPU tensors
                    return type(obj)(*[move_to_cpu_recursive(item) for item in obj])
                elif isinstance(obj, (list, tuple)):
                    converted = [move_to_cpu_recursive(item) for item in obj]
                    return type(obj)(converted) if isinstance(obj, tuple) else converted
                else:
                    return obj
            
            saved_state['opt_state'] = move_to_cpu_recursive(state.opt_state)
        else:
            saved_state['opt_state'] = {}  # Empty dict as fallback

        # Add aux if it exists and is not empty
        if hasattr(state, 'aux') and state.aux:
            saved_state['sampler_state_aux'] = {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in state.aux.items()
            }

        # Save collected samples for SGMCMC samplers
        if self.sampler_type in ['sgld', 'sghmc', 'baoa']:
            if self.collected_samples:
                saved_state['collected_samples'] = [
                    {k: v.cpu() for k, v in sample.items()}
                    for sample in self.collected_samples
                ]
                print(f"  Saving {len(self.collected_samples)} collected samples...")

        torch.save(saved_state, model_path)
        print(f"✓ Model saved: {model_path}")
        
        # Save metrics as JSON
        metrics_path = version_dir / f'{self.sampler_type}_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(complete_metrics, f, indent=2)
        print(f"✓ Metrics saved: {metrics_path}")
        
        # Save summary report
        report_path = version_dir / f'{self.sampler_type}_summary.txt'
        self._save_summary_report(complete_metrics, report_path)
        print(f"✓ Summary saved: {report_path}")
        
        # Log model to W&B as artifact
        if self.use_wandb:
            try:
                artifact = wandb.Artifact(
                    name=f"bayesian-model-{self.sampler_type}",
                    type='model',
                    description=f"Bayesian {self.sampler_type.upper()} model checkpoint"
                )
                artifact.add_file(str(model_path))
                artifact.add_file(str(metrics_path))
                artifact.add_file(str(report_path))
                wandb.log_artifact(artifact)
                print(f"✓ Model logged to W&B as artifact")
            except Exception as e:
                print(f"Warning: Could not log artifact to W&B: {e}")
        
        print(f"{'='*60}\n")
        
        return save_dir
    
    def finish(self):
        """Finish W&B run"""
        if self.use_wandb and self.wandb_run:
            wandb.finish()
            print("✓ W&B run finished")    
            
    def _save_summary_report(self, metrics, report_path):
        """Generate and save human-readable summary"""
        report = f"""
{'='*70}
BAYESIAN MODEL TRAINING SUMMARY
{'='*70}
Sampler: {self.sampler_type.upper()}
Generated: {metrics['timestamp']}

CONFIGURATION:
{'-'*70}
"""
        for key, value in metrics['config'].items():
            report += f"  {key}: {value}\n"

        # Training summary for SGMCMC samplers
        training_metrics = metrics['training_metrics']
        report += f"""
TRAINING SUMMARY:
{'-'*70}
Total Steps: {training_metrics.get('total_steps', 'N/A')}
Samples Collected: {training_metrics.get('samples_collected', 'N/A')}
"""
        if 'avg_momentum' in training_metrics:
            report += f"Avg Momentum: {training_metrics['avg_momentum']:.6f}\n"
        
        if 'evaluation' in metrics:
            eval_data = metrics['evaluation']
            
            # Deterministic vs Bayesian comparison
            if 'deterministic' in eval_data:
                det = eval_data['deterministic']
                bay = eval_data['posterior_mean']
                
                report += f"""
EVALUATION RESULTS:
{'-'*70}

Deterministic Model (Original):
  Loss: {det['loss']:.4f}
  Perplexity: {det['perplexity']:.4f}

Bayesian Posterior Mean:
  Loss: {bay['loss']:.4f}
  Perplexity: {bay['perplexity']:.4f}
  Improvement: {bay['improvement_over_deterministic']:+.4f}
  Status: {'✓ Better than deterministic' if bay['better_than_deterministic'] else '✗ Worse than deterministic'}

"""
            
            # Posterior sampling results
            if 'posterior_sampling' in eval_data:
                samp = eval_data['posterior_sampling']
                report += f"""Posterior Sampling ({samp['num_samples']} samples):
  Mean Loss: {samp['mean_loss']:.4f} ± {samp['std_loss']:.4f}
  Loss Range: [{samp['min_loss']:.4f}, {samp['max_loss']:.4f}]

"""
            
            # Uncertainty quantification
            if 'uncertainty' in eval_data:
                unc = eval_data['uncertainty']
                report += f"""Uncertainty Quantification:
  Avg Predictive Entropy: {unc['avg_predictive_entropy']:.4f}
  Entropy Range: [{unc['min_predictive_entropy']:.4f}, {unc['max_predictive_entropy']:.4f}]
"""
            
            # Parameter uncertainty
            if 'parameter_uncertainty' in eval_data:
                param_unc = eval_data['parameter_uncertainty']
                report += f"""
Parameter Uncertainty:
  Avg Parameter Std: {param_unc['avg_parameter_std']:.6f}
"""
        
        report += f"\n{'='*70}\n"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)


def run_bayesian_pipeline(training_batches, sampler_type='sgld', config=None, use_wandb=True):
    """
    Complete pipeline with W&B integration

    Args:
        training_batches: Training data batches
        sampler_type: One of 'sgld', 'sghmc', 'baoa'
        config: Configuration dictionary (optional, uses defaults if not provided)
        use_wandb: Whether to use Weights & Biases tracking

    Returns:
        state: Training state (NamedTuple with params and sampler-specific fields)
        metrics: Dictionary of training metrics
        eval_results: Dictionary of evaluation results
        collected_samples: List of parameter dictionaries
    """
    params = {k: v.clone().to(DEVICE) for k, v in MODEL.named_parameters()}

    # Select appropriate config if not provided
    if config is None:
        if sampler_type == 'sgld':
            config = CONFIG_SGLD.copy()
        elif sampler_type == 'sghmc':
            config = CONFIG_SGHMC.copy()
        elif sampler_type == 'baoa':
            config = CONFIG_BAOA.copy()
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}. Must be one of: 'sgld', 'sghmc', 'baoa'")

    pipeline = BayesianSamplerPipeline(
        sampler_type=sampler_type,
        config=config,
        use_wandb=use_wandb
    )
    
    try:
        transform, state = pipeline.setup_sampler(log_posterior_fn, params)
        state, metrics = pipeline.run_training(
            transform, state, training_batches,
            single_batch_loss, log_posterior_fn
        )
        
        # Pass the model for deterministic comparison
        eval_results = pipeline.evaluate_predictions(
            state, MODEL, training_batches[0], num_samples=5
        )
        
        save_dir = pipeline.save_model(state, MODEL, eval_results)

        # Get collected samples for SGMCMC samplers before pipeline is destroyed
        collected_samples = pipeline.collected_samples if pipeline.collected_samples else None

        return state, metrics, eval_results, collected_samples

    finally:
        pipeline.finish()


# Usage examples:
# state_sgld, metrics_sgld, eval_sgld, samples_sgld = run_bayesian_pipeline(training_batches, 'sgld')
# state_sghmc, metrics_sghmc, eval_sghmc, samples_sghmc = run_bayesian_pipeline(training_batches, 'sghmc')
# state_baoa, metrics_baoa, eval_baoa, samples_baoa = run_bayesian_pipeline(training_batches, 'baoa')
