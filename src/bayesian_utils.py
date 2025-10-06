"""
Utility functions for Bayesian inference with deterministic model comparison.
"""
import torch
import wandb
import datetime
import numpy as np
from torch import func
import posteriors
import torch.nn.functional as F
from pathlib import Path
from config import DEVICE, CONFIG, MODEL_PATH, WANDB_AVAILABLE
from src.nanogpt_utils import load_model
MODEL, checkpoint = load_model(Path(MODEL_PATH))



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
    try:
        nll = single_batch_loss(params, batch)
        print(f"NLL computed successfully: {nll}")
    except Exception as e:
        print(f"Error in single_batch_loss: {e}")
        raise
    
    try:
        # TODO: Adjust prior std if needed
        log_prior = posteriors.diag_normal_log_prob(params)
        print(f"Log prior computed successfully: {log_prior}")
    except Exception as e:
        print(f"Error in diag_normal_log_prob: {e}")
        raise
    
    try:
        num_data_tensor = torch.tensor(float(CONFIG['train_samples']), device=DEVICE, dtype=torch.float32)
        log_posterior = -nll + log_prior / num_data_tensor
        print(f"Log posterior computed successfully: {log_posterior}")
        return log_posterior, {}
    except Exception as e:
        print(f"Error in final computation: {e}")
        raise


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
    
    AVAILABLE_SAMPLERS = ['vi', 'ekf', 'laplace', 'sgmcmc']
    
    def __init__(self, sampler_type: str, config: dict, use_wandb: bool = True):
        """
        Initialize the Bayesian sampler pipeline
        
        Args:
            sampler_type: One of 'vi', 'ekf', 'laplace', 'sgmcmc'
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
        
        if self.sampler_type == 'vi':
            return self._setup_vi(log_posterior_fn, params)
        elif self.sampler_type == 'ekf':
            return self._setup_ekf(log_posterior_fn, params)
        elif self.sampler_type == 'laplace':
            return self._setup_laplace(log_posterior_fn, params)
        elif self.sampler_type == 'sgmcmc':
            return self._setup_sgmcmc(log_posterior_fn, params)
    
    def _setup_vi(self, log_posterior_fn, params):
        """Setup Variational Inference (Diagonal Gaussian)"""
        import torchopt
        optimizer = torchopt.adam(lr=self.config.get('learning_rate', 1e-3))
        
        transform = posteriors.vi.diag.build(
            log_posterior=log_posterior_fn,
            optimizer=optimizer,
            temperature=self.config.get('temperature', 1.0),
            n_samples=self.config.get('vi_n_samples', 1)
        )
        
        state = transform.init(params)
        
        print(f"VI configured with:")
        print(f"- Learning rate: {self.config.get('learning_rate', 1e-3)}")
        print(f"- Temperature: {self.config.get('temperature', 1.0)}")
        print(f"- Samples per update: {self.config.get('vi_n_samples', 1)}")
        
        return transform, state
    
    def _setup_ekf(self, log_posterior_fn, params):
        """Setup Extended Kalman Filter"""
        transform = posteriors.ekf.diag_fisher.build(
            log_posterior=log_posterior_fn,
            lr=self.config.get('learning_rate', 1e-3)
        )
        
        state = transform.init(params)
        
        print(f"EKF configured with:")
        print(f"- Learning rate: {self.config.get('learning_rate', 1e-3)}")
        print(f"- Using diagonal Fisher approximation")
        
        return transform, state
    
    def _setup_laplace(self, log_posterior_fn, params):
        """Setup Laplace Approximation"""
        transform = posteriors.laplace.diag_fisher.build(
            log_posterior=log_posterior_fn,
            lr=self.config.get('learning_rate', 1e-3)
        )
        
        state = transform.init(params)
        
        print(f"✓ Laplace configured with:")
        print(f"- Learning rate: {self.config.get('learning_rate', 1e-3)}")
        print(f"- Using diagonal Fisher approximation")
        
        return transform, state
    
    def _setup_sgmcmc(self, log_posterior_fn, params):
        """Setup Stochastic Gradient MCMC (SGHMC)"""
        transform = posteriors.sgmcmc.sghmc.build(
            log_posterior=log_posterior_fn,
            lr=self.config.get('learning_rate', 1e-3),
            alpha=self.config.get('sgmcmc_alpha', 0.01),
            beta=self.config.get('sgmcmc_beta', 0.0)
        )
        
        state = transform.init(params)
        
        print(f"SGMCMC (SGHMC) configured with:")
        print(f"- Learning rate: {self.config.get('learning_rate', 1e-3)}")
        print(f"- Alpha (momentum decay): {self.config.get('sgmcmc_alpha', 0.01)}")
        print(f"- Beta (noise coefficient): {self.config.get('sgmcmc_beta', 0.0)}")
        
        return transform, state
    
    def run_training(self, transform, state, training_batches, 
                     single_batch_loss, log_posterior_fn):
        """Run Bayesian training with the configured sampler and W&B tracking"""
        import datetime
        
        print(f"\n{'='*60}")
        print(f"Starting Bayesian Training with {self.sampler_type.upper()}")
        print(f"{'='*60}")
        print(f"Configuration:")
        print(f"  - Epochs: {self.config['num_epochs']}")
        print(f"  - Batches per epoch: {len(training_batches)}")
        print(f"  - Total iterations: {self.config['num_epochs'] * len(training_batches)}")
        print(f"{'='*60}\n")
        
        start_time = datetime.datetime.now()
        global_step = 0
        
        for epoch in range(self.config['num_epochs']):
            epoch_start = datetime.datetime.now()
            epoch_losses = []
            epoch_log_posts = []
            
            print(f"Epoch {epoch + 1}/{self.config['num_epochs']}")
            print("-" * 60)
            
            for batch_idx, batch in enumerate(training_batches):
                try:
                    state = transform.update(state, batch)
                    
                    with torch.no_grad():
                        current_loss = single_batch_loss(state.params, batch)
                        current_log_post, _ = log_posterior_fn(state.params, batch)
                        
                        epoch_losses.append(current_loss.item())
                        epoch_log_posts.append(current_log_post.item())
                    
                    if self.use_wandb:
                        wandb.log({
                            'batch_loss': current_loss.item(),
                            'batch_log_posterior': current_log_post.item(),
                            'epoch': epoch + 1,
                            'global_step': global_step
                        }, step=global_step)
                    
                    global_step += 1
                    
                    if (batch_idx + 1) % max(1, len(training_batches) // 5) == 0:
                        recent_loss = np.mean(epoch_losses[-5:])
                        recent_log_post = np.mean(epoch_log_posts[-5:])
                        print(f"  [{batch_idx + 1:3d}/{len(training_batches)}] "
                              f"Loss: {recent_loss:.4f} | "
                              f"Log Post: {recent_log_post:.4f}")
                
                except Exception as e:
                    print(f"  ✗ Error in batch {batch_idx + 1}: {e}")
                    continue
            
            if epoch_losses:
                avg_loss = np.mean(epoch_losses)
                avg_log_post = np.mean(epoch_log_posts)
                epoch_time = (datetime.datetime.now() - epoch_start).total_seconds()
                
                self.metrics['training_losses'].append(avg_loss)
                self.metrics['log_posterior_values'].append(avg_log_post)
                self.metrics['epoch_times'].append(epoch_time)
                
                sampler_metrics = self._extract_sampler_metrics(state, epoch)
                if sampler_metrics:
                    self.metrics['sampler_specific_metrics'].append(sampler_metrics)
                
                if self.use_wandb:
                    log_dict = {
                        'epoch_loss': avg_loss,
                        'epoch_log_posterior': avg_log_post,
                        'epoch_time': epoch_time,
                        'epoch': epoch + 1
                    }
                    if sampler_metrics:
                        log_dict.update({f"epoch_{k}": v for k, v in sampler_metrics.items()})
                    wandb.log(log_dict, step=global_step)
                
                print(f"  ✓ Epoch Complete:")
                print(f"    Loss: {avg_loss:.4f} | Log Post: {avg_log_post:.4f} | "
                      f"Time: {epoch_time:.2f}s")
            else:
                print(f"  ✗ Epoch failed - no successful batches")
                break
            
            print()
        
        total_time = (datetime.datetime.now() - start_time).total_seconds()
        
        print(f"{'='*60}")
        print(f"Training Complete!")
        if self.metrics['training_losses']:
            print(f"  Final Loss: {self.metrics['training_losses'][-1]:.4f}")
            print(f"  Final Log Posterior: {self.metrics['log_posterior_values'][-1]:.4f}")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"{'='*60}\n")
        
        if self.use_wandb and self.metrics['training_losses']:
            wandb.summary['final_loss'] = self.metrics['training_losses'][-1]
            wandb.summary['final_log_posterior'] = self.metrics['log_posterior_values'][-1]
            wandb.summary['total_training_time'] = total_time
        
        return state, self.metrics
    
    def _extract_sampler_metrics(self, state, epoch):
        """Extract sampler-specific metrics from state"""
        metrics = {'epoch': epoch}
        
        if self.sampler_type in ['vi', 'ekf', 'laplace']:
            if hasattr(state, 'log_scale'):
                avg_std = torch.mean(torch.stack([
                    torch.exp(v).mean() for v in state.log_scale.values()
                ])).item()
                metrics['avg_parameter_std'] = avg_std
        
        if self.sampler_type == 'sgmcmc':
            if hasattr(state, 'momentum'):
                avg_momentum = torch.mean(torch.stack([
                    torch.abs(v).mean() for v in state.momentum.values()
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
        
        # 0. Deterministic model evaluation
        print("\n0. Deterministic Model (Original):")
        deterministic_results = evaluate_deterministic_model(model, test_batch)
        results['deterministic'] = deterministic_results
        
        print(f"  Loss: {deterministic_results['loss']:.4f}")
        print(f"  Perplexity: {deterministic_results['perplexity']:.4f}")
        
        # 1. Posterior mean prediction
        print("\n1. Bayesian Posterior Mean Prediction:")
        with torch.no_grad():
            logits, _ = func.functional_call(model, state.params, (x_test,))
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y_test.view(-1)
            )
            perplexity = torch.exp(loss)
            
            # Calculate improvement
            loss_improvement = deterministic_results['loss'] - loss.item()
            better_than_deterministic = loss_improvement > 0
            
            results['posterior_mean'] = {
                'loss': loss.item(),
                'perplexity': perplexity.item(),
                'improvement_over_deterministic': loss_improvement,
                'better_than_deterministic': better_than_deterministic
            }
            
            print(f"  Loss: {loss.item():.4f}")
            print(f"  Perplexity: {perplexity.item():.4f}")
            print(f"  Improvement over deterministic: {loss_improvement:+.4f}")
            print(f"  {'✓ Better' if better_than_deterministic else '✗ Worse'} than deterministic")
        
        # 2. Multiple posterior samples
        print(f"\n2. Posterior Sampling ({num_samples} samples):")
        sample_losses = []
        posterior_samples = []
        
        for i in range(num_samples):
            if self.sampler_type == 'vi':
                sample_params = posteriors.vi.diag.sample(state)
            elif self.sampler_type == 'ekf':
                sample_params = posteriors.ekf.diag_fisher.sample(state)
            elif self.sampler_type == 'laplace':
                sample_params = posteriors.laplace.diag_fisher.sample(state)
            elif self.sampler_type == 'sgmcmc':
                sample_params = state.params
            
            with torch.no_grad():
                sample_logits, _ = func.functional_call(model, sample_params, (x_test,))
                sample_loss = F.cross_entropy(
                    sample_logits.view(-1, sample_logits.size(-1)),
                    y_test.view(-1)
                )
                sample_losses.append(sample_loss.item())
                posterior_samples.append(sample_logits)
        
        results['posterior_sampling'] = {
            'mean_loss': np.mean(sample_losses),
            'std_loss': np.std(sample_losses),
            'min_loss': np.min(sample_losses),
            'max_loss': np.max(sample_losses),
            'num_samples': num_samples
        }
        
        print(f"  Mean Loss: {np.mean(sample_losses):.4f} ± {np.std(sample_losses):.4f}")
        print(f"  Range: [{np.min(sample_losses):.4f}, {np.max(sample_losses):.4f}]")
        
        # 3. Predictive uncertainty
        print(f"\n3. Predictive Uncertainty:")
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
        
        # 4. Parameter uncertainty (if available)
        if hasattr(state, 'log_scale'):
            print(f"\n4. Parameter Uncertainty:")
            total_params = 0
            total_std = 0
            
            for name in state.params:
                param_std = torch.exp(state.log_scale[name]).mean().item()
                param_count = state.params[name].numel()
                
                total_params += param_count
                total_std += param_std * param_count
            
            if total_params > 0:
                avg_param_std = total_std / total_params
                results['parameter_uncertainty'] = {
                    'avg_parameter_std': avg_param_std
                }
                print(f"  Average parameter std: {avg_param_std:.6f}")
        
        # Log evaluation results to W&B
        if self.use_wandb:
            wandb.log({
                'eval/deterministic_loss': results['deterministic']['loss'],
                'eval/deterministic_perplexity': results['deterministic']['perplexity'],
                'eval/posterior_mean_loss': results['posterior_mean']['loss'],
                'eval/posterior_mean_perplexity': results['posterior_mean']['perplexity'],
                'eval/improvement_over_deterministic': results['posterior_mean']['improvement_over_deterministic'],
                'eval/sampling_mean_loss': results['posterior_sampling']['mean_loss'],
                'eval/sampling_std_loss': results['posterior_sampling']['std_loss'],
                'eval/avg_predictive_entropy': results['uncertainty']['avg_predictive_entropy']
            })
            
            wandb.summary.update({
                'eval_deterministic_loss': results['deterministic']['loss'],
                'eval_bayesian_loss': results['posterior_mean']['loss'],
                'eval_improvement': results['posterior_mean']['improvement_over_deterministic'],
                'eval_perplexity': results['posterior_mean']['perplexity'],
                'eval_uncertainty': results['uncertainty']['avg_predictive_entropy']
            })
        
        print(f"{'='*60}\n")
        
        return results
    
    def save_model(self, state, model, evaluation_results=None):
        """Save model, state, and all metrics (with W&B artifact logging)"""
        import datetime
        import json
        
        project_root = Path().resolve().parents[0]
        save_dir = project_root / self.config.get('save_dir', 'checkpoints') / f"{self.sampler_type}_sampler"
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        version_dir = save_dir / f"run_{timestamp}"
        version_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Saving Model and Metrics")
        print(f"{'='*60}")
        
        # Prepare complete metrics
        complete_metrics = {
            'sampler_type': self.sampler_type,
            'config': self.config,
            'training_metrics': self.metrics,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        if evaluation_results:
            complete_metrics['evaluation'] = evaluation_results
        
        # Save model state
        model_path = version_dir / f'{self.sampler_type}_model.pt'
        saved_state = {
            'model_state_dict': model.state_dict(),
            'sampler_state_params': {k: v.cpu() for k, v in state.params.items()},
            'sampler_state_aux': {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                                  for k, v in state.aux.items()} if hasattr(state, 'aux') else {},
            'complete_metrics': complete_metrics
        }
        
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
        
        report += f"""
TRAINING SUMMARY:
{'-'*70}
Total Epochs: {len(metrics['training_metrics']['training_losses'])}
Final Training Loss: {metrics['training_metrics']['training_losses'][-1]:.4f}
Final Log Posterior: {metrics['training_metrics']['log_posterior_values'][-1]:.4f}
Total Training Time: {sum(metrics['training_metrics']['epoch_times']):.2f}s
Avg Time per Epoch: {np.mean(metrics['training_metrics']['epoch_times']):.2f}s
"""
        
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
        
        with open(report_path, 'w') as f:
            f.write(report)


def run_bayesian_pipeline(training_batches, sampler_type='vi', use_wandb=True):
    """
    Complete pipeline with W&B integration and deterministic comparison
    
    Args:
        training_batches: Training data batches
        sampler_type: One of 'vi', 'ekf', 'laplace', 'sgmcmc'
        use_wandb: Whether to use Weights & Biases tracking
    """
    params = dict(MODEL.named_parameters())

    pipeline = BayesianSamplerPipeline(
        sampler_type=sampler_type,
        config=CONFIG,
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
        
        return state, metrics, eval_results
    
    finally:
        pipeline.finish()


# Usage examples:
# state_vi, metrics_vi, eval_vi = run_bayesian_pipeline(training_batches, 'vi')
# state_ekf, metrics_ekf, eval_ekf = run_bayesian_pipeline(training_batches, 'ekf')
# state_laplace, metrics_laplace, eval_laplace = run_bayesian_pipeline(training_batches, 'laplace')
# state_sgmcmc, metrics_sgmcmc, eval_sgmcmc = run_bayesian_pipeline(training_batches, 'sgmcmc')