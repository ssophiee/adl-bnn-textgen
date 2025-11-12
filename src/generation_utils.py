import json
from datetime import datetime
from typing import Optional, Dict, Any
import torch 
from torch import func

def load_checkpoint_for_generation(checkpoint_path, device='cpu'):
    """
    Load a saved checkpoint and prepare it for text generation.

    This function handles both VI-based samplers (VI, EKF, Laplace) and SGMCMC samplers (SGLD, SGHMC, BAOA).

    Args:
        checkpoint_path: Path to the saved .pt checkpoint file
        device: Device to load tensors to ('cpu', 'cuda', 'mps')

    Returns:
        dict with:
            - 'sampler_type': Type of sampler used
            - 'state': Reconstructed state object (for VI-based samplers)
            - 'collected_samples': List of parameter samples (for SGMCMC samplers)
            - 'params': Final parameters (always available)
            - 'model_state_dict': Model state dictionary

    Example:
        >>> checkpoint_data = load_checkpoint_for_generation('path/to/vi_model.pt')
        >>> # For VI-based generation, use checkpoint_data['state']
        >>> # For SGMCMC generation, use checkpoint_data['collected_samples']
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract sampler type from metrics
    sampler_type = checkpoint['complete_metrics']['sampler_type']

    result = {
        'sampler_type': sampler_type,
        'model_state_dict': checkpoint['model_state_dict'],
        'params': {k: v.to(device) for k, v in checkpoint['sampler_state_params'].items()},
        'metrics': checkpoint['complete_metrics']
    }

    # For VI-based samplers, reconstruct the state
    if sampler_type in ['vi', 'ekf', 'laplace']:
        from posteriors.vi.diag import VIDiagState

        state = VIDiagState(
            params={k: v.to(device) for k, v in checkpoint['sampler_state_params'].items()},
            log_sd_diag={k: v.to(device) for k, v in checkpoint['log_sd_diag'].items()},
            opt_state=checkpoint['opt_state']
        )
        result['state'] = state
        result['log_sd_diag'] = {k: v.to(device) for k, v in checkpoint['log_sd_diag'].items()}

    # For SGMCMC samplers, load collected samples if available
    elif sampler_type in ['sgld', 'sghmc', 'baoa', 'sgmcmc']:
        if 'collected_samples' in checkpoint and checkpoint['collected_samples']:
            result['collected_samples'] = [
                {k: v.to(device) for k, v in sample.items()}
                for sample in checkpoint['collected_samples']
            ]
            print(f"Loaded {len(result['collected_samples'])} collected samples from checkpoint")
        else:
            result['collected_samples'] = []
            result['note'] = (
                f"No collected samples found in checkpoint. "
                f"You can use the final params as a MAP estimate for generation."
            )

    return result


def generate_text_bayesian_sgmcmc(model, collected_samples, start_prompt, encode_fn, decode_fn,
                                   max_new_tokens=500, temperature=1.0, top_k=None,
                                   num_samples=None):
    """
    Generate text using collected SGMCMC samples.

    This function is specifically for SGMCMC samplers (SGLD, SGHMC, BAOA) that collect
    discrete parameter samples during training.

    Args:
        model: The base model
        collected_samples: List of parameter dictionaries from SGMCMC sampling
        start_prompt: Starting text string
        encode_fn: Function to encode text to tokens
        decode_fn: Function to decode tokens to text
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature
        top_k: If set, only sample from top k tokens
        num_samples: Number of samples to use (default: all samples)

    Returns:
        generated_text: String of generated text
        uncertainty_info: Dict with token-level uncertainties

    Example:
        >>> # After training with SGMCMC
        >>> pipeline = BayesianSamplerPipeline('sgld', config)
        >>> # ... setup and training ...
        >>> # Use collected samples for generation
        >>> text, unc = generate_text_bayesian_sgmcmc(
        ...     model, pipeline.collected_samples, "To be or not to be",
        ...     encode, decode, max_new_tokens=100, num_samples=10
        ... )
    """
    import numpy as np
    import torch.nn.functional as F

    model.eval()
    device = next(model.parameters()).device

    # Select samples to use
    if num_samples is None or num_samples > len(collected_samples):
        param_samples = collected_samples
    else:
        # Use the last N samples (most converged)
        param_samples = collected_samples[-num_samples:]

    print(f"Using {len(param_samples)} SGMCMC samples for generation")

    # Encode starting prompt
    context = torch.tensor(encode_fn(start_prompt), dtype=torch.long, device=device).unsqueeze(0)

    generated_tokens = []
    token_uncertainties = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get predictions from all parameter samples
            all_logits = []

            for params in param_samples:
                # Forward pass with current parameters
                logits, _ = func.functional_call(model, params, (context,))
                # Get logits for last token
                logits = logits[:, -1, :] / temperature
                all_logits.append(logits)

            # Average logits across samples
            avg_logits = torch.stack(all_logits).mean(dim=0)

            # Calculate uncertainty (entropy of averaged predictions)
            probs = torch.stack([F.softmax(l, dim=-1) for l in all_logits])
            mean_probs = probs.mean(dim=0)
            entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=-1)
            token_uncertainties.append(entropy.item())

            # Apply top-k filtering if specified
            if top_k is not None:
                v, _ = torch.topk(avg_logits, min(top_k, avg_logits.size(-1)))
                avg_logits[avg_logits < v[:, [-1]]] = -float('Inf')

            # Sample next token
            probs = F.softmax(avg_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated_tokens.append(next_token.item())

            # Append to context
            context = torch.cat([context, next_token], dim=1)

            # Crop context to max sequence length
            if context.size(1) > model.config.block_size:
                context = context[:, -model.config.block_size:]

    # Decode generated tokens
    generated_text = decode_fn(generated_tokens)
    full_text = start_prompt + generated_text

    uncertainty_info = {
        'token_uncertainties': token_uncertainties,
        'avg_uncertainty': np.mean(token_uncertainties),
        'max_uncertainty': np.max(token_uncertainties),
        'num_samples_used': len(param_samples)
    }

    return full_text, uncertainty_info

def save_generation_result(
    start_prompt: str,
    text: str,
    unc_info: Dict[str, Any],
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
    num_samples: int,
    collected_samples: Optional[Any] = None,
    save_path: str = "generation_results.json",
    sample_id: Optional[str] = None
):
    """
    Save generation results to a JSON file.
    
    Args:
        start_prompt: The prompt used to start generation
        text: The generated text (including prompt)
        unc_info: Uncertainty information dictionary
        max_new_tokens: Maximum new tokens parameter
        temperature: Temperature parameter
        top_k: Top-k parameter
        num_samples: Number of samples used
        collected_samples: The collected samples (can be None if not saved)
        save_path: Path to JSON file (default: "generation_results.json")
        sample_id: Optional custom ID for this sample (default: timestamp)
    
    Returns:
        sample_id: The ID assigned to this sample
    """
    # Load existing results or create new dict
    try:
        with open(save_path, 'r') as f:
            results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        results = {}
    
    # Generate sample ID if not provided
    if sample_id is None:
        sample_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create entry for this sample
    results[sample_id] = {
        "generation_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "num_samples": num_samples,
            "has_collected_samples": collected_samples is not None
        },
        "start_prompt": start_prompt,
        "text": text,
        "unc_info": {
            "avg_uncertainty": float(unc_info.get('avg_uncertainty', 0.0)),
            "max_uncertainty": float(unc_info.get('max_uncertainty', 0.0)),
            "num_samples_used": unc_info.get('num_samples_used', num_samples)
        }
    }
    
    # Save updated results
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved generation result with ID: {sample_id}")
    return sample_id


def load_generation_results(save_path: str = "generation_results.json") -> Dict:
    """Load all generation results from JSON file."""
    try:
        with open(save_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"No results file found at {save_path}")
        return {}
