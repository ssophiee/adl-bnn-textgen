import json
import pickle
from datetime import datetime
from typing import Optional, Dict, Any
import torch
from torch import func
from pathlib import Path
from config import MODEL_PATH, META_PATH, DEVICE

def load_checkpoint_for_generation(checkpoint_path, device=DEVICE):
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

    # For MCMC samplers, load collected samples if available
    elif sampler_type in ['sgld', 'sghmc', 'baoa']:
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


def _load_tokenizer(meta_path):
    """Load tokenizer from meta.pkl file."""
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    return encode, decode


def generate_text_bayesian_sgmcmc(model_path, start_prompt,
                                   max_new_tokens=500, temperature=1.0, top_k=None,
                                   num_samples=None, device=DEVICE, meta_path=None):
    """
    Generate text using collected SGMCMC samples from a trained Bayesian model.

    This function loads the model, tokenizer, and collected samples, then generates text
    with uncertainty quantification.

    Args:
        model_path: Path to the saved Bayesian model checkpoint (.pt file)
        start_prompt: Starting text string
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature
        top_k: If set, only sample from top k tokens
        num_samples: Number of samples to use (default: all samples)
        device: Device to run generation on ('cpu', 'cuda', 'mps')
        meta_path: Path to meta.pkl file (default: auto-detect from baselines/nanogpt/)

    Returns:
        generated_text: String of generated text
        uncertainty_info: Dict with token-level uncertainties

    Example:
        >>> text, unc = generate_text_bayesian_sgmcmc(
        ...     'checkpoints/sgld_model.pt',
        ...     "To be or not to be",
        ...     max_new_tokens=100,
        ...     temperature=0.8,
        ...     top_k=200,
        ...     num_samples=20
        ... )
    """
    import numpy as np
    import torch.nn.functional as F
    from baselines.nanogpt.model import GPT, GPTConfig

    # Load checkpoint
    checkpoint_data = load_checkpoint_for_generation(model_path, device=device)

    # Check if we have collected samples
    if 'collected_samples' not in checkpoint_data or not checkpoint_data['collected_samples']:
        raise ValueError(
            f"No collected samples found in checkpoint. "
            f"This function requires SGMCMC samples (SGLD, SGHMC, BAOA)."
        )

    collected_samples = checkpoint_data['collected_samples']

    # Load model architecture
    model_state_dict = checkpoint_data['model_state_dict']

    model_args = checkpoint_data.get('model_args', {
        'n_layer': 6, 'n_head': 6, 'n_embd': 384, 
        'block_size': 256, 'bias': False, 'vocab_size': 65, 'dropout': 0.0
    })
    gptconf = GPTConfig(**model_args)

    model = GPT(gptconf)
    model.eval()  
    model.to(DEVICE)
    
    # Load tokenizer
    if meta_path is None:
        meta_path = Path(META_PATH)

    encode, decode = _load_tokenizer(meta_path)

    # Select samples to use
    if num_samples is None or num_samples > len(collected_samples):
        param_samples = collected_samples
    else:
        # Use the last N samples (most converged)
        param_samples = collected_samples[-num_samples:]

    print(f"Using {len(param_samples)} SGMCMC samples for generation")

    # Encode starting prompt
    context = torch.tensor(encode(start_prompt), dtype=torch.long, device=device).unsqueeze(0)

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
    generated_text = decode(generated_tokens)
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
    texts: Any,  # Can be str or list
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
    num_samples: int,
    model_path: Path,
    save_path: str = "generation_results.json",
    sample_id: Optional[str] = None,
    unc_info: Optional[Dict[str, float]] = None,
    has_collected_samples: bool = False
):
    """
    Unified function to save generation results to a JSON file.

    Supports both standard (non-Bayesian) and Bayesian models.
    Automatically increments sample IDs based on the prefix.

    Args:
        start_prompt: The prompt used to start generation
        texts: Generated text(s) - can be a single string or list of strings
        max_new_tokens: Maximum new tokens parameter
        temperature: Temperature parameter
        top_k: Top-k parameter
        num_samples: Number of samples used
        model_path: Path to the model checkpoint used
        save_path: Path to JSON file (default: "generation_results.json")
        sample_id:  sample ID (e.g., "standard_example", "sgmcmc_example"). If None, will use timestamp
        unc_info: Optional dict with uncertainty metrics (avg_uncertainty, max_uncertainty, num_samples_used)
        has_collected_samples: Whether this generation used collected samples (for Bayesian models)

    Returns:
        sample_id: The ID assigned to this sample
    """
    # Load existing results or create new dict
    try:
        with open(save_path, 'r') as f:
            results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        results = {}

    # Generate sample ID
    if sample_id is None:
        sample_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        pass

    # Create entry for this sample
    results[sample_id] = {
        "generation_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "num_samples": num_samples,
            "model_path": str(model_path),
            "has_collected_samples": has_collected_samples
        },
        "start_prompt": start_prompt
    }

    # Add texts (handle both single string and list)
    if isinstance(texts, list):
        results[sample_id]["texts"] = texts
    else:
        results[sample_id]["text"] = texts

    # Add uncertainty info if provided
    if unc_info is not None:
        results[sample_id]["unc_info"] = {
            "avg_uncertainty": float(unc_info.get('avg_uncertainty', 0.0)),
            "max_uncertainty": float(unc_info.get('max_uncertainty', 0.0)),
            "num_samples_used": unc_info.get('num_samples_used', num_samples)
        }

    # Save updated results
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)

    output = f"Saved generation result with ID: {sample_id} to {save_path}"
    return output


def generate_text_standard(model_path, start_prompt,
                          max_new_tokens=500, temperature=1.0, top_k=None,
                          num_samples=1, device='cpu', meta_path=None):
    """
    Generate text using a standard (non-Bayesian) pretrained model.

    This function loads the model and tokenizer, then generates text without
    uncertainty quantification.

    Args:
        model_path: Path to the pretrained model checkpoint (.pt file)
        start_prompt: Starting text string
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature
        top_k: If set, only sample from top k tokens
        num_samples: Number of independent samples to generate
        device: Device to run generation on ('cpu', 'cuda', 'mps')
        meta_path: Path to meta.pkl file (default: auto-detect from baselines/nanogpt/)

    Returns:
        generated_texts: List of generated text strings (including prompt)

    Example:
        >>> texts = generate_text_standard(
        ...     'checkpoints/baseline/models/ckpt.pt',
        ...     "To be or not to be",
        ...     max_new_tokens=100,
        ...     temperature=0.8,
        ...     top_k=200,
        ...     num_samples=3
        ... )
    """
    from baselines.nanogpt.model import GPT, GPTConfig

    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)

    # Remove unwanted prefixes if any
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    # Load tokenizer
    if meta_path is None:
        # Auto-detect meta.pkl from baselines
        meta_path = Path(META_PATH)

    encode, decode = _load_tokenizer(meta_path)

    generated_texts = []

    with torch.no_grad():
        for i in range(num_samples):
            # Encode starting prompt
            start_ids = encode(start_prompt)
            x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

            # Generate text
            y = model.generate(x, max_new_tokens=max_new_tokens,
                             temperature=temperature, top_k=top_k)

            # Decode generated tokens
            generated_text = decode(y[0].tolist())
            generated_texts.append(generated_text)

    return generated_texts


def load_generation_results(save_path: str = "generation_results.json") -> Dict:
    """Load all generation results from JSON file."""
    try:
        with open(save_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"No results file found at {save_path}")
        return {}
