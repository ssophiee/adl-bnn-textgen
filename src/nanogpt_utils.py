"""
Utility functions for NanoGPT model evaluation and Bayesian inference.
"""

import torch
import torch.nn.functional as F
import pickle
import yaml
import re
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# Add the parent directory to the path to import config
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

# Import config from parent directory
import config

# Add the baselines directory to path so we can import the model
def _import_nanogpt():
    baselines_path = config.BASE_DIR / "baselines"
    if str(baselines_path) not in sys.path:
        sys.path.append(str(baselines_path))
    from nanogpt.model import GPT, GPTConfig
    return GPT, GPTConfig

# Import NanoGPT classes
GPT, GPTConfig = _import_nanogpt()


def load_model(model_path: Optional[Path] = None, device: str = 'cpu') -> Tuple[Any, Dict[str, Any]]:
    """
    Load the trained NanoGPT model from checkpoint.

    Args:
        model_path: Path to the model checkpoint file (uses config default if None)
        device: Device to load the model on ('cpu', 'cuda', or 'mps')

    Returns:
        Tuple of (model, checkpoint_dict)
    """
    GPT, GPTConfig = _import_nanogpt()

    if model_path is None:
        model_path = config.MODEL_PATH

    # Load checkpoint - handle MPS device compatibility
    # First load to CPU to avoid MPS device management issues
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Get model configuration
    model_args = checkpoint.get('model_args', {})
    print(f"Model arguments: {model_args}")
    
    # Create model configuration
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    
    # Load state dict
    state_dict = checkpoint['model']
    
    # Remove unwanted prefixes if any
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.to(device)
        
    return model, checkpoint


def load_tokenizer(meta_path: Optional[Path] = None) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Load the character-level tokenizer from meta file.
    
    Args:
        meta_path: Path to the meta pickle file (uses config default if None)
        
    Returns:
        Tuple of (stoi, itos) dictionaries
    """
    if meta_path is None:
        meta_path = config.META_PATH
        
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    return meta['stoi'], meta['itos']


def encode(text: str, stoi: Dict[str, int]) -> List[int]:
    """
    Encode text to token indices using character-level tokenizer.
    
    Args:
        text: Input text string
        stoi: String-to-index mapping dictionary
        
    Returns:
        List of token indices
    """
    for c in text:
        if c not in stoi:
            raise ValueError(f"Character '{c}' not in vocabulary")
        
    return [stoi[c] for c in text]


def decode(tokens: List[int], itos: Dict[int, str]) -> str:
    """
    Decode token indices to text using character-level tokenizer.
    
    Args:
        tokens: List of token indices
        itos: Index-to-string mapping dictionary
        
    Returns:
        Decoded text string
    """
    for c in tokens:
        if c not in itos:
            raise ValueError(f"Token index '{c}' not in vocabulary")
    return ''.join([itos[i] for i in tokens])


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to the YAML config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_shakespeare_dataset(dataset_path: Optional[Path] = None) -> Tuple[Optional[str], List[str], List[str]]:
    """
    Load the Shakespeare dataset and create test prompts and references.
    
    Args:
        dataset_path: Path to the dataset text file (uses config default if None)
        
    Returns:
        Tuple of (full_text, prompts, references)
    """
    if dataset_path is None:
        dataset_path = config.DATASET_PATH
        
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
        print(f"Successfully loaded Shakespeare dataset: {len(full_text):,} characters")
    except FileNotFoundError:
        print(f"Dataset not found at {dataset_path}")
        return None, [], []
    
    # Split the text into meaningful chunks (dialogue segments)
    lines = full_text.split('\n')
    
    # Filter out empty lines and very short lines
    meaningful_lines = [line.strip() for line in lines if line.strip() and len(line.strip()) > 10]
    
    print(f"Found {len(meaningful_lines)} meaningful lines in the dataset")
    
    # Create test segments by taking consecutive lines as references
    prompts = []
    references = []
    
    # Set random seed for reproducible test set
    random.seed(42)
    
    # Sample test segments from different parts of the text
    num_test_samples = 20  # Number of test samples
    segment_length = 4     # Number of lines per segment
    prompt_length = 2      # Number of lines to use as prompt
    
    # Ensure we have enough lines
    if len(meaningful_lines) < num_test_samples * segment_length:
        print("Warning: Not enough lines for desired test samples, reducing sample size")
        num_test_samples = len(meaningful_lines) // segment_length
    
    # Sample segments from throughout the text
    available_starts = list(range(0, len(meaningful_lines) - segment_length + 1, segment_length))
    selected_starts = random.sample(available_starts, min(num_test_samples, len(available_starts)))
    
    for start_idx in selected_starts:
        segment = meaningful_lines[start_idx:start_idx + segment_length]
        
        # Use first part as prompt, rest as reference
        prompt_lines = segment[:prompt_length]
        reference_lines = segment[prompt_length:]
        
        # Join lines properly
        prompt = ' '.join(prompt_lines)
        reference = ' '.join(reference_lines)
        
        # Clean up the text
        prompt = re.sub(r'\s+', ' ', prompt).strip()
        reference = re.sub(r'\s+', ' ', reference).strip()
        
        # Only include if both prompt and reference are meaningful
        if len(prompt) > 20 and len(reference) > 20:
            prompts.append(prompt)
            references.append(reference)
    
    print(f"Created {len(prompts)} test samples from the dataset")
    
    return full_text, prompts, references


def extract_character_dialogues(text: str, num_samples: int = 15) -> Tuple[List[str], List[str]]:
    """
    Extract character dialogues from Shakespeare text for more structured evaluation.
    
    Args:
        text: Full Shakespeare text
        num_samples: Number of dialogue samples to extract
        
    Returns:
        Tuple of (prompts, references) from character dialogues
    """
    lines = text.split('\n')
    dialogues = []
    
    current_speaker = None
    current_dialogue = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if line is a character name (usually ends with :)
        if ':' in line and len(line.split(':')[0].split()) <= 3:
            # Save previous dialogue if it exists
            if current_speaker and current_dialogue:
                full_dialogue = ' '.join(current_dialogue)
                if len(full_dialogue) > 30:  # Only keep substantial dialogues
                    dialogues.append((current_speaker, full_dialogue))
            
            # Start new dialogue
            current_speaker = line.split(':')[0].strip()
            remaining = ':'.join(line.split(':')[1:]).strip()
            current_dialogue = [remaining] if remaining else []
        else:
            # Continue current dialogue
            if current_speaker:
                current_dialogue.append(line)
    
    # Add the last dialogue
    if current_speaker and current_dialogue:
        full_dialogue = ' '.join(current_dialogue)
        if len(full_dialogue) > 30:
            dialogues.append((current_speaker, full_dialogue))
    
    print(f"Extracted {len(dialogues)} character dialogues")
    
    # Sample dialogues for testing
    random.seed(42)
    if len(dialogues) > num_samples:
        sampled_dialogues = random.sample(dialogues, num_samples)
    else:
        sampled_dialogues = dialogues
    
    # Create prompts and references from dialogues
    prompts = []
    references = []
    
    for speaker, dialogue in sampled_dialogues:
        words = dialogue.split()
        if len(words) > 8:  # Ensure dialogue is long enough
            # Use speaker name + first few words as prompt
            prompt_words = min(5, len(words) // 2)
            prompt = f"{speaker}: {' '.join(words[:prompt_words])}"
            reference = ' '.join(words[prompt_words:])
            
            prompts.append(prompt)
            references.append(reference)
    
    return prompts, references


def clean_generated_text(text: str, prompt: str) -> str:
    """
    Clean and extract the generated portion of text.
    
    Args:
        text: Full generated text including prompt
        prompt: Original prompt text to remove
        
    Returns:
        Cleaned generated text
    """
    # Remove the prompt from the beginning
    if text.startswith(prompt):
        generated = text[len(prompt):].strip()
    else:
        generated = text.strip()
    
    # Clean up the text
    generated = re.sub(r'\s+', ' ', generated)
    
    # Extract meaningful portion (stop at reasonable length or natural breaks)
    sentences = generated.split('.')
    if len(sentences) > 0:
        # Take first complete sentence or reasonable chunk
        meaningful_text = sentences[0].strip()
        if len(meaningful_text) < 15 and len(sentences) > 1:
            meaningful_text = '.'.join(sentences[:2]).strip()
    else:
        # Fallback: take first reasonable chunk
        words = generated.split()
        meaningful_text = ' '.join(words[:20]) if len(words) > 20 else generated
    
    return meaningful_text


def generate_text(model, prompt: str, stoi: Dict[str, int], itos: Dict[int, str], 
                 max_new_tokens: int = 50, temperature: float = 0.8, top_k: Optional[int] = None,
                 device: str = 'cpu') -> str:
    """
    Generate text using the NanoGPT model.
    
    Args:
        model: The NanoGPT model
        prompt: Input prompt text
        stoi: String-to-index mapping
        itos: Index-to-string mapping
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        device: Device to run inference on
        
    Returns:
        Generated text including the prompt
    """
    model.eval()
    
    # Encode the prompt
    encoded_prompt = encode(prompt, stoi)
    x = torch.tensor(encoded_prompt, dtype=torch.long, device=device)[None, ...]
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # If sequence is too long, crop it
            x_cond = x if x.size(1) <= model.config.block_size else x[:, -model.config.block_size:]
            
            # Forward pass
            logits, _ = model(x_cond)
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering if specified
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            x = torch.cat((x, next_token), dim=1)
    
    # Decode the generated sequence
    generated_tokens = x[0].tolist()
    generated_text = decode(generated_tokens, itos)
    
    return generated_text


