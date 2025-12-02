import re
import sys
import yaml
import pickle
import random

import torch
import torch.nn.functional as F

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import config

parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

def _import_nanogpt():
    baselines_path = config.BASE_DIR / "baselines"
    if str(baselines_path) not in sys.path:
        sys.path.append(str(baselines_path))
    from nanogpt.model import GPT, GPTConfig
    return GPT, GPTConfig
GPT, GPTConfig = _import_nanogpt()


def load_model(model_path: Optional[Path] = None, device: str = 'cpu') -> Tuple[Any, Dict[str, Any]]:
    GPT, GPTConfig = _import_nanogpt()

    if model_path is None:
        model_path = config.MODEL_PATH

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model_args = checkpoint.get('model_args', {})
    print(f"Model arguments: {model_args}")
    
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    
    state_dict = checkpoint['model']
    
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.to(device)
        
    return model, checkpoint


def load_tokenizer(meta_path: Optional[Path] = None) -> Tuple[Dict[str, int], Dict[int, str]]:
    if meta_path is None:
        meta_path = config.META_PATH
        
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    return meta['stoi'], meta['itos']


def encode(text: str, stoi: Dict[str, int]) -> List[int]:
    for c in text:
        if c not in stoi:
            raise ValueError(f"Character '{c}' not in vocabulary")
    return [stoi[c] for c in text]


def decode(tokens: List[int], itos: Dict[int, str]) -> str:
    for c in tokens:
        if c not in itos:
            raise ValueError(f"Token index '{c}' not in vocabulary")
    return ''.join([itos[i] for i in tokens])


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_shakespeare_dataset(dataset_path: Optional[Path] = None) -> Tuple[Optional[str], List[str], List[str]]:
    if dataset_path is None:
        dataset_path = config.DATASET_PATH
        
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
        print(f"Loaded {len(full_text):,} characters")
    except FileNotFoundError:
        print(f"Dataset not found at {dataset_path}")
        return None, [], []
    
    lines = full_text.split('\n')
    meaningful_lines = [line.strip() for line in lines if line.strip() and len(line.strip()) > 10]
    
    prompts = []
    references = []
    random.seed(42)
    
    num_test_samples = 20
    segment_length = 4
    prompt_length = 2
    
    if len(meaningful_lines) < num_test_samples * segment_length:
        num_test_samples = len(meaningful_lines) // segment_length
    
    available_starts = list(range(0, len(meaningful_lines) - segment_length + 1, segment_length))
    selected_starts = random.sample(available_starts, min(num_test_samples, len(available_starts)))
    
    for start_idx in selected_starts:
        segment = meaningful_lines[start_idx:start_idx + segment_length]
        prompt = ' '.join(segment[:prompt_length])
        reference = ' '.join(segment[prompt_length:])
        
        prompt = re.sub(r'\s+', ' ', prompt).strip()
        reference = re.sub(r'\s+', ' ', reference).strip()
        
        if len(prompt) > 20 and len(reference) > 20:
            prompts.append(prompt)
            references.append(reference)
    
    print(f"Created {len(prompts)} test samples")
    return full_text, prompts, references


def extract_character_dialogues(text: str, num_samples: int = 15) -> Tuple[List[str], List[str]]:
    lines = text.split('\n')
    dialogues = []
    current_speaker = None
    current_dialogue = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if ':' in line and len(line.split(':')[0].split()) <= 3:
            if current_speaker and current_dialogue:
                full_dialogue = ' '.join(current_dialogue)
                if len(full_dialogue) > 30:
                    dialogues.append((current_speaker, full_dialogue))
            
            current_speaker = line.split(':')[0].strip()
            remaining = ':'.join(line.split(':')[1:]).strip()
            current_dialogue = [remaining] if remaining else []
        else:
            if current_speaker:
                current_dialogue.append(line)
    
    if current_speaker and current_dialogue:
        full_dialogue = ' '.join(current_dialogue)
        if len(full_dialogue) > 30:
            dialogues.append((current_speaker, full_dialogue))
    
    random.seed(42)
    sampled = random.sample(dialogues, min(num_samples, len(dialogues)))
    
    prompts = []
    references = []
    
    for speaker, dialogue in sampled:
        words = dialogue.split()
        if len(words) > 8:
            prompt_words = min(5, len(words) // 2)
            prompt = f"{speaker}: {' '.join(words[:prompt_words])}"
            reference = ' '.join(words[prompt_words:])
            prompts.append(prompt)
            references.append(reference)
    
    return prompts, references


def clean_generated_text(text: str, prompt: str) -> str:
    generated = text[len(prompt):].strip() if text.startswith(prompt) else text.strip()
    generated = re.sub(r'\s+', ' ', generated)
    
    sentences = generated.split('.')
    if sentences:
        result = sentences[0].strip()
        if len(result) < 15 and len(sentences) > 1:
            result = '.'.join(sentences[:2]).strip()
        return result
    
    words = generated.split()
    return ' '.join(words[:20]) if len(words) > 20 else generated


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
    
    encoded_prompt = encode(prompt, stoi)
    x = torch.tensor(encoded_prompt, dtype=torch.long, device=device)[None, ...]
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            x_cond = x if x.size(1) <= model.config.block_size else x[:, -model.config.block_size:]
            logits, _ = model(x_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_token), dim=1)
    
    return decode(x[0].tolist(), itos)