"""NanoGPT Evaluation Utilities

Provides a reusable evaluator class and helper to run evaluation across splits.
Import from scripts or notebooks:

    from evaluation.nanogpt_evaluator import NanoGPTEvaluator, evaluate_splits

Typical usage:

    config = {
        'data_dir': 'nanoGPT/data/shakespeare_char',
        'model_path': 'checkpoints/baseline_nanogpt/baseline_nanogpt.pt',
        'meta_path': 'checkpoints/baseline_nanogpt/meta.pkl',
        'device': 'auto',
        'splits': ['val', 'train'],
        'batch_size': 32,
        'max_eval_samples': 100,
        'num_text_samples': 10,
        'prompt_length': 20,
        'generation_length': 30,
        'max_tokens': None,
    }

    evaluator = NanoGPTEvaluator(config['model_path'], config['meta_path'], config['device'])
    results = evaluate_splits(evaluator, config)

Results is a dict keyed by split with metrics (perplexity, bleu, rouge1, rouge2, rougeL, total_tokens).
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

try:
    import evaluate as hf_evaluate  # HuggingFace evaluate
except ImportError:  # pragma: no cover
    hf_evaluate = None

from src.nanogpt_utils import load_model, load_tokenizer, decode

__all__ = [
    'NanoGPTEvaluator',
    'evaluate_splits',
]


@dataclass
class EvalResult:
    split: str
    total_tokens: int
    perplexity: float
    bleu: float
    rouge1: float
    rouge2: float
    rougeL: float


class NanoGPTEvaluator:
    """Evaluator for NanoGPT models (character or token level)."""

    def __init__(self, model_path: str | Path, meta_path: str | Path, device: str = 'auto'):
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.model, self.checkpoint = load_model(Path(model_path), self.device)
        self.stoi, self.itos = load_tokenizer(Path(meta_path))
        self.vocab_size = len(self.itos)
        self.model.eval()

        # Metrics (skip if evaluate not installed)
        if hf_evaluate is not None:
            try:
                self.bleu_metric = hf_evaluate.load('bleu')
                self.rouge_metric = hf_evaluate.load('rouge')
                self.perplexity_metric = hf_evaluate.load('perplexity', module_type='metric')
            except Exception:  # pragma: no cover
                self.bleu_metric = None
                self.rouge_metric = None
                self.perplexity_metric = None
        else:
            self.bleu_metric = None
            self.rouge_metric = None
            self.perplexity_metric = None

    # ------------------------------- Data Loading --------------------------- #
    def load_data(self, data_dir: str | Path, split: str = 'val', max_tokens: Optional[int] = None) -> np.ndarray:
        filename = f'{split}.bin'
        filepath = Path(data_dir) / filename
        if not filepath.exists():
            raise FileNotFoundError(f'Data file not found: {filepath}')
        if max_tokens is not None:
            return np.memmap(str(filepath), dtype=np.uint16, mode='r', shape=(max_tokens,))
        return np.memmap(str(filepath), dtype=np.uint16, mode='r')

    # ------------------------------- Tokenizer ------------------------------ #
    def get_tokenizer(self):
        def custom_tokenizer(text: str) -> List[str]:
            return [ch for ch in text if ch in self.stoi]
        return custom_tokenizer

    # ------------------------------- Perplexity ----------------------------- #
    def calculate_perplexity(self, data: np.ndarray, batch_size: int = 16, max_batches: int = 50) -> Optional[float]:
        if self.perplexity_metric is None:
            return None
        seq_len = min(self.model.config.block_size, 256)
        max_start = len(data) - seq_len
        if max_start <= 0:
            return None
        num_batches = min(max_batches, max_start // batch_size) or 1
        texts: List[str] = []
        stride = max(1, max_start // (num_batches * batch_size))
        for i in range(num_batches * batch_size):
            start_idx = i * stride
            end_idx = start_idx + seq_len
            if end_idx <= len(data):
                tokens = data[start_idx:end_idx].astype(np.int64)
                text = decode(tokens.tolist(), self.itos).strip()
                if text:
                    texts.append(text)
        if not texts:
            return None
        try:
            result = self.perplexity_metric.compute(predictions=texts, model_id='gpt2')
            return float(result.get('mean_perplexity', None))
        except Exception:  # pragma: no cover
            return None

    # ------------------------------- Generation ---------------------------- #
    def generate_samples_for_metrics(
        self,
        data: np.ndarray,
        num_samples: int = 20,
        prompt_length: int = 20,
        generation_length: int = 30,
    ) -> Tuple[List[str], List[str]]:
        if len(data) < prompt_length + 5:
            return [], []
        max_possible = max(1, (len(data) - prompt_length - generation_length) // 100)
        num_samples = min(num_samples, max_possible)
        references, predictions = [], []
        for _ in range(num_samples):
            if len(data) > prompt_length + generation_length + 10:
                start_idx = np.random.randint(0, len(data) - prompt_length - generation_length - 10)
            else:
                start_idx = 0
            prompt_tokens = data[start_idx:start_idx + prompt_length].astype(np.int64)
            reference_tokens = data[start_idx + prompt_length:start_idx + prompt_length + generation_length].astype(np.int64)
            reference_text = decode(reference_tokens.tolist(), self.itos).strip()
            x = torch.tensor(prompt_tokens, dtype=torch.long, device=self.device)[None, ...]
            generated_tokens: List[int] = []
            with torch.no_grad():
                for _ in range(generation_length):
                    x_cond = x if x.size(1) <= self.model.config.block_size else x[:, -self.model.config.block_size:]
                    logits, _ = self.model(x_cond)
                    logits = logits[:, -1, :] / 0.8
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    generated_tokens.append(next_token.item())
                    x = torch.cat((x, next_token), dim=1)
            prediction_text = decode(generated_tokens, self.itos).strip()
            if reference_text and prediction_text:
                references.append(reference_text)
                predictions.append(prediction_text)
        return references, predictions

    # ------------------------------- BLEU / ROUGE -------------------------- #
    def calculate_bleu_score(self, references: List[str], predictions: List[str]) -> Optional[float]:
        if self.bleu_metric is None or not references or not predictions:
            return None
        try:
            tokenizer = self.get_tokenizer()
            formatted_refs = [[r] for r in references]
            result = self.bleu_metric.compute(predictions=predictions, references=formatted_refs, tokenizer=tokenizer)
            return float(result.get('bleu', 0.0))
        except Exception:  # pragma: no cover
            return None

    def calculate_rouge_score(self, references: List[str], predictions: List[str]) -> Dict[str, float]:
        if self.rouge_metric is None or not references or not predictions:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        try:
            tokenizer = self.get_tokenizer()
            result = self.rouge_metric.compute(predictions=predictions, references=references, tokenizer=tokenizer)
            return {
                'rouge1': float(result.get('rouge1', 0.0)),
                'rouge2': float(result.get('rouge2', 0.0)),
                'rougeL': float(result.get('rougeL', 0.0)),
            }
        except Exception:  # pragma: no cover
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

    # ------------------------------- Main Eval ----------------------------- #
    def evaluate_dataset(
        self,
        data_dir: str | Path,
        split: str = 'val',
        batch_size: int = 16,
        max_eval_samples: int = 1000,
        num_text_samples: int = 50,
        prompt_length: int = 20,
        generation_length: int = 30,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        t0 = time.time()
        data = self.load_data(data_dir, split, max_tokens)
        results: Dict[str, Any] = {'split': split, 'total_tokens': len(data)}

        perplexity = self.calculate_perplexity(data, batch_size, max_batches=min(100, max_eval_samples // batch_size))
        results['perplexity'] = perplexity if perplexity is not None else 0.0

        if len(data) > 100:
            refs, preds = self.generate_samples_for_metrics(data, num_text_samples, prompt_length, generation_length)
            if refs and preds:
                bleu = self.calculate_bleu_score(refs, preds)
                results['bleu'] = bleu if bleu is not None else 0.0
                rouge = self.calculate_rouge_score(refs, preds)
                results.update(rouge)
                # Example generations preview
                examples = []
                for i in range(min(3, len(refs))):
                    examples.append({
                        'reference': refs[i][:100],
                        'generated': preds[i][:100],
                    })
                results['examples'] = examples
            else:
                results.update({'bleu': 0.0, 'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0})
        else:
            results.update({'bleu': 0.0, 'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0})

        results['elapsed_seconds'] = round(time.time() - t0, 2)
        return results


# ------------------------------- Helper API ------------------------------- #

def evaluate_splits(evaluator: NanoGPTEvaluator, config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Evaluate all splits in config['splits'] returning dict of split->metrics."""
    all_results: Dict[str, Dict[str, Any]] = {}
    for split in config.get('splits', []):
        try:
            res = evaluator.evaluate_dataset(
                config['data_dir'],
                split=split,
                batch_size=config.get('batch_size', 16),
                max_eval_samples=config.get('max_eval_samples', 1000),
                num_text_samples=config.get('num_text_samples', 50),
                prompt_length=config.get('prompt_length', 20),
                generation_length=config.get('generation_length', 30),
                max_tokens=config.get('max_tokens'),
            )
            all_results[split] = res
        except Exception as e:
            all_results[split] = {'error': str(e), 'traceback': traceback.format_exc()}
    return all_results


if __name__ == '__main__':  # Quick CLI usage
    import json
    import argparse

    parser = argparse.ArgumentParser(description='Quick NanoGPT evaluation CLI')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--meta-path', required=True)
    parser.add_argument('--splits', nargs='*', default=['val'])
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-eval-samples', type=int, default=200)
    parser.add_argument('--num-text-samples', type=int, default=20)
    parser.add_argument('--prompt-length', type=int, default=20)
    parser.add_argument('--generation-length', type=int, default=30)
    parser.add_argument('--max-tokens', type=int, default=None)
    parser.add_argument('--output', type=str, default='evaluation_results.json')
    args = parser.parse_args()

    cfg = {
        'data_dir': args.data_dir,
        'model_path': args.model_path,
        'meta_path': args.meta_path,
        'splits': args.splits,
        'device': args.device,
        'batch_size': args.batch_size,
        'max_eval_samples': args.max_eval_samples,
        'num_text_samples': args.num_text_samples,
        'prompt_length': args.prompt_length,
        'generation_length': args.generation_length,
        'max_tokens': args.max_tokens,
    }

    ev = NanoGPTEvaluator(cfg['model_path'], cfg['meta_path'], cfg['device'])
    results = evaluate_splits(ev, cfg)

    payload = {
        'config': cfg,
        'results': results,
        'vocab_size': ev.vocab_size,
    }
    with open(args.output, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'Wrote {args.output}')
