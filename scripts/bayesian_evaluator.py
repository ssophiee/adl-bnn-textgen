import time
import math
import numpy as np
import evaluate as hf_evaluate
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import func
import torch.nn.functional as F

import posteriors
from src.nanogpt_utils import decode

__all__ = [
    'BayesianNanoGPTEvaluator',
    'evaluate_bayesian_splits',
]


class BayesianNanoGPTEvaluator:
    def __init__(
        self,
        model,
        stoi: dict,
        itos: dict,
        sampler_type: str,
        state=None,
        collected_samples: Optional[List[Dict]] = None,
        device: str = 'auto',
        num_posterior_samples: int = 10,
    ):
        """
        Initialize Bayesian evaluator.

        Args:
            model: NanoGPT model instance
            stoi: Character to index mapping
            itos: Index to character mapping
            sampler_type: One of 'vi', 'ekf', 'laplace', 'sgld', 'sghmc', 'baoa'
            state: State object for VI-based samplers (VI, EKF, Laplace)
            collected_samples: List of parameter dicts for SGMCMC samplers
            device: Device to use ('auto', 'cuda', 'cpu', 'mps')
            num_posterior_samples: Number of posterior samples for generation
        """
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.model = model.to(self.device)
        self.model.eval()
        self.stoi = stoi
        self.itos = itos
        self.vocab_size = len(itos)
        self.sampler_type = sampler_type
        self.state = state
        self.collected_samples = collected_samples
        self.num_posterior_samples = num_posterior_samples

        if sampler_type in ['vi', 'ekf', 'laplace'] and state is None:
            raise ValueError(f"state required for {sampler_type} sampler")
        if sampler_type in ['sgld', 'sghmc', 'baoa'] and not collected_samples:
            raise ValueError(f"collected_samples required for {sampler_type} sampler")

        self.bleu_metric = None
        self.rouge_metric = None
        self.perplexity_metric = None
        if hf_evaluate is not None:
            try:
                self.bleu_metric = hf_evaluate.load('bleu')
                self.rouge_metric = hf_evaluate.load('rouge')
                self.perplexity_metric = hf_evaluate.load('perplexity', module_type='metric')
            except Exception:
                pass

    def load_data(self, data_dir: str, split: str = 'val', max_tokens: Optional[int] = None) -> np.ndarray:
        from pathlib import Path as P
        filename = f'{split}.bin'
        filepath = P(data_dir) / filename
        if not filepath.exists():
            raise FileNotFoundError(f'Data file not found: {filepath}')
        if max_tokens is not None:
            return np.memmap(str(filepath), dtype=np.uint16, mode='r', shape=(max_tokens,))
        return np.memmap(str(filepath), dtype=np.uint16, mode='r')

    def get_tokenizer(self):
        def custom_tokenizer(text: str) -> List[str]:
            return [ch for ch in text if ch in self.stoi]
        return custom_tokenizer

    def sample_parameters(self) -> List[Dict[str, torch.Tensor]]:
        if self.sampler_type in ['vi', 'ekf', 'laplace']:
            samples = []
            for _ in range(self.num_posterior_samples):
                if self.sampler_type == 'vi':
                    sampled_params = posteriors.vi.diag.sample(self.state)
                elif self.sampler_type == 'ekf':
                    sampled_params = posteriors.ekf.diag_fisher.sample(self.state)
                elif self.sampler_type == 'laplace':
                    sampled_params = posteriors.laplace.diag_fisher.sample(self.state)
                samples.append(sampled_params)
            return samples

        elif self.sampler_type in ['sgld', 'sghmc', 'baoa']:
            if not self.collected_samples:
                raise ValueError("No collected samples available for SGMCMC evaluation")
            
            num_available = len(self.collected_samples)
            num_to_use = min(self.num_posterior_samples, num_available)
            return self.collected_samples[-num_to_use:]

        else:
            raise ValueError(f"Unknown sampler type: {self.sampler_type}")

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

        param_samples = self.sample_parameters()

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

                    logits_list = []
                    for params in param_samples:
                        logits, _ = func.functional_call(self.model, params, (x_cond,))
                        logits_list.append(logits[:, -1, :])

                    avg_logits = torch.stack(logits_list).mean(dim=0)
                    avg_logits = avg_logits / 0.8

                    probs = F.softmax(avg_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    generated_tokens.append(next_token.item())
                    x = torch.cat((x, next_token), dim=1)

            prediction_text = decode(generated_tokens, self.itos).strip()

            if reference_text and prediction_text:
                references.append(reference_text)
                predictions.append(prediction_text)

        return references, predictions

    def calculate_perplexity_internal(
        self,
        data: np.ndarray,
        batch_size: int = 16,
        max_batches: int = 50,
    ) -> Optional[float]:
        """Compute perplexity under *this* model using the training tokenizer.

        Note: For Bayesian models, we estimate the expected NLL by averaging the
        per-sample cross-entropy loss over posterior samples (Monte Carlo), then
        exponentiating.
        """
        if len(data) < 3:
            return None

        # We need seq_len+1 tokens to form (x, y) pairs of length seq_len.
        seq_len = min(int(self.model.config.block_size), 256)
        window_len = seq_len + 1
        max_start = len(data) - window_len
        if max_start <= 0:
            return None

        num_batches = min(max_batches, max_start // batch_size) or 1
        stride = max(1, max_start // (num_batches * batch_size))

        param_samples = self.sample_parameters()
        if not param_samples:
            return None

        total_nll = 0.0
        total_tokens = 0

        self.model.eval()
        with torch.no_grad():
            for b in range(num_batches):
                batch_x = []
                batch_y = []
                for j in range(batch_size):
                    i = b * batch_size + j
                    start_idx = i * stride
                    end_idx = start_idx + window_len
                    if end_idx > len(data):
                        continue
                    window = data[start_idx:end_idx].astype(np.int64)
                    x = torch.from_numpy(window[:-1])
                    y = torch.from_numpy(window[1:])
                    batch_x.append(x)
                    batch_y.append(y)

                if not batch_x:
                    continue

                x_t = torch.stack(batch_x, dim=0).to(self.device)
                y_t = torch.stack(batch_y, dim=0).to(self.device)

                losses = []
                for params in param_samples:
                    _, loss = func.functional_call(self.model, params, (x_t, y_t))
                    if loss is not None:
                        losses.append(loss)

                if not losses:
                    continue

                # Model loss is mean cross-entropy over all tokens in the batch.
                mean_loss = torch.stack(losses).mean()
                num_tokens_batch = int(y_t.numel())
                total_nll += float(mean_loss.detach().cpu().double()) * num_tokens_batch
                total_tokens += num_tokens_batch

        if total_tokens == 0:
            return None
        return float(math.exp(total_nll / total_tokens))

    def calculate_perplexity_external_gpt2(
        self,
        data: np.ndarray,
        batch_size: int = 16,
        max_batches: int = 50,
    ) -> Optional[float]:
        """Compute an *external* perplexity using HF's GPT-2 metric (BPE tokenizer)."""
        if self.perplexity_metric is None:
            return None

        seq_len = min(int(self.model.config.block_size), 256)
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
        except Exception:
            return None

    def calculate_bleu_score(self, references: List[str], predictions: List[str]) -> Optional[float]:
        if self.bleu_metric is None or not references or not predictions:
            return None
        try:
            tokenizer = self.get_tokenizer()
            formatted_refs = [[r] for r in references]
            result = self.bleu_metric.compute(
                predictions=predictions,
                references=formatted_refs,
                tokenizer=tokenizer
            )
            return float(result.get('bleu', 0.0))
        except Exception:
            return None

    def calculate_rouge_score(self, references: List[str], predictions: List[str]) -> Dict[str, float]:
        if self.rouge_metric is None or not references or not predictions:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        try:
            tokenizer = self.get_tokenizer()
            result = self.rouge_metric.compute(
                predictions=predictions,
                references=references,
                tokenizer=tokenizer
            )
            return {
                'rouge1': float(result.get('rouge1', 0.0)),
                'rouge2': float(result.get('rouge2', 0.0)),
                'rougeL': float(result.get('rougeL', 0.0)),
            }
        except Exception:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

    def evaluate_dataset(
        self,
        data_dir: str,
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
        results: Dict[str, Any] = {
            'split': split,
            'total_tokens': len(data),
            'sampler_type': self.sampler_type,
            'num_posterior_samples': self.num_posterior_samples,
        }

        max_ppl_batches = min(100, max_eval_samples // batch_size)
        # Canonical: perplexity under our model + training tokenizer.
        ppl_internal = self.calculate_perplexity_internal(data, batch_size, max_batches=max_ppl_batches)
        results['perplexity'] = ppl_internal if ppl_internal is not None else 0.0

        # Additional: external GPT-2 (BPE) perplexity for rough comparability.
        ppl_external_gpt2 = self.calculate_perplexity_external_gpt2(data, batch_size, max_batches=max_ppl_batches)
        results['perplexity_external_gpt2'] = ppl_external_gpt2 if ppl_external_gpt2 is not None else 0.0

        if len(data) > 100:
            refs, preds = self.generate_samples_for_metrics(
                data,
                num_text_samples,
                prompt_length,
                generation_length
            )
            if refs and preds:
                bleu = self.calculate_bleu_score(refs, preds)
                results['bleu'] = bleu if bleu is not None else 0.0
                rouge = self.calculate_rouge_score(refs, preds)
                results.update(rouge)

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


def evaluate_bayesian_splits(
    evaluator: BayesianNanoGPTEvaluator,
    config: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    import traceback

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
            all_results[split] = {
                'error': str(e),
                'traceback': traceback.format_exc()
            }

    return all_results
