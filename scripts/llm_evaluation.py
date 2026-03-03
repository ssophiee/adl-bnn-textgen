"""
LLM Evaluation Pipeline for Bayesian SGMCMC Models

This module provides a comprehensive evaluation pipeline that:
1. Generates text from multiple Bayesian models with different hyperparameters
2. Uses Qwen2.5-7B [unsloth/Qwen2.5-7B-Instruct-bnb-4bit - colab] as an LLM judge to evaluate generation quality
3. Aggregates results to compare model performance

"""

import json
import time
import uuid
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime
import numpy as np

from src.generation_utils import (
    generate_text_bayesian_sgmcmc, generate_text_standard,
    load_bayesian_model, generate_text_bayesian_from_loaded,
)
from config import DEVICE


class EvaluationConfig:
    """Configuration for the evaluation pipeline."""

    DEFAULT_TEMPERATURE = [0.3, 0.8]
    DEFAULT_TOP_K = [10, 20, 50]
    DEFAULT_NUM_SAMPLES = [10, 20, 30]
    DEFAULT_MAX_NEW_TOKENS = 600
  
    
    def __init__(self,
                 test_prompts: List[str],
                 model_paths: List[str],
                 model_types: Optional[List[str]] = None,
                 change_params: bool = False,
                 output_path: str = "results/generation_outputs/generation_results_testing.json",
                 device: str = DEVICE):
        """
        Initialize evaluation configuration.

        Args:
            test_prompts: List of starting prompts for generation
            model_paths: List of model checkpoint paths to evaluate
            model_types: List of model types ('bayesian' or 'standard') for each model path.
                        If None, assumes all models are 'bayesian'
            change_params: If True, sweep over hyperparameters
            output_path: Path to save generation results
            device: Device for inference ('cpu', 'cuda', 'mps')
        """
        self.test_prompts = test_prompts
        self.model_paths = model_paths
        self.model_types = model_types if model_types is not None else ['bayesian'] * len(model_paths)
        self.change_params = change_params
        self.output_path = output_path
        self.device = device

        # Validate model_types length matches model_paths
        if len(self.model_types) != len(self.model_paths):
            raise ValueError(
                f"Length of model_types ({len(self.model_types)}) must match "
                f"length of model_paths ({len(self.model_paths)})"
            )

        # Set parameter ranges based on change_params
        if change_params:
            self.temperatures = self.DEFAULT_TEMPERATURE
            self.top_k_values = self.DEFAULT_TOP_K
            self.num_samples_values = self.DEFAULT_NUM_SAMPLES
        else:
            # Use single default values
            self.temperatures = [0.3]
            self.top_k_values = [10]
            self.num_samples_values = [10] 

        self.max_new_tokens = self.DEFAULT_MAX_NEW_TOKENS


# =============================================================================
# Resume / Checkpoint Helpers
# =============================================================================

def _combo_key(result: Dict) -> tuple:
    """Return a hashable key that uniquely identifies a generation combo."""
    return (
        str(result.get('model_path', '')),
        str(result.get('prompt', '')),
        result.get('temperature'),
        result.get('top_k'),
        result.get('num_samples'),
    )


# =============================================================================
# Text Generation Phase
# =============================================================================

def generate_all_texts(config: EvaluationConfig,
                       previous_results: Optional[Dict[str, Dict]] = None) -> Dict[str, Dict[str, Any]]:
    """
    Generate texts from all models with all parameter combinations.

    Args:
        config: EvaluationConfig object with pipeline settings

    Returns:
        Dictionary mapping unique_id -> generation result
        Format:
        {
            "unique_id_1": {
                "model_path": str,
                "prompt": str,
                "temperature": float,
                "top_k": int,
                "num_samples": int,
                "generated_text": str,
                "uncertainty_info": dict
            },
            ...
        }
    """
    # Build skip set from previous results
    results = {}
    skip_set = set()
    if previous_results:
        for uid, res in previous_results.items():
            if 'error' not in res:
                skip_set.add(_combo_key(res))
                results[uid] = res  # carry forward

    total_generations = (len(config.model_paths) *
                        len(config.test_prompts) *
                        len(config.temperatures) *
                        len(config.top_k_values) *
                        len(config.num_samples_values))

    print(f"\n{'='*80}")
    print(f"Starting Text Generation Phase")
    print(f"{'='*80}")
    print(f"Total generations to perform: {total_generations}")
    if skip_set:
        print(f"Already completed (resuming): {len(skip_set)}")
        print(f"Remaining: {total_generations - len(skip_set)}")
    print(f"Models: {len(config.model_paths)}")
    print(f"Prompts: {len(config.test_prompts)}")
    print(f"Parameter combinations: {len(config.temperatures) * len(config.top_k_values) * len(config.num_samples_values)}")
    print(f"{'='*80}\n")

    generation_count = 0
    new_generation_count = 0  # tracks unsaved new generations
    completed_count = 0  # actual generations done (not skipped)
    gen_start_time = time.time()
    remaining_generations = total_generations - len(skip_set)

    # Load prompt sources once at the beginning
    prompt_sources = {}
    all_prompts_path = Path(__file__).resolve().parent / "extracted_prompts.json"
    if all_prompts_path.exists():
        with open(all_prompts_path, 'r') as f:
            all_prompts = json.load(f)
        for prompt in all_prompts.get('train_prompts', []):
            prompt_sources[prompt] = 'train'
        for prompt in all_prompts.get('val_prompts', []):
            prompt_sources[prompt] = 'val'

    for model_idx, (model_path, model_type) in enumerate(zip(config.model_paths, config.model_types)):
        print(f"\n--- Processing model {model_idx + 1}/{len(config.model_paths)}: {Path(model_path).name} (type: {model_type}) ---")

        # Check if all generations for this model are already cached
        all_cached = all(
            (str(model_path), prompt, temp, tk, ns) in skip_set
            for prompt in config.test_prompts
            for temp in config.temperatures
            for tk in config.top_k_values
            for ns in config.num_samples_values
        )
        if all_cached:
            n_combos = len(config.test_prompts) * len(config.temperatures) * len(config.top_k_values) * len(config.num_samples_values)
            generation_count += n_combos
            print(f"  All {n_combos} generations cached — skipping model load")
            continue

        # Load model once for all generations of this model
        bayesian_loaded = None
        if model_type == 'bayesian':
            try:
                bayesian_loaded = load_bayesian_model(model_path, device=config.device)
                print(f"  Loaded model with {bayesian_loaded[2]} collected samples (vmap-stacked)")
            except Exception as e:
                print(f"  Failed to load model: {e}")

        for prompt in config.test_prompts:
            prompt_source = prompt_sources.get(prompt, 'unknown')
            print(f"\n  Prompt: '{prompt[:50]}...' [source: {prompt_source}]")

            for temperature in config.temperatures:
                for top_k in config.top_k_values:
                    for num_samples in config.num_samples_values:
                        generation_count += 1

                        # Skip if already completed
                        combo = (str(model_path), prompt, temperature, top_k, num_samples)
                        if combo in skip_set:
                            print(f"    [{generation_count}/{total_generations}] "
                                  f"temp={temperature}, top_k={top_k}, samples={num_samples}... SKIP (cached)")
                            continue

                        # Generate unique ID
                        unique_id = str(uuid.uuid4())

                        print(f"    [{generation_count}/{total_generations}] "
                              f"temp={temperature}, top_k={top_k}, samples={num_samples}... ",
                              end='', flush=True)

                        try:
                            if model_type == 'bayesian':
                                if bayesian_loaded is None:
                                    raise RuntimeError("Model failed to load earlier")
                                model_obj, stacked_params, total_samples, encode, decode = bayesian_loaded
                                generated_text, unc_info = generate_text_bayesian_from_loaded(
                                    model_obj, stacked_params, total_samples,
                                    encode, decode,
                                    start_prompt=prompt,
                                    max_new_tokens=config.max_new_tokens,
                                    temperature=temperature,
                                    top_k=top_k,
                                    num_samples=num_samples,
                                    device=config.device
                                )

                                # Store result
                                results[unique_id] = {
                                    "model_path": str(model_path),
                                    "model_type": model_type,
                                    "prompt": prompt,
                                    "prompt_source": prompt_source,
                                    "temperature": temperature,
                                    "top_k": top_k,
                                    "num_samples": num_samples,
                                    "generated_text": generated_text,
                                    "uncertainty_info": unc_info
                                }

                            elif model_type == 'standard':
                                generated_texts = generate_text_standard(
                                    model_path=model_path,
                                    start_prompt=prompt,
                                    max_new_tokens=config.max_new_tokens,
                                    temperature=temperature,
                                    top_k=top_k,
                                    num_samples=1,
                                    device=config.device
                                )

                                results[unique_id] = {
                                    "model_path": str(model_path),
                                    "model_type": model_type,
                                    "prompt": prompt,
                                    "prompt_source": prompt_source,
                                    "temperature": temperature,
                                    "top_k": top_k,
                                    "num_samples": num_samples,
                                    "generated_text": generated_texts[0],
                                    "uncertainty_info": None
                                }

                            else:
                                raise ValueError(f"Unknown model_type: {model_type}. Must be 'bayesian' or 'standard'")

                            completed_count += 1
                            elapsed = time.time() - gen_start_time
                            avg_time = elapsed / completed_count
                            eta_seconds = avg_time * (remaining_generations - completed_count)
                            eta_min = eta_seconds / 60
                            print(f"✓ ({avg_time:.1f}s/gen, ETA: {eta_min:.1f}min)")

                            # Save checkpoint every 10 new generations
                            new_generation_count += 1
                            if new_generation_count % 10 == 0:
                                save_generation_results(results, config.output_path)
                                print(f"    [checkpoint saved — {new_generation_count} new generations]")

                        except Exception as e:
                            print(f"✗ Error: {str(e)}")
                            results[unique_id] = {
                                "model_path": str(model_path),
                                "model_type": model_type,
                                "prompt": prompt,
                                "prompt_source": prompt_source,
                                "temperature": temperature,
                                "top_k": top_k,
                                "num_samples": num_samples,
                                "error": str(e)
                            }

        # Save checkpoint after finishing all generations for this model
        if new_generation_count > 0:
            save_generation_results(results, config.output_path)
            print(f"  [checkpoint saved after model {model_idx + 1}]")
            new_generation_count = 0

        # Free model memory before loading next model
        del bayesian_loaded
        import torch, gc
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n{'='*80}")
    print(f"Generation Phase Complete: {len([r for r in results.values() if 'error' not in r])}/{total_generations} successful")
    print(f"{'='*80}\n")

    return results


def save_generation_results(results: Dict[str, Dict], output_path: str):
    """
    Save generation results to JSON file.

    Args:
        results: Dictionary of generation results
        output_path: Path to save JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def load_generation_results(input_path: str) -> Dict[str, Dict]:
    """
    Load generation results from JSON file.

    Args:
        input_path: Path to JSON file

    Returns:
        Dictionary of generation results
    """
    with open(input_path, 'r') as f:
        return json.load(f)


# =============================================================================
# LLM Judge Evaluation (Qwen2.5-7B)
# =============================================================================

def prepare_judge_prompt(results: Dict[str, Dict], prompt: str) -> str:
    """
    Prepare evaluation prompt for Qwen3-8B judge.

    Args:
        results: Dictionary of results filtered for a specific prompt
        prompt: The starting prompt being evaluated

    Returns:
        Formatted prompt string for the LLM judge
    """
    prompt_text = f"""You are an expert evaluator of text generation quality. You will evaluate multiple text continuations that start from the same prompt.

Starting Prompt: "{prompt}"

Below are {len(results)} different text continuations. Evaluate each one carefully.

"""

    # Add each text with its unique ID
    for unique_id, result in results.items():
        if 'error' in result:
            continue

        prompt_text += f"""
{'='*80}
Text ID: {unique_id}
{'='*80}
{result['generated_text']}

"""

    prompt_text += """
For each text, evaluate on these criteria:

1. **Quality (0-10)**: Assess coherence, grammar, style, and overall writing quality
   - 0-3: Poor (incoherent, many errors)
   - 4-6: Fair (some issues but readable)
   - 7-8: Good (well-written, minor issues)
   - 9-10: Excellent (professional quality)

2. **Diversity (0-10)**: Assess creativity, uniqueness, and interesting content
   - 0-3: Generic or repetitive
   - 4-6: Somewhat creative
   - 7-8: Creative and interesting
   - 9-10: Highly creative and unique

3. **Relevance (0-10)**: Assess how well it stays on topic from the prompt
   - 0-3: Off-topic or nonsensical
   - 4-6: Loosely related
   - 7-8: Stays on topic
   - 9-10: Perfectly relevant and coherent continuation

Provide your evaluation in the following JSON format:

```json
{
    "unique_id_1": {
        "quality_score": 8.5,
        "diversity_score": 7.0,
        "relevance_score": 9.0,
        "brief_reasoning": "Brief explanation of scores"
    },
    "unique_id_2": {
        ...
    }
}
```

Only output the JSON, nothing else."""

    return prompt_text


def evaluate_with_qwen(generation_results: Dict[str, Dict],
                       model_name: str = "qwen/qwen-2.5-7b-instruct",
                       api_key: Optional[str] = None,
                       use_local: bool = False,
                       scores_checkpoint_path: Optional[str] = None) -> Dict[str, Dict]:
    """
    Use Qwen2.5-7B to score all generated texts.

    This function groups texts by prompt and evaluates them together
    for fair comparison.

    Args:
        generation_results: Dictionary of generation results from generate_all_texts
        model_name: Qwen model to use (default: qwen-2.5-7b-instruct)
        api_key: API key for the model provider (if using API)
        use_local: If True, use local Qwen model via transformers

    Returns:
        Dictionary mapping unique_id -> evaluation scores
        Format:
        {
            "unique_id_1": {
                "quality_score": 8.5,
                "diversity_score": 7.0,
                "relevance_score": 9.0,
                "brief_reasoning": "..."
            },
            ...
        }
    """
    print(f"\n{'='*80}")
    print(f"Starting LLM Judge Evaluation Phase")
    print(f"{'='*80}")
    print(f"Using model: {model_name}")
    print(f"Total texts to evaluate: {len([r for r in generation_results.values() if 'error' not in r])}")
    print(f"{'='*80}\n")

    # Group results by prompt
    results_by_prompt = {}
    for unique_id, result in generation_results.items():
        if 'error' in result:
            continue

        prompt = result['prompt']
        if prompt not in results_by_prompt:
            results_by_prompt[prompt] = {}
        results_by_prompt[prompt][unique_id] = result

    all_scores = {}

    if use_local:
        # Use local transformers-based evaluation
        all_scores = _evaluate_with_local_qwen(results_by_prompt, model_name,
                                                scores_checkpoint_path=scores_checkpoint_path)
    else:
        # Use API-based evaluation (placeholder - needs implementation)
        print("Note: API-based evaluation requires implementation.")
        print("Falling back to local evaluation or mock scores.")
        all_scores = _evaluate_with_mock_scores(results_by_prompt)

    print(f"\n{'='*80}")
    print(f"Evaluation Phase Complete: {len(all_scores)} texts evaluated")
    print(f"{'='*80}\n")

    return all_scores


def _score_batch_with_qwen(batch: Dict[str, Dict], prompt: str,
                           model, tokenizer, torch_module) -> Dict[str, Dict]:
    """
    Score a single sub-batch of texts with Qwen. Returns parsed scores dict.
    Retries once on JSON parse failure with a reminder to output valid JSON.
    """
    judge_prompt = prepare_judge_prompt(batch, prompt)

    messages = [
        {"role": "system", "content": "You are an expert text evaluator. Always respond with valid JSON only."},
        {"role": "user", "content": judge_prompt}
    ]

    # ~100 output tokens per text (scores + brief reasoning)
    max_new_tokens = max(512, len(batch) * 120)

    for attempt in range(2):
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt")
        if torch_module.cuda.is_available():
            model_inputs = model_inputs.to("cuda")

        with torch_module.no_grad():
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                do_sample=True
            )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Parse JSON
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            try:
                scores = json.loads(response[json_start:json_end])
                return scores
            except json.JSONDecodeError:
                pass

        # Retry: append a reminder message
        if attempt == 0:
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": "Your response was not valid JSON. Please output ONLY the JSON object with scores, nothing else."})

    return {}  # both attempts failed


def _evaluate_with_local_qwen(results_by_prompt: Dict[str, Dict],
                               model_name: str,
                               scores_checkpoint_path: Optional[str] = None,
                               batch_size: int = 8) -> Dict[str, Dict]:
    """
    Evaluate texts using local Qwen model via transformers.

    Splits each prompt group into sub-batches to avoid exceeding context
    limits and improve JSON parse reliability.

    Args:
        results_by_prompt: Results grouped by prompt
        model_name: Qwen model identifier
        scores_checkpoint_path: Path to save score checkpoints
        batch_size: Max texts per Qwen call (default 8)

    Returns:
        Dictionary of evaluation scores
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        model.eval()
        print("Model loaded successfully!\n")

        # Count total batches for ETA
        total_batches = sum(
            (len(results) + batch_size - 1) // batch_size
            for results in results_by_prompt.values()
        )
        total_texts = sum(len(r) for r in results_by_prompt.values())
        print(f"Scoring {total_texts} texts in {total_batches} batches (batch_size={batch_size})\n")

        all_scores = {}
        batch_num = 0
        score_start_time = time.time()

        for prompt_idx, (prompt, results) in enumerate(results_by_prompt.items(), 1):
            print(f"Evaluating prompt {prompt_idx}/{len(results_by_prompt)}: '{prompt[:50]}...'")

            # Split into sub-batches
            items = list(results.items())
            for i in range(0, len(items), batch_size):
                sub_batch = dict(items[i:i + batch_size])
                batch_num += 1

                scores = _score_batch_with_qwen(sub_batch, prompt, model, tokenizer, torch)

                if scores:
                    all_scores.update(scores)
                    scored = len(scores)
                else:
                    scored = 0

                # ETA
                elapsed = time.time() - score_start_time
                avg_per_batch = elapsed / batch_num
                eta_min = avg_per_batch * (total_batches - batch_num) / 60

                print(f"  batch {batch_num}/{total_batches}: "
                      f"scored {scored}/{len(sub_batch)} texts "
                      f"({avg_per_batch:.1f}s/batch, ETA: {eta_min:.1f}min)")

                if scored < len(sub_batch):
                    missing = set(sub_batch.keys()) - set(scores.keys()) if scores else set(sub_batch.keys())
                    print(f"    ✗ {len(missing)} texts failed to score")

                # Checkpoint after each batch
                if scores_checkpoint_path and scores:
                    with open(scores_checkpoint_path, 'w') as f:
                        json.dump(all_scores, f, indent=2)

        return all_scores

    except Exception as e:
        print(f"Error loading local model: {e}")
        print("Falling back to mock scores...")
        return _evaluate_with_mock_scores(results_by_prompt)


def _evaluate_with_mock_scores(results_by_prompt: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Generate mock evaluation scores for testing (when real LLM is unavailable).

    Args:
        results_by_prompt: Results grouped by prompt

    Returns:
        Dictionary of mock evaluation scores
    """
    print("\n⚠️  Using mock scores for testing purposes")
    print("For production, implement proper LLM judge integration\n")

    all_scores = {}

    for prompt, results in results_by_prompt.items():
        for unique_id in results.keys():
            # Generate realistic-looking mock scores with some variance
            all_scores[unique_id] = {
                "quality_score": -509,
                "diversity_score": -509,
                "relevance_score": -509,
                "brief_reasoning": "Mock evaluation score (replace with real LLM judge)"
            }

    return all_scores


# =============================================================================
# Results Aggregation and Analysis
# =============================================================================

def aggregate_results(generation_results: Dict[str, Dict],
                     evaluation_scores: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Merge evaluation scores with generation results and compute aggregate statistics.

    Args:
        generation_results: Original generation results
        evaluation_scores: Scores from LLM judge

    Returns:
        Aggregated results per model with breakdown by configuration
        Format:
        {
            "model_path": {
                "overall_avg_quality": float,
                "overall_avg_diversity": float,
                "overall_avg_relevance": float,
                "total_generations": int,
                "by_config": {
                    "temp_0.3_topk_10_samples_10": {
                        "avg_quality": float,
                        "avg_diversity": float,
                        "avg_relevance": float,
                        "count": int
                    },
                    ...
                }
            },
            ...
        }
    """
    print(f"\n{'='*80}")
    print(f"Aggregating Results")
    print(f"{'='*80}\n")

    # First, merge scores into generation results
    merged_results = {}
    for unique_id, gen_result in generation_results.items():
        if 'error' in gen_result:
            continue

        merged_results[unique_id] = {
            **gen_result,
            **evaluation_scores.get(unique_id, {})
        }

    # Now aggregate by model
    model_stats = {}

    for unique_id, result in merged_results.items():
        model_path = result['model_path']

        if model_path not in model_stats:
            model_stats[model_path] = {
                'total_generations': 0,
                'quality_scores': [],
                'diversity_scores': [],
                'relevance_scores': [],
                'by_config': {}
            }

        # Skip if scores are missing
        if 'quality_score' not in result:
            continue

        model_stats[model_path]['total_generations'] += 1
        model_stats[model_path]['quality_scores'].append(result['quality_score'])
        model_stats[model_path]['diversity_scores'].append(result['diversity_score'])
        model_stats[model_path]['relevance_scores'].append(result['relevance_score'])

        # Track by configuration
        config_key = f"temp_{result['temperature']}_topk_{result['top_k']}_samples_{result['num_samples']}"
        if config_key not in model_stats[model_path]['by_config']:
            model_stats[model_path]['by_config'][config_key] = {
                'quality_scores': [],
                'diversity_scores': [],
                'relevance_scores': [],
                'count': 0
            }

        model_stats[model_path]['by_config'][config_key]['quality_scores'].append(result['quality_score'])
        model_stats[model_path]['by_config'][config_key]['diversity_scores'].append(result['diversity_score'])
        model_stats[model_path]['by_config'][config_key]['relevance_scores'].append(result['relevance_score'])
        model_stats[model_path]['by_config'][config_key]['count'] += 1

    # Compute averages
    final_results = {}
    for model_path, stats in model_stats.items():
        final_results[model_path] = {
            'overall_avg_quality': float(np.mean(stats['quality_scores'])),
            'overall_avg_diversity': float(np.mean(stats['diversity_scores'])),
            'overall_avg_relevance': float(np.mean(stats['relevance_scores'])),
            'total_generations': stats['total_generations'],
            'by_config': {}
        }

        # Compute per-config averages
        for config_key, config_stats in stats['by_config'].items():
            final_results[model_path]['by_config'][config_key] = {
                'avg_quality': float(np.mean(config_stats['quality_scores'])),
                'avg_diversity': float(np.mean(config_stats['diversity_scores'])),
                'avg_relevance': float(np.mean(config_stats['relevance_scores'])),
                'count': config_stats['count']
            }

    # Print summary
    print("\n--- Summary Statistics ---\n")
    for model_path, results in final_results.items():
        print(f"Model: {Path(model_path).name}")
        print(f"  Overall Quality: {results['overall_avg_quality']:.2f}")
        print(f"  Overall Diversity: {results['overall_avg_diversity']:.2f}")
        print(f"  Overall Relevance: {results['overall_avg_relevance']:.2f}")
        print(f"  Total Generations: {results['total_generations']}")
        print()

    print(f"{'='*80}\n")

    return final_results, merged_results


# =============================================================================
# Main Pipeline Function
# =============================================================================

def run_evaluation_pipeline(
    test_prompts: Optional[List[str]] = None,
    model_paths: List[str] = None,
    change_params: bool = False,
    output_path: str = "results/generation_outputs/generation_results_testing.json",
    use_local_qwen: bool = False,
    model_types: Optional[List[str]] = None,
    qwen_model: str = "Qwen/Qwen2.5-7B-Instruct",
    device: str = DEVICE,
    use_external_data: bool = True
) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """
    Run the complete evaluation pipeline.

    This is the main entry point that orchestrates:
    1. Text generation from all models
    2. LLM judge evaluation
    3. Results aggregation

    Args:
        test_prompts: List of starting prompts for generation (required if use_external_data=False)
        model_paths: List of model checkpoint paths to evaluate
        change_params: If True, sweep over hyperparameters
        output_path: Path to save generation results JSON
        use_local_qwen: If True, use local Qwen model for evaluation
        qwen_model: Qwen model identifier
        device: Device for inference
        use_external_data: If False, load prompts from extracted_prompts.json; if True, use test_prompts parameter

    Returns:
        Tuple of (aggregated_results, full_results_with_scores)
        - aggregated_results: Average scores per model
        - full_results_with_scores: Complete results with individual scores

    Example:
        >>> test_prompts = ["To be or not to be;", "Once upon a time"]
        >>> model_paths = ["path/to/model1.pt", "path/to/model2.pt"]
        >>> results, full_data = run_evaluation_pipeline(
        ...     test_prompts=test_prompts,
        ...     model_paths=model_paths,
        ...     change_params=True
        ... )
    """
    start_time = datetime.now()
    print(f"\n{'#'*80}")
    print(f"# LLM Evaluation Pipeline")
    print(f"# Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*80}\n")

    # Load prompts based on flag
    if not use_external_data:
        extracted_prompts_path = Path(__file__).resolve().parent / "extracted_prompts.json"
        if extracted_prompts_path.exists():
            with open(extracted_prompts_path, 'r') as f:
                extracted_prompts = json.load(f)
            test_prompts = extracted_prompts['all_prompts']
            print(f"Loaded {len(test_prompts)} prompts from extracted_prompts.json")
            print(f"  Train prompts: {len(extracted_prompts['train_prompts'])}")
            print(f"  Val prompts: {len(extracted_prompts['val_prompts'])}\n")
        else:
            raise FileNotFoundError(f"extracted_prompts.json not found at: {extracted_prompts_path}")
    else:
        if test_prompts is None:
            raise ValueError("test_prompts must be provided when use_external_data=True")
        print(f"Using {len(test_prompts)} provided prompts\n")

    # Initialize configuration
    config = EvaluationConfig(
        test_prompts=test_prompts,
        model_paths=model_paths,
        model_types=model_types,
        change_params=change_params,
        output_path=output_path,
        device=device
    )

    # Auto-resume: load previous generation results if output file exists
    previous_results = None
    output_file = Path(output_path)
    if output_file.exists() and output_file.stat().st_size > 0:
        try:
            previous_results = load_generation_results(str(output_file))
            print(f"Loaded {len(previous_results)} previous results from: {output_file}")
        except Exception as e:
            print(f"Warning: could not load previous results: {e}")

    # Phase 1: Generate all texts (skips already-completed combos)
    generation_results = generate_all_texts(config, previous_results=previous_results)

    # Save generation results
    save_generation_results(generation_results, config.output_path)

    # Phase 2: Evaluate with LLM judge
    # Auto-resume: load previous scores from _with_scores.json or _scores_checkpoint.json
    previous_scores = {}
    for scores_file in [
        config.output_path.replace('.json', '_with_scores.json'),
        config.output_path.replace('.json', '_scores_checkpoint.json'),
    ]:
        scores_path = Path(scores_file)
        if scores_path.exists() and scores_path.stat().st_size > 0:
            try:
                prev_scored = json.loads(scores_path.read_text())
                for uid, res in prev_scored.items():
                    if uid not in previous_scores and 'quality_score' in res:
                        # Skip mock/placeholder scores (-509) so they get re-evaluated
                        if res['quality_score'] == -509 or res['diversity_score'] == -509 or res['relevance_score'] == -509:
                            continue
                        previous_scores[uid] = {
                            'quality_score': res['quality_score'],
                            'diversity_score': res['diversity_score'],
                            'relevance_score': res['relevance_score'],
                            'brief_reasoning': res.get('brief_reasoning', ''),
                        }
                print(f"Loaded {len(previous_scores)} previous scores from: {scores_path}")
            except Exception as e:
                print(f"Warning: could not load previous scores from {scores_path}: {e}")

    # Only send un-scored results to the judge
    new_results = {uid: res for uid, res in generation_results.items()
                   if uid not in previous_scores and 'error' not in res}

    scores_checkpoint = config.output_path.replace('.json', '_scores_checkpoint.json')
    if new_results:
        new_scores = evaluate_with_qwen(
            new_results,
            model_name=qwen_model,
            use_local=use_local_qwen,
            scores_checkpoint_path=scores_checkpoint
        )
    else:
        new_scores = {}
        print("All results already scored — skipping LLM judge phase.")

    evaluation_scores = {**previous_scores, **new_scores}

    # Phase 3: Aggregate results
    aggregated_results, full_results = aggregate_results(
        generation_results,
        evaluation_scores
    )

    # Save final results
    final_output_path = config.output_path.replace('.json', '_with_scores.json')
    with open(final_output_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    print(f"Saved full results with scores to: {final_output_path}")

    aggregated_output_path = config.output_path.replace('.json', '_aggregated.json')
    with open(aggregated_output_path, 'w') as f:
        json.dump(aggregated_results, f, indent=2)
    print(f"Saved aggregated results to: {aggregated_output_path}")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"\n{'#'*80}")
    print(f"# Pipeline Complete!")
    print(f"# Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print(f"# Ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*80}\n")

    return aggregated_results, full_results


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    import glob
    
    # Option 1: provide custom prompts as a list (default)
    external_data = True #  False will load prompts from `extracted_prompts.json`
    
    test_prompts = [
        "to be or not to be;",
        "Once upon a time",
        "In the beginning",
    ]

    # Example: Find recent SGMCMC models
    # Find SGHMC models
    sghmc_models = sorted(glob.glob("checkpoints/samplers/sghmc_sampler/*/sghmc_model.pt"))

    # Find BAOA models
    baoa_models = sorted(glob.glob("checkpoints/samplers/baoa_sampler/*/baoa_model.pt"))

    # Take most recent models
    model_paths = []
    if sghmc_models:
        model_paths.append(sghmc_models[-1])
    if baoa_models:
        model_paths.append(baoa_models[-1])

    print(f"\nFound {len(model_paths)} models to evaluate:")
    for path in model_paths:
        print(f"  - {path}")

    if not model_paths:
        print("\n No models found! Please provide model paths manually.")
        print("Example:")
        print("  model_paths = ['checkpoints/samplers/sghmc_sampler/run_xyz/sghmc_model.pt']")
    else:
        # Run pipeline
        results, full_data = run_evaluation_pipeline(
            test_prompts=test_prompts if external_data else None,
            model_paths=model_paths,
            change_params=False,  # Set to True to sweep hyperparameters
            use_local_qwen=False,  # Set to True to use local Qwen model
            device=DEVICE,
            use_external_data=external_data  # Toggle between file and custom prompts
        )

        print("\n" + "="*80)
        print("Final Results Summary:")
        print("="*80)
        for model_path, model_results in results.items():
            print(f"\nModel: {Path(model_path).name}")
            print(f"  Quality: {model_results['overall_avg_quality']:.2f}")
            print(f"  Diversity: {model_results['overall_avg_diversity']:.2f}")
            print(f"  Relevance: {model_results['overall_avg_relevance']:.2f}")
