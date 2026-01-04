# Bayesian Neural Network Text Generation

This project implements Bayesian inference methods for text generation using a NanoGPT-based character-level language model trained on Shakespeare text.

## Overview

The project supports multiple Bayesian inference samplers:
- **EKF** (Extended Kalman Filter) - Diagonal Fisher approximation
- **SGLD** (Stochastic Gradient Langevin Dynamics) - MCMC sampler
- **SGHMC** (Stochastic Gradient Hamiltonian Monte Carlo) - MCMC sampler with momentum
- **BAOA** (Bayesian Adaptive Optimization Algorithm) - MCMC sampler with adaptive momentum

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```bash
BASE_DIR=/path/to/your/project
MODEL_PATH=${BASE_DIR}/checkpoints/baseline/models/baseline_model_2k.pt
META_PATH=${BASE_DIR}/checkpoints/baseline/models/meta.pkl
DATA_DIR=${BASE_DIR}/notebooks/nanoGPT
DEVICE="cuda"  # or "cpu"
WANDB_AVAILABLE="true"  # or "false"
```

### 3. Data Source

Shakespeare data: [TinyShakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)

## Training Bayesian Models

### Basic Usage

```bash
python scripts/bayesian_training_script.py --sampler <SAMPLER_NAME>
```

Where `<SAMPLER_NAME>` is one of: `ekf`, `sgld`, `sghmc`, `baoa`

### Training with Evaluation

To train a model and run automatic evaluation:

```bash
python scripts/bayesian_training_script.py --sampler baoa --eval
```

### Advanced Options

```bash
python scripts/bayesian_training_script.py \
  --sampler sghmc \
  --learning-rate 2e-5 \
  --epochs 15 \
  --batch-size 16 \
  --train-samples 10000 \
  --eval \
  --eval-splits val train \
  --eval-num-posterior-samples 10 \
  --eval-max-samples 200 \
  --eval-num-text-samples 20
```

**Key Parameters:**
- `--sampler`: Bayesian inference method (`ekf`, `sgld`, `sghmc`, `baoa`)
- `--learning-rate`: Step size for the sampler (critical for MCMC methods)
- `--epochs`: Number of training epochs (ignored for MCMC samplers)
- `--batch-size`: Training batch size
- `--train-samples`: Number of training samples to use
- `--eval`: Enable post-training evaluation
- `--no-wandb`: Disable Weights & Biases logging

### Running in Background (tmux)

For long training runs, use tmux:

```bash
# Start a new tmux session
tmux new -s bayesian_training

# Inside tmux, run training
python scripts/bayesian_training_script.py --sampler sghmc --eval

# Detach from tmux: Ctrl+B, then D
# Reattach later: tmux attach -t bayesian_training
# Kill session: tmux kill-session -t bayesian_training
```

## Evaluation

The project supports two types of evaluation:

### 1. Automatic Bayesian Evaluation (Internal)

Runs automatically during training when `--eval` flag is used. Computes:
- **Perplexity**: Model uncertainty on held-out data
- **BLEU Score**: N-gram overlap with reference text
- **ROUGE Score**: Recall-oriented text similarity
- **Predictive Entropy**: Uncertainty quantification

Results are saved to:
```
src/evaluation/llm_results/internal_data/*.json
```

### 2. LLM-Based Quality Evaluation (External)

Generates text samples and evaluates them using an LLM judge for:
- **Quality**: Grammatical correctness and coherence
- **Diversity**: Vocabulary and stylistic variety
- **Relevance**: Adherence to Shakespeare style

To run LLM evaluation, first train a model, then use the generated samples.

**Aggregated results location:**
```
src/evaluation/llm_results/external_data/generation_results_aggregated.json
```

This file contains quality scores across different generation configurations (temperature, top-k sampling, etc.)

## Configuration

### Sampler Hyperparameters

Edit [config.py](config.py) to adjust sampler-specific settings:

**SGHMC Configuration:**
```python
CONFIG_SGHMC = {
    'learning_rate': 1e-5,      # Step size
    'sghmc_alpha': 0.1,         # Friction coefficient
    'sghmc_beta': 0.0,          # Noise estimate
    'sghmc_sigma': 1.0,         # Prior std for momenta
    'temperature': 1.0,         # Posterior tempering
    'warmup_steps': 200,        # Burn-in period
    'sampling_steps': 1000,     # Sampling iterations
    'thinning': 10,             # Collect every Nth sample
}
```

**BAOA Configuration:**
```python
CONFIG_BAOA = {
    'learning_rate': 1e-6,      # Step size (smaller = more stable)
    'baoa_alpha': 0.01,         # Momentum decay
    'baoa_sigma': 1.0,          # Prior std for momenta
    'temperature': 1.0,
    'warmup_steps': 200,
    'sampling_steps': 1000,
    'thinning': 10,
}
```

### Tuning Step Size (Learning Rate)

The step size is critical for MCMC sampler performance. To experiment:

**Option 1: Command line override**
```bash
python scripts/bayesian_training_script.py --sampler sghmc --learning-rate 2e-5 --eval
```

**Option 2: Edit config.py**
```python
CONFIG_SGHMC = {
    'learning_rate': 2e-5,  # Change this value
    ...
}
```

## Results Directory Structure

```
checkpoints/
├── baseline/
│   └── models/
│       ├── baseline_model_2k.pt      # Deterministic model
│       └── meta.pkl                   # Tokenizer metadata
└── samplers/
    ├── sghmc_sampler/
    │   └── run_<timestamp>/
    │       ├── sghmc_model.pt         # Trained model + samples
    │       ├── sghmc_metrics.json     # Training metrics
    │       ├── sghmc_summary.txt      # Human-readable summary
    │       └── bayesian_eval_*.json   # Evaluation results
    ├── baoa_sampler/
    │   └── run_<timestamp>/
    │       └── ...
    └── vi_sampler/
        └── run_<timestamp>/
            └── ...

src/evaluation/llm_results/
├── external_data/
│   └── generation_results_aggregated.json  # LLM quality scores
└── internal_data/
    └── *.json                              # Internal evaluation data
```

## Example Workflow

### 1. Train with Different Step Sizes

```bash
python scripts/bayesian_training_script.py --sampler baoa --learning-rate 1e-5 --eval
```

### 2. Monitor Results

Check the generated files:
```bash
# View training summary
cat checkpoints/samplers/baoa_sampler/run_<timestamp>/baoa_summary.txt

# View Bayesian metrics
cat checkpoints/samplers/baoa_sampler/bayesian_eval_<timestamp>.json

# Compare LLM quality scores
cat src/evaluation/llm_results/external_data/generation_results_aggregated.json
```

## Results

### Key Findings

**Best Model:** BAOA @ LR=1e-06 outperforms deterministic baseline on all metrics
- Quality: +2.8% improvement over baseline
- Diversity: +2.2% improvement over baseline
- Relevance: +1.4% improvement over baseline

**Optimal Generation Config:** temp=0.3, top_k=10, samples=10

**Learning Rate Sensitivity:**
- BAOA is robust across LR range (1e-06 to 5e-06)
- SGHMC is highly sensitive to LR (requires LR=5e-06)

### Analysis Reports

Detailed analysis available in:
- [results/evaluation/analysis_bnn_vs_baseline.md](results/evaluation/analysis_bnn_vs_baseline.md) - Main findings

### Recommendations

**For production use:**
- Use BAOA sampler with LR=1e-06
- Generation settings: temp=0.3, top_k=10, samples=10
- Expected quality: 4.857/10 (vs baseline 4.386/10)

**Avoid:**
- SGHMC with LR > 5e-06 (performance degradation)

## References

- Posteriors Package: [Documentation](https://normal-computing.github.io/posteriors/)
- NanoGPT: Character-level language model [GitHub](https://github.com/karpathy/nanoGPT)
- TinyShakespeare Dataset: [Link](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
