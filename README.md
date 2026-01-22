# Bayesian Neural Network Text Generation

Bayesian inference methods for text generation using a NanoGPT-based character-level language model trained on Shakespeare text. This project compares SGMCMC samplers (BAOA, SGHMC, SGLD) against a deterministic baseline to evaluate whether Bayesian approaches can improve text generation quality while providing uncertainty estimates.

**Key Finding:** BAOA achieves the best perplexity (−16% vs baseline) and outperforms the deterministic baseline on all LLM-judge metrics (+2.8% quality, +2.2% diversity, +1.4% relevance).

## Repository Structure

```
├── src/                           # Core implementation
│   ├── bayesian_utils.py          # SGMCMC samplers (SGLD, SGHMC, BAOA)
│   ├── generation_utils.py        # Text generation utilities
│   └── nanogpt_utils.py           # Model/tokenizer loading
│
├── scripts/                       # Runnable scripts
│   ├── bayesian_training_script.py  # Main training entry point
│   ├── bayesian_evaluator.py      # BLEU/ROUGE/Perplexity evaluation
│   ├── llm_evaluation.py          # LLM-judge evaluation
│   └── llm_evaluation_parallel.py # Parallel LLM evaluation
│
├── notebooks/                     # Jupyter notebooks
│   ├── generation_pipeline.ipynb  # Text generation workflow
│   ├── mcmc_training_colab.ipynb  # MCMC training notebook
│   ├── comparison_report_step_size.ipynb  # Step size analysis
│   └── blue_rouge_perplexity_eval.ipynb   # Metrics analysis
│
├── external/nanogpt/
│   └── model.py                   # NanoGPT architecture (from Karpathy)
│
├── data/
│   ├── raw/                       # Raw text data (train/val .txt)
│   └── tokenized/                 # Tokenized data (train/val .bin)
│
├── checkpoints/
│   ├── baseline/models/           # Deterministic baseline model
│   └── samplers/                  # Trained Bayesian models
│       ├── sghmc_sampler/
│       └── baoa_sampler/
│
├── results/
│   ├── final_report.md            # Evaluation report with findings
│   ├── generation_outputs/        # Generated text samples
│   └── evaluation/
│       ├── automatic_metrics/     # BLEU/ROUGE/Perplexity results
│       │   └── figures/           # Metric visualizations
│       └── llm_judge/             # LLM-judge evaluation results
│
├── config.py                      # Hyperparameter configurations
└── requirements.txt
```

## Setup

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment

Create a `.env` file:

```bash
BASE_DIR=/path/to/your/project
MODEL_PATH=${BASE_DIR}/checkpoints/baseline/models/baseline_model_2k.pt
META_PATH=${BASE_DIR}/checkpoints/baseline/models/meta.pkl
DATA_DIR=${BASE_DIR}/data/tokenized
DEVICE="cuda"  # or "cpu"
WANDB_AVAILABLE="true"  # or "false"
```

### Data

Shakespeare data: [TinyShakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)

## Training

### Basic Usage

```bash
python scripts/bayesian_training_script.py --sampler <SAMPLER_NAME>
```

Where `<SAMPLER_NAME>` is one of: `sgld`, `sghmc`, `baoa`

### Advanced Options

```bash
python scripts/bayesian_training_script.py \
  --sampler baoa \
  --learning-rate 1e-6 \
  --batch-size 16 \
  --train-samples 10000
```

Key parameters:
- `--sampler`: Bayesian inference method (`sgld`, `sghmc`, `baoa`)
- `--learning-rate`: Step size (critical for MCMC performance)
- `--no-wandb`: Disable Weights & Biases logging

## Configuration

Sampler hyperparameters in [config.py](config.py):

```python
CONFIG_BAOA = {
    'learning_rate': 1e-6,      # Step size
    'baoa_alpha': 0.01,         # Momentum decay
    'warmup_steps': 200,        # Burn-in period
    'sampling_steps': 1000,     # Sampling iterations
    'thinning': 10,             # Collect every Nth sample
}
```

## Results

See [results/final_report.md](results/final_report.md) for the complete evaluation.

### Automatic Metrics (BLEU/ROUGE/Perplexity)

| Model | Step Size | BLEU | ROUGE-2 | Perplexity |
|-------|-----------|------|---------|------------|
| Baseline | N/A | **0.258** | **0.523** | 125.6 |
| **BAOA** | 5e-06 | 0.256 | 0.515 | **105.7** |
| SGHMC | 5e-06 | 0.252 | 0.513 | 122.4 |

Model families are close on BLEU/ROUGE (within ~2%). BAOA achieves the best perplexity (−16%).

### LLM-Judge Metrics (Quality/Diversity/Relevance)

| Model | vs Baseline | Verdict |
|-------|-------------|---------|
| **BAOA @ 1e-06** | **+2.8% quality, +2.2% diversity, +1.4% relevance** | **Recommended** |
| SGHMC @ 5e-06 | +0.8% quality, +2.6% diversity, -4.7% relevance | Mixed |
| SGHMC @ 1e-05 | -7.9% quality, -13.8% relevance | Avoid |

### Recommended Configuration

- **Model:** BAOA with learning rate 1e-06
- **Generation:** temp=0.3, top_k=10, samples=10

## Evaluation

Two evaluation approaches:

1. **Automatic metrics** (BLEU, ROUGE, Perplexity) - via `scripts/bayesian_evaluator.py`
2. **LLM-judge** (Quality, Diversity, Relevance) - via `scripts/llm_evaluation.py`

Results are in [results/evaluation/](results/evaluation/).

## References

- [Posteriors Package](https://normal-computing.github.io/posteriors/)
- [NanoGPT](https://github.com/karpathy/nanoGPT)
- [TinyShakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)

## License

MIT License. See [LICENSE](LICENSE).
