# Bayesian Neural Network Text Generation

Bayesian inference methods for text generation using a NanoGPT-based character-level language model trained on Shakespeare text. This project compares SGMCMC samplers (BAOA, SGHMC) against a deterministic baseline to evaluate whether Bayesian approaches can improve text generation quality.

**Key Finding:** Individual LLM-judge metrics can be improved over the deterministic baseline — SGHMC with a pretrained-centered prior achieves higher relevance (7.70 vs baseline peak 7.47), and BAOA-ZC-1e6 matches baseline quality and exceeds diversity. However, no configuration wins on all three metrics simultaneously.

> **Open direction:** All experiments use a fixed `prior_std=1.0` (`sd_diag` in the [posteriors](https://normal-computing.github.io/posteriors/) library), which controls the strength of prior regularization. Tuning `prior_std` may be the missing ingredient to allow Bayesian models to outperform the baseline across all metrics at once.

## Repository Structure

```
├── src/                           # Core implementation
│   ├── bayesian_utils.py          # SGMCMC samplers (SGLD, SGHMC, BAOA)
│   ├── generation_utils.py        # Text generation utilities
│   └── nanogpt_utils.py           # Model/tokenizer loading
│
├── scripts/                       # Runnable scripts
│   ├── bayesian_training_script.py  # Main training entry point
│   ├── bayesian_evaluator.py      # BLEU/ROUGE/Perplexity evaluation (Bayesian)
│   ├── nanogpt_evaluator.py       # BLEU/ROUGE/Perplexity evaluation (baseline)
│   ├── llm_evaluation.py          # LLM-judge evaluation
│   └── llm_evaluation_parallel.py # Parallel LLM evaluation
│
├── notebooks/                     # Jupyter notebooks
│   ├── generation_pipeline.ipynb  # Text generation workflow
│   ├── mcmc_training_colab.ipynb  # MCMC training notebook (Colab)
│   ├── nanogpt_training_colab.ipynb  # Baseline training notebook (Colab)
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

### Prior Configuration

All reported results use a **pretrained-centered prior** (`prior_center: 'pretrained'`), meaning the prior mean is set to the pretrained NanoGPT weights rather than zero. This is a form of empirical Bayes that provides better performance by leveraging the pretrained initialization.

Alternative: Set `prior_center: 'zero'` for a standard zero-centered Bayesian prior.

### Sampler Hyperparameters

Sampler hyperparameters in [config.py](config.py):

```python
CONFIG_BAOA = {
    'learning_rate': 1e-6,      # Step size
    'baoa_alpha': 0.01,         # Momentum decay
    'warmup_steps': 200,        # Burn-in period
    'sampling_steps': 1000,     # Sampling iterations
    'thinning': 10,             # Collect every Nth sample
    'prior_std': 1.0,           # Prior standard deviation (sd_diag in posteriors)
}
```

> **Note on `prior_std`:** This maps to `sd_diag` in the [posteriors](https://normal-computing.github.io/posteriors/) library — the diagonal of the prior covariance, controlling the strength of prior regularization. All reported experiments use `prior_std=1.0`. Varying this parameter is a natural next step, particularly for the zero-centered prior where tighter regularization (smaller `prior_std`) may compensate for the lack of a pretrained initialization.

## Results

See [results/final_report.md](results/final_report.md) for the complete evaluation.

### Automatic Metrics (BLEU/ROUGE/Perplexity)

**Note on perplexity:** this repo can report two variants.

- **External GPT-2 perplexity** (`perplexity_external_gpt2`): computed via HuggingFace `evaluate` with `model_id='gpt2'` (GPT-2 BPE tokenizer). This is an *external* reference metric.
- **Internal (training-tokenizer) perplexity** (`perplexity` in the Python evaluators, and `perplexity_internal` in the `blue_rouge_perplexity_eval` notebook): computed under the trained NanoGPT model using the same tokenizer/vocabulary as training (character-level `meta.pkl`).

Unless explicitly labeled as "internal", tables/plots show the GPT-2 external perplexity.

| Model | Step Size | BLEU | ROUGE-2 | Perplexity |
|-------|-----------|------|---------|------------|
| Baseline | N/A | **0.258** | **0.523** | 125.6 |
| **BAOA** | 5e-06 | 0.256 | 0.515 | **105.7** |
| SGHMC | 5e-06 | 0.252 | 0.513 | 122.4 |

Model families are close on BLEU/ROUGE (within ~2%). BAOA achieves the best perplexity (−16%).

### LLM-Judge Metrics (Quality/Diversity/Relevance, 0–10 scale)

Evaluated with Qwen2.5-7B-Instruct across a 2×2×2 design (sampler × prior × LR), 15 prompts × 8 decoding configs per model.

| Model | Prior | LR | Quality | Diversity | Relevance |
|-------|-------|----|---------|-----------|-----------|
| **Baseline** | — | — | **6.13** | **5.77** | 6.21 |
| SGHMC-PP-1e5 | Pretrained | 1e-05 | 6.11 | 5.66 | 6.28 |
| SGHMC-ZC-5e6 | Zero | 5e-06 | 6.13 | 5.65 | 6.25 |
| BAOA-PP-5e6 | Pretrained | 5e-06 | 6.05 | 5.58 | **6.34** |
| SGHMC-PP-5e6 | Pretrained | 5e-06 | 6.00 | 5.40 | 6.40 |
| BAOA-ZC-1e6 | Zero | 1e-06 | 5.98 | **5.70** | 6.05 |
| BAOA-PP-1e6 | Pretrained | 1e-06 | 5.95 | 5.68 | 6.12 |
| BAOA-ZC-5e6 | Zero | 5e-06 | 5.91 | 5.48 | 6.28 |
| SGHMC-ZC-1e5 | Zero | 1e-05 | 5.78 | 5.59 | 5.95 |

**At optimal decoding configs:**

| Goal | Model | Config | Quality | Diversity | Relevance |
|------|-------|--------|---------|-----------|-----------|
| Max relevance | SGHMC-PP-5e6 | temp=0.3, top_k=50, samples=20 | 6.77 | 6.43 | **7.70** |
| Max quality + diversity | BAOA-ZC-1e6 | temp=0.8, top_k=50, samples=20 | **6.83** | **6.60** | 6.97 |
| Baseline reference | Baseline | temp=0.3, top_k=10, samples=20 | 6.83 | 6.57 | 7.47 |

Pretrained-centered prior is the most important design choice. Avoid SGHMC-ZC-1e5 (zero prior + high LR) — largest drop across all metrics.

## Evaluation

Two evaluation approaches:

1. **Automatic metrics** (BLEU, ROUGE, Perplexity) - via `scripts/bayesian_evaluator.py`
2. **LLM-judge** (Quality, Diversity, Relevance) - via `scripts/llm_evaluation.py`

Results are in [results/evaluation/](results/evaluation/).

## AI Usage

This project used AI assistants during development:

- **Claude (Anthropic):** Code review and documentation writing
- **GitHub Copilot:** Code autocompletion during development

All AI-generated code was reviewed and validated by the authors. The core Bayesian inference logic, experimental design, and analysis were performed by the authors.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{bnn_textgen_2025,
  author = {Nikolenko, Sofiia and Tarkhanyan, Aik},
  title = {Bayesian Neural Network Text Generation},
  year = {2025},
  url = {https://github.com/ssophiee/adl-bnn-textgen}
}
```

## References

- [Posteriors Package](https://normal-computing.github.io/posteriors/)
- [NanoGPT](https://github.com/karpathy/nanoGPT)
- [TinyShakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)

## License

MIT License. See [LICENSE](LICENSE).
