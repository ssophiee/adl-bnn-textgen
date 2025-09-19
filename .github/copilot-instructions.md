# Copilot Instructions for Bayesian Neural Networks Text Generation (adl-bnn-textgen)

## Project Overview
This project implements **Bayesian Neural Networks (BNNs)** for text generation using the `posteriors` library for variational inference on NanoGPT models. The goal is to quantify uncertainty in language model predictions through Bayesian approaches.

## Architecture
- **Baseline Models**: Standard NanoGPT implementations in `baselines/nanogpt/`
- **Bayesian Inference**: Uses `posteriors` library for variational inference (VI) with diagonal Gaussian approximations
- **Model Checkpoints**: Pre-trained models stored in `checkpoints/` with corresponding metadata
- **Notebooks**: Jupyter notebooks for training, inference, and evaluation workflows

## Key Components

### Configuration System
- Central configuration in `config.py` with hardcoded paths to checkpoints and datasets
- Model paths: `MODEL_PATH`, `META_PATH`, `CONFIG_PATH`, `DATASET_PATH`
- Device handling: `DEVICE = "cuda" if __name__ == "__main__" else "cpu"`

### Model Structure
- Based on Karpathy's NanoGPT with standard GPT architecture (CausalSelfAttention, LayerNorm, etc.)
- Model configs stored as YAML files in `baselines/nanogpt/shakespeare-char/config.yaml`
- Character-level tokenization using pickle files (`meta.pkl`) with `stoi`/`itos` mappings

### Bayesian Inference Workflow
1. **Load baseline model** from checkpoint (`.pt` files)
2. **Extract parameters** using `dict(model.named_parameters())`
3. **Define log_posterior function** combining negative log-likelihood + log prior
4. **Build VI transform** using `posteriors.vi.diag.build()` with Adam optimizer
5. **Train posterior** through `vi_transform.update()` iterations
6. **Sample parameters** using `posteriors.vi.diag.sample(vi_state)`

### Data Handling
- Shakespeare character data loaded as binary `.bin` files using `np.memmap`
- Batch creation through random sampling of sequences
- Standard sequence length: 128-256 tokens
- Training data path pattern: `nanoGPT/data/shakespeare_char/train.bin`

## Development Workflows

### Training New Models
1. Use Google Colab notebook (`nanogpt_training_colab.ipynb`) for GPU training
2. Copy trained checkpoint to `checkpoints/baseline_nanogpt/`
3. Rename `ckpt.pt` to `baseline_nanogpt.pt`
4. Copy `meta.pkl` from data directory

### Bayesian Inference
1. Use `nanogpt_bayesian_inference.ipynb` as main workflow
2. Configure paths in CONFIG dictionary at notebook start
3. Use `func.functional_call(model, params, (x,))` for parameter sampling
4. Monitor VI training through loss tracking and log posterior values

### Text Generation
- **Deterministic**: Standard model forward pass
- **Bayesian**: Sample multiple parameter sets, generate with each, analyze diversity
- Temperature control for both inference (`temperature` in VI) and generation (`generation_temperature`)

### Evaluation
- Use notebooks in `notebooks/` for model evaluation
- BLEU scores using SacreBLEU (documented in `docs/evaluation_metrics.md`)
- Uncertainty quantification through prediction variance across posterior samples

## Critical Dependencies
- **posteriors**: Core Bayesian inference library
- **torchopt**: Optimizers for variational inference  
- **optree**: Tree operations (used by posteriors)
- **torch.func**: Required for `functional_call` with sampled parameters

## Common Patterns

### Parameter Sampling Pattern
```python
# Sample from posterior
sampled_params = posteriors.vi.diag.sample(vi_state)

# Use with functional call
logits, _ = func.functional_call(model, sampled_params, (x,))
```

### Log Posterior Definition
```python
def log_posterior_fn(params, batch):
    nll = single_batch_loss(params, batch)  # Negative log-likelihood
    log_prior = sum(Normal(0, prior_std).log_prob(p).sum() for p in params.values())
    return -nll + log_prior / num_data  # Scale prior by data size
```

### Virtual Environment
- Python environment located in `bnn/` directory
- Activate with `bnn\Scripts\activate.bat` on Windows
- Install packages using pip within this environment

## File Naming Conventions
- Checkpoints: `{model_type}_{training_details}.pt` (e.g., `baseline_nanogpt.pt`)
- Metadata: `{model_type}_meta.pkl` 
- Configs: `config.yaml` with wandb integration details
- Notebooks: Descriptive names with underscores (e.g., `nanogpt_bayesian_inference.ipynb`)

## Gotchas
- Always use `func.functional_call()` instead of direct model calls when working with sampled parameters
- Path handling: Use absolute paths from `config.py` constants
- Device consistency: Ensure tensors and model are on same device (CPU/CUDA)
- Sequence length limits: Crop sequences to `model.config.block_size` to avoid memory issues
- Prior scaling: Always scale log prior by `1/num_data` for proper Bayesian inference