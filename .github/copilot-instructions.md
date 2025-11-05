# Copilot Instructions (adl-bnn-textgen)

Purpose: Fast orientation for AI coding agents working on Bayesian extensions of NanoGPT with the `posteriors` library.

Core Architecture:
- Deterministic NanoGPT in `baselines/nanogpt/model.py` (character-level, weight tying, block size enforcement).
- Bayesian layer is external: we never wrap the model; we operate on param dicts via `torch.func.functional_call`.
- Global params & sampler configs live in `config.py` (env-driven: `MODEL_PATH`, `META_PATH`, `DATA_DIR`). Override via `.env` rather than hardcoding.

Primary Workflow (scripts):
1. Load checkpoint + tokenizer: `load_model(MODEL_PATH)`, `load_tokenizer(META_PATH)`.
2. Build random next-character batches with `create_training_batches(...)` (each sequence predicts 1 next char).
3. Select sampler: `vi`, `ekf`, `sgmcmc` (laplace placeholder). Pipeline in `src/bayesian_utils.BayesianSamplerPipeline`.
4. Train: loop calls `transform.update(state, batch)`; for VI clamp `state.log_sd_diag[...]` to `max_log_scale` to prevent variance blow-up.
5. Evaluate: deterministic vs posterior mean vs sampled posterior (`evaluate_predictions`). Tracks loss, perplexity, predictive entropy.
6. Save artifacts to `checkpoints/samplers/{sampler}_sampler/run_TIMESTAMP/` (model, metrics JSON, summary text).

Sampler Differences:
- VI: `posteriors.vi.diag.build(...); sample via posteriors.vi.diag.sample(state)`.
- EKF: `posteriors.ekf.diag_fisher.build` (uses Fisher diag, samples via ekf module).
- SGMCMC: collects thinning samples into `state.collected_samples`; evaluate last N; no explicit variance params.

Log Posterior Pattern (character-level): scale likelihood to full dataset size, prior tempered by `prior_beta`:
```python
log_prior = posteriors.diag_normal_log_prob(params, mean=INITIAL_PARAMS, sd_diag=CONFIG['prior_std'])
log_likelihood = -nll * CONFIG['train_samples']
log_posterior = log_likelihood + CONFIG['prior_beta'] * log_prior
```

Generation:
- Deterministic: `model(x_cond)`; Bayesian: sample params then `func.functional_call(model, sampled, (x,))`.
- Always crop to `model.config.block_size` before forward to avoid assert.

Uncertainty Metrics:
- Predictive entropy computed over stacked softmax of sampled logits.
- Parameter std (VI/EKF) from `exp(log_sd_diag)` averaged across tensors.

Training Entry Point:
```bash
python scripts\bayesian_training_script.py --sampler vi --epochs 10 --train-samples 2000
```
Use `--no-wandb` if WANDB_AVAILABLE false. On Windows activate env: `bnn\Scripts\activate.bat`.

Conventions & Pitfalls:
- Always use `functional_call` with sampled params (do not mutate `model` weights in place).
- Clamp VI log std to avoid exploding predictive entropy.
- Each batch predicts exactly one next char; scaling must reflect total sequence count (`train_samples`).
- For SGMCMC ensure burn-in (`sgmcmc_burnin_epochs`) and thinning (`sgmcmc_thinning`) config keys before relying on collected samples.

Extend Safely:
- Add new sampler by implementing `_setup_<name>` returning (transform, state) mirroring existing ones and update `AVAILABLE_SAMPLERS`.
- Keep saved state structure: params, sampler aux, metrics JSON.

Immediate References:
`scripts/bayesian_training_script.py`, `src/bayesian_utils.py`, `src/nanogpt_utils.py`, `baselines/nanogpt/model.py`.

Ask for clarification if modifying prior scaling or adding sequence-level objectives.