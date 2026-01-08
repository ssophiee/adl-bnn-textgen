# Model Evaluation Metrics Report

_Last update: 2026-01-08T03:37:10.752977_

## 1. Dataset & evaluation footprint

- Configurations evaluated: **24**
- Total sample evaluations (all configurations combined): **overall=96**, **train=72**, **val=24**
- Per-configuration footprint appears to be constant: **overall=4**, **train=3**, **val=1** samples.

## 2. Model-level comparison (overall split)

### 2.1 Overall averages across hyperparameters (higher is better for BLEU/ROUGE; lower is better for perplexity)

| model | configs | bleu_mean | bleu_std | rouge2_mean | rouge2_std | rougeL_mean | ppl_mean | ppl_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baoa | 8 | 0.2561 | 0.0210 | 0.5146 | 0.0447 | 0.3797 | 105.7359 | 60.2734 |
| baseline | 8 | 0.2581 | 0.0190 | 0.5228 | 0.0424 | 0.3807 | 125.6453 | 85.6553 |
| sghmc | 8 | 0.2522 | 0.0146 | 0.5130 | 0.0409 | 0.3784 | 122.3775 | 74.7445 |

**Readout:** the three model families are close on average. Baseline is marginally highest on mean BLEU/ROUGE2, but variance is high.

### 2.2 Best configuration per model (ranked by overall BLEU)

| model | temperature | top_k | samples_setting | bleu | rouge2 | rougeL | perplexity | config_key |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.8000 | 50 | 10 | 0.2854 | 0.5810 | 0.3827 | 198.9008 | baseline_temp0.8_topk50_samples10_baseline |
| baoa | 0.8000 | 10 | 30 | 0.2793 | 0.5596 | 0.3796 | 178.9917 | baoa_temp0.8_topk10_samples30_run_20251224-145920 |
| sghmc | 0.8000 | 10 | 30 | 0.2714 | 0.5569 | 0.3757 | 206.5861 | sghmc_temp0.8_topk10_samples30_run_20251226-113201 |

## 3. Hyperparameter effects (overall split)

### 3.1 Temperature (0.8 vs 0.3): consistent quality uplift, large perplexity increase

Average deltas when moving from **T=0.3 → T=0.8** (overall):

| model | bleu_delta_0.8-0.3 | rouge2_delta_0.8-0.3 | rougeL_delta_0.8-0.3 | ppl_delta_0.8-0.3 |
| --- | --- | --- | --- | --- |
| baoa | 0.0324 | 0.0749 | -0.0033 | 108.8970 |
| baseline | 0.0335 | 0.0764 | -0.0059 | 137.2908 |
| sghmc | 0.0209 | 0.0684 | -0.0082 | 132.1510 |

**Interpretation**

- Across **all** models and settings, **T=0.8 increases BLEU/ROUGE2** materially.
- This comes with a **very large perplexity increase**, which is consistent with generating less probable sequences (higher diversity). Whether this is “bad” depends on how perplexity is defined in your evaluation pipeline and whether you optimize for fluency vs similarity.

### 3.2 Top-k (50 vs 10): strong interaction with `samples_setting`

The effect of increasing **top_k 10 → 50** is **not monotonic**; it depends strongly on `samples_setting` (10 vs 30):

| model | temperature | samples_setting | bleu_topk10 | bleu_topk50 | bleu_delta | rouge2_topk10 | rouge2_topk50 | rouge2_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baoa | 0.3000 | 10 | 0.2334 | 0.2605 | 0.0271 | 0.4703 | 0.5114 | 0.0411 |
| baoa | 0.3000 | 30 | 0.2444 | 0.2212 | -0.0232 | 0.4856 | 0.4412 | -0.0443 |
| baoa | 0.8000 | 10 | 0.2729 | 0.2749 | 0.0020 | 0.5449 | 0.5583 | 0.0135 |
| baoa | 0.8000 | 30 | 0.2793 | 0.2621 | -0.0172 | 0.5596 | 0.5452 | -0.0144 |
| baseline | 0.3000 | 10 | 0.2375 | 0.2387 | 0.0012 | 0.4739 | 0.4821 | 0.0082 |
| baseline | 0.3000 | 30 | 0.2443 | 0.2451 | 0.0008 | 0.4857 | 0.4968 | 0.0111 |
| baseline | 0.8000 | 10 | 0.2747 | 0.2854 | 0.0107 | 0.5614 | 0.5810 | 0.0196 |
| baseline | 0.8000 | 30 | 0.2756 | 0.2638 | -0.0119 | 0.5535 | 0.5482 | -0.0053 |
| sghmc | 0.3000 | 10 | 0.2360 | 0.2545 | 0.0186 | 0.4717 | 0.5075 | 0.0359 |
| sghmc | 0.3000 | 30 | 0.2464 | 0.2302 | -0.0163 | 0.4819 | 0.4539 | -0.0279 |
| sghmc | 0.8000 | 10 | 0.2629 | 0.2669 | 0.0040 | 0.5446 | 0.5622 | 0.0177 |
| sghmc | 0.8000 | 30 | 0.2714 | 0.2496 | -0.0218 | 0.5569 | 0.5250 | -0.0318 |

**Interpretation (pattern)**

- When **samples_setting=10**, moving to **top_k=50 usually improves** BLEU/ROUGE2 (positive deltas) across models.
- When **samples_setting=30**, moving to **top_k=50 usually hurts** BLEU/ROUGE2 (negative deltas) across models.
- Practical takeaway: **avoid pairing “large top_k” with “large samples_setting”** unless you have a reason—this combination appears to inject noise that outweighs benefits in n-gram similarity.

### 3.3 Num-samples setting (30 vs 10): helps at top_k=10, often hurts at top_k=50

Effect of increasing **samples_setting 10 → 30** (overall):

| model | temperature | top_k | bleu_samples10 | bleu_samples30 | bleu_delta_30-10 | rouge2_samples10 | rouge2_samples30 | rouge2_delta_30-10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baoa | 0.3000 | 10 | 0.2334 | 0.2444 | 0.0110 | 0.4703 | 0.4856 | 0.0152 |
| baoa | 0.3000 | 50 | 0.2605 | 0.2212 | -0.0393 | 0.5114 | 0.4412 | -0.0702 |
| baoa | 0.8000 | 10 | 0.2729 | 0.2793 | 0.0064 | 0.5449 | 0.5596 | 0.0147 |
| baoa | 0.8000 | 50 | 0.2749 | 0.2621 | -0.0128 | 0.5583 | 0.5452 | -0.0132 |
| baseline | 0.3000 | 10 | 0.2375 | 0.2443 | 0.0069 | 0.4739 | 0.4857 | 0.0119 |
| baseline | 0.3000 | 50 | 0.2387 | 0.2451 | 0.0064 | 0.4821 | 0.4968 | 0.0147 |
| baseline | 0.8000 | 10 | 0.2747 | 0.2756 | 0.0010 | 0.5614 | 0.5535 | -0.0079 |
| baseline | 0.8000 | 50 | 0.2854 | 0.2638 | -0.0216 | 0.5810 | 0.5482 | -0.0328 |
| sghmc | 0.3000 | 10 | 0.2360 | 0.2464 | 0.0105 | 0.4717 | 0.4819 | 0.0102 |
| sghmc | 0.3000 | 50 | 0.2545 | 0.2302 | -0.0244 | 0.5075 | 0.4539 | -0.0536 |
| sghmc | 0.8000 | 10 | 0.2629 | 0.2714 | 0.0085 | 0.5446 | 0.5569 | 0.0123 |
| sghmc | 0.8000 | 50 | 0.2669 | 0.2496 | -0.0173 | 0.5622 | 0.5250 | -0.0372 |

**Interpretation (pattern)**

- At **top_k=10**, increasing `samples_setting` to **30** generally **improves** BLEU/ROUGE2.
- At **top_k=50**, increasing `samples_setting` to **30** generally **degrades** BLEU/ROUGE2 (often substantially).

## 4. Train vs validation (for completeness; interpret with caution)

### 4.1 Top-5 configurations by train BLEU

| model | temperature | top_k | samples_setting | bleu | rouge2 | rougeL | perplexity | config_key |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.8000 | 50 | 10 | 0.2756 | 0.5693 | 0.3734 | 154.9316 | baseline_temp0.8_topk50_samples10_baseline |
| baseline | 0.8000 | 10 | 30 | 0.2736 | 0.5559 | 0.3718 | 137.1759 | baseline_temp0.8_topk10_samples30_baseline |
| baseline | 0.8000 | 10 | 10 | 0.2729 | 0.5607 | 0.3750 | 163.5744 | baseline_temp0.8_topk10_samples10_baseline |
| baoa | 0.8000 | 10 | 30 | 0.2704 | 0.5431 | 0.3713 | 164.4621 | baoa_temp0.8_topk10_samples30_run_20251224-145920 |
| baoa | 0.8000 | 10 | 10 | 0.2695 | 0.5423 | 0.3746 | 161.9486 | baoa_temp0.8_topk10_samples10_run_20251224-145920 |

### 4.2 Top-5 configurations by val BLEU

| model | temperature | top_k | samples_setting | bleu | rouge2 | rougeL | perplexity | config_key |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.8000 | 50 | 10 | 0.3137 | 0.6163 | 0.4071 | 330.8084 | baseline_temp0.8_topk50_samples10_baseline |
| baoa | 0.8000 | 10 | 30 | 0.3052 | 0.6090 | 0.3985 | 222.5806 | baoa_temp0.8_topk10_samples30_run_20251224-145920 |
| baoa | 0.8000 | 50 | 10 | 0.3050 | 0.6028 | 0.4071 | 166.2633 | baoa_temp0.8_topk50_samples10_run_20251224-145920 |
| sghmc | 0.8000 | 10 | 30 | 0.2935 | 0.5844 | 0.3961 | 156.4516 | sghmc_temp0.8_topk10_samples30_run_20251226-113201 |
| sghmc | 0.8000 | 50 | 10 | 0.2853 | 0.5795 | 0.3973 | 150.6408 | sghmc_temp0.8_topk50_samples10_run_20251226-113201 |

**Caution:** because `val` uses 1 sample and is consistently flagged as “prompt not found”, treat these rankings as **diagnostic only**.

## 5. Recommendations (based on this run)

### If you are optimizing for BLEU/ROUGE (similarity)

- **Baseline**: **T=0.8, top_k=50, samples_setting=10** (best overall BLEU/ROUGE2 in this run).
- **BAOA / SGHMC**: **T=0.8, top_k=10, samples_setting=30** (best overall BLEU within each sampler family).

### If you want a more conservative / lower-temperature regime

- **BAOA** at **T=0.3, top_k=50, samples_setting=10** is the strongest low-temperature configuration by overall BLEU in this dataset.

### If you want a stable tuning heuristic (given the observed interactions)

1. Pick temperature (likely **0.8** for max similarity).
2. Decide whether you want to spend “budget” on exploration via **top_k** or via **samples_setting**:
   - Either **top_k=50 & samples_setting=10**, or
   - **top_k=10 & samples_setting=30**
3. Avoid **top_k=50 & samples_setting=30** unless you re-evaluate on a larger validation set.

## Appendix A: Full configuration list (overall)

| model | temperature | top_k | samples_setting | bleu | rouge1 | rouge2 | rougeL | perplexity | config_key |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baoa | 0.3000 | 10 | 10 | 0.2334 | 0.7189 | 0.4703 | 0.3735 | 39.5207 | baoa_temp0.3_topk10_samples10_run_20251224-145920 |
| baoa | 0.3000 | 10 | 30 | 0.2444 | 0.7309 | 0.4856 | 0.3739 | 42.2241 | baoa_temp0.3_topk10_samples30_run_20251224-145920 |
| baoa | 0.3000 | 50 | 10 | 0.2605 | 0.7461 | 0.5114 | 0.3961 | 71.2529 | baoa_temp0.3_topk50_samples10_run_20251224-145920 |
| baoa | 0.3000 | 50 | 30 | 0.2212 | 0.7121 | 0.4412 | 0.3821 | 52.1518 | baoa_temp0.3_topk50_samples30_run_20251224-145920 |
| baoa | 0.8000 | 10 | 10 | 0.2729 | 0.7526 | 0.5449 | 0.3791 | 146.1892 | baoa_temp0.8_topk10_samples10_run_20251224-145920 |
| baoa | 0.8000 | 10 | 30 | 0.2793 | 0.7548 | 0.5596 | 0.3796 | 178.9917 | baoa_temp0.8_topk10_samples30_run_20251224-145920 |
| baoa | 0.8000 | 50 | 10 | 0.2749 | 0.7468 | 0.5583 | 0.3806 | 141.4701 | baoa_temp0.8_topk50_samples10_run_20251224-145920 |
| baoa | 0.8000 | 50 | 30 | 0.2621 | 0.7517 | 0.5452 | 0.3729 | 174.0866 | baoa_temp0.8_topk50_samples30_run_20251224-145920 |
| baseline | 0.3000 | 10 | 10 | 0.2375 | 0.7266 | 0.4739 | 0.3836 | 59.1522 | baseline_temp0.3_topk10_samples10_baseline |
| baseline | 0.3000 | 10 | 30 | 0.2443 | 0.7303 | 0.4857 | 0.3858 | 46.5250 | baseline_temp0.3_topk10_samples30_baseline |
| baseline | 0.3000 | 50 | 10 | 0.2387 | 0.7269 | 0.4821 | 0.3806 | 36.0250 | baseline_temp0.3_topk50_samples10_baseline |
| baseline | 0.3000 | 50 | 30 | 0.2451 | 0.7468 | 0.4968 | 0.3845 | 86.2971 | baseline_temp0.3_topk50_samples30_baseline |
| baseline | 0.8000 | 10 | 10 | 0.2747 | 0.7563 | 0.5614 | 0.3797 | 172.0179 | baseline_temp0.8_topk10_samples10_baseline |
| baseline | 0.8000 | 10 | 30 | 0.2756 | 0.7486 | 0.5535 | 0.3791 | 127.0487 | baseline_temp0.8_topk10_samples30_baseline |
| baseline | 0.8000 | 50 | 10 | 0.2854 | 0.7578 | 0.5810 | 0.3827 | 198.9008 | baseline_temp0.8_topk50_samples10_baseline |
| baseline | 0.8000 | 50 | 30 | 0.2638 | 0.7440 | 0.5482 | 0.3696 | 279.1952 | baseline_temp0.8_topk50_samples30_baseline |
| sghmc | 0.3000 | 10 | 10 | 0.2360 | 0.7376 | 0.4717 | 0.3852 | 61.2631 | sghmc_temp0.3_topk10_samples10_run_20251226-113201 |
| sghmc | 0.3000 | 10 | 30 | 0.2464 | 0.7355 | 0.4819 | 0.3846 | 66.4532 | sghmc_temp0.3_topk10_samples30_run_20251226-113201 |
| sghmc | 0.3000 | 50 | 10 | 0.2545 | 0.7376 | 0.5075 | 0.3839 | 57.4491 | sghmc_temp0.3_topk50_samples10_run_20251226-113201 |
| sghmc | 0.3000 | 50 | 30 | 0.2302 | 0.7259 | 0.4539 | 0.3763 | 40.0425 | sghmc_temp0.3_topk50_samples30_run_20251226-113201 |
| sghmc | 0.8000 | 10 | 10 | 0.2629 | 0.7502 | 0.5446 | 0.3784 | 194.7182 | sghmc_temp0.8_topk10_samples10_run_20251226-113201 |
| sghmc | 0.8000 | 10 | 30 | 0.2714 | 0.7535 | 0.5569 | 0.3757 | 206.5861 | sghmc_temp0.8_topk10_samples30_run_20251226-113201 |
| sghmc | 0.8000 | 50 | 10 | 0.2669 | 0.7551 | 0.5622 | 0.3791 | 215.7529 | sghmc_temp0.8_topk50_samples10_run_20251226-113201 |
| sghmc | 0.8000 | 50 | 30 | 0.2496 | 0.7392 | 0.5250 | 0.3641 | 136.7547 | sghmc_temp0.8_topk50_samples30_run_20251226-113201 |

