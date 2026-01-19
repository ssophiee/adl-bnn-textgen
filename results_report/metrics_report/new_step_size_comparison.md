# Evaluation Figures — Step Size 5e−06 (Updated)

### TL;DR — Are Bayesian samplers better than baseline?

- Overall, there is no clear across-the-board win over the baseline on BLEU/ROUGE; families are close on means.
- In this run, the baseline is marginally higher on BLEU/ROUGE (with sizable variance), while Bayesian samplers provide competitive quality.
- Depending on decoding settings, Bayesian samplers can match or exceed baseline on specific configs; choose based on your priority (e.g., uncertainty, stability/diversity) rather than expecting a universal quality uplift.

Note: In these plots, UNKNOWN denotes the deterministic baseline.

## Temperature Effects (0.3 vs 0.8)

![metrics_vs_temperature](metrics_vs_temperature.png)

- Why: Temperature controls diversity vs likelihood during decoding.
- Key takeaways:
	- 0.8 tends to increase BLEU/ROUGE1; ROUGE‑L often dips slightly.
	- Perplexity rises markedly at 0.8 (less probable but more diverse generations).


## Number of Samples (10 → 30)

![metrics_vs_num_samples](metrics_vs_num_samples.png)

- Why: More samples can stabilize estimates or average out modes.
- Key takeaways:
	- At top‑k=10, increasing to 30 samples commonly improves BLEU/ROUGE.
	- At top‑k=50, the same increase often hurts BLEU/ROUGE.

## Top‑k Effects (10 vs 50)

![metrics_vs_topk](metrics_vs_topk.png)

- Why: Top‑k expands the candidate set, affecting lexical variety and tail risk.
- Key takeaways:
	- Effects are non‑monotonic and interact with both model family and number of samples.
	- Benefits are clearer at samples=10; regressions are common at samples=30.

## Model Comparison (Overall)

![model_comparison_overall](model_comparison_overall.png)

- What to read: Overall averages by family (baseline, BAOA, SGHMC).
- Key takeaways: Families are close on means; in this run, baseline is slightly higher on BLEU/ROUGE1 but also shows higher perplexity.


## Train − Val Difference (BLEU/ROUGE)

![model_comparison_train_val_difference](model_comparison_train_val_difference.png)

- What to read: Train − Val deltas as a qualitative overfitting signal.
- DISCLAIMER: the validation split uses few samples and may be noisy.

## Train − Val Difference (Perplexity)

![perplexity_train_val_difference](perplexity_train_val_difference.png)

- What to read: Train − Val perplexity gaps across families.

---
