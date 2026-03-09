# BNN Text Generation: Evaluation Report

**Key insights:**
- **No Bayesian config beats baseline on all three metrics, but individual metrics can be improved** 
- **Relevance can be exceeded**: SGHMC-PP-5e6 @ temp_0.3, top_k=50, samples=20 achieves R=7.70 (vs baseline peak 7.47), but quality drops (6.77)
- **Quality and diversity can be matched or exceeded**: BAOA-ZC-1e6 @ temp_0.8, top_k=50, samples=20 ties baseline quality (6.83) and beats diversity (6.60 vs 6.57), but relevance drops (6.97 vs 7.47)
- **Pretrained prior is consistently better**; without it, SGHMC at high LR (SGHMC-ZC-1e5) is the worst configuration

---

## 1) Setup

**Model**: NanoGPT (10.65M params), character-level Shakespeare. **Baseline**: deterministic AdamW, 2000 steps.

### 2×2×2 Design

| Factor | Levels |
|--------|--------|
| Sampler | BAOA, SGHMC |
| Prior | Pretrained-centered, Zero-centered |
| LR | Lower (BAOA: 1e-6 / SGHMC: 5e-6), Higher (BAOA: 5e-6 / SGHMC: 1e-5) |

### Model Registry

| ID | Sampler | Prior | LR | Run ID |
|----|---------|-------|----|--------|
| BAOA-PP-1e6 | BAOA | Pretrained | 1e-06 | run_20260207-192540 |
| BAOA-PP-5e6 | BAOA | Pretrained | 5e-06 | run_20260225-101215 |
| SGHMC-PP-5e6 | SGHMC | Pretrained | 5e-06 | run_20260224-230657 |
| SGHMC-PP-1e5 | SGHMC | Pretrained | 1e-05 | run_20260224-232547 |
| BAOA-ZC-1e6 | BAOA | Zero | 1e-06 | run_20260207-232141 |
| BAOA-ZC-5e6 | BAOA | Zero | 5e-06 | run_20260225-102111 |
| SGHMC-ZC-5e6 | SGHMC | Zero | 5e-06 | run_20260224-231606 |
| SGHMC-ZC-1e5 | SGHMC | Zero | 1e-05 | run_20260224-233423 |

Shared training: warmup 200 steps, sampling 1000 steps, thinning every 10th, batch 16, seq 128.

### LLM-Judge Evaluation

- **Judge**: Qwen2.5-7B-Instruct (unsloth/Qwen2.5-7B-Instruct-bnb-4bit), scores 0–10
- **Metrics**: Quality (coherence, grammar, style), Diversity (creativity, variety), Relevance (prompt alignment)
- **Generation grid**: temperature {0.3, 0.8} × top_k {10, 50} × num_samples {10, 20} → 8 configs per model
- **Prompts**: 15 Shakespeare prompts, 1 generation per prompt per config (120 total per model)
- **Data**: `results/evaluation/llm_judge/updated_results/`

---

## 2) Results

### Overall Model Ranking (averaged across all 8 decoding configs)

| Model | Prior | LR | Quality | Diversity | Relevance |
|-------|-------|----|---------|-----------|-----------|
| **Baseline** | — | — | **6.13** | **5.77** | 6.21 |
| SGHMC-PP-1e5 | Pretrained | 1e-05 | 6.11 | 5.66 | 6.28 |
| SGHMC-ZC-5e6 | Zero | 5e-06 | **6.13** | 5.65 | 6.25 |
| BAOA-PP-5e6 | Pretrained | 5e-06 | 6.05 | 5.58 | **6.34** |
| SGHMC-PP-5e6 | Pretrained | 5e-06 | 6.00 | 5.40 | 6.40 |
| BAOA-ZC-1e6 | Zero | 1e-06 | 5.98 | **5.70** | 6.05 |
| BAOA-PP-1e6 | Pretrained | 1e-06 | 5.95 | 5.68 | 6.12 |
| BAOA-ZC-5e6 | Zero | 5e-06 | 5.91 | 5.48 | 6.28 |
| SGHMC-ZC-1e5 | Zero | 1e-05 | 5.78 | 5.59 | 5.95 |

### Prior: Pretrained vs Zero-Centered

**Pretrained prior consistently outperforms zero-centered on quality and relevance.**

| Metric | Pretrained prior (4 runs) | Zero-centered prior (4 runs) | Winner |
|--------|--------------------------|------------------------------|--------|
| Quality | **6.03** | 5.95 | **Pretrained** |
| Diversity | 5.58 | **5.60** | Zero (marginal) |
| Relevance | **6.28** | 6.13 | **Pretrained** |

The effect is stronger for SGHMC (+0.24 relevance) than BAOA (+0.07 relevance).

**Critical insight: SGHMC with zero prior is highly LR-sensitive:**

| SGHMC | Low LR (5e-6) | High LR (1e-5) |
|-------|--------------|----------------|
| Pretrained prior | 5.93 | **6.02** |
| Zero-centered prior | **6.01** | 5.77 ← collapses |

> **Limitation — `prior_std` not analyzed:** All runs use a fixed `prior_std=1.0`, which maps to `sd_diag` in the posteriors library (the diagonal of the prior covariance). This parameter directly controls the strength of prior regularization — smaller values tighten the prior, stronger penalization; larger values loosen it. Varying `prior_std` is a natural next axis to study, especially for the zero-centered prior where the regularization strength may compensate for the lack of a pretrained initialization.


### Sampler: BAOA vs SGHMC

**SGHMC has a higher ceiling and lower floor; BAOA is more stable.**

| | BAOA range | SGHMC range |
|--|-----------|------------|
| Overall scores | 5.89–5.99 | 5.77–6.02 |
| Best config | BAOA-PP-5e6 | SGHMC-PP-1e5 |
| Worst config | BAOA-ZC-5e6 | SGHMC-ZC-1e5 |

On average the two samplers are similar. Sampler choice matters less than prior and LR.

### Learning Rate

- **BAOA**: robust to LR choice; slight edge to higher LR (5e-6) with pretrained prior
- **SGHMC**: prior-conditional — high LR (1e-5) is better with pretrained prior, harmful with zero-centered prior

---

## 3) Best Configurations

| Goal | Model | Config | Quality | Diversity | Relevance |
|------|-------|--------|---------|-----------|-----------|
| **Max relevance** | **SGHMC-PP-5e6** | **temp=0.3, top_k=50, samples=20** | **6.77** | **6.43** | **7.70** |
| Max quality + diversity | BAOA-ZC-1e6 | temp=0.8, top_k=50, samples=20 | 6.83 | 6.60 | 6.97 |
| Best zero-prior model | SGHMC-ZC-5e6 | temp=0.3, top_k=10, samples=20 | 6.50 | 6.10 | 6.63 |
| Baseline reference | Baseline | temp=0.3, top_k=10, samples=20 | 6.83 | 6.57 | 7.47 |

At their optimal decoding configs, several Bayesian models exceed the baseline on individual metrics — this is not a forced quality-relevance trade-off. However, no single config beats the baseline on all three simultaneously.

---

## 4) Conclusion

| Question | Answer |
|----------|--------|
| Do BNNs outperform the baseline? | Individual metrics can be exceeded at the right decoding config |
| Which factor matters most? | **Prior** — pretrained prior improves performance |
| Best LR? | Sampler-dependent; SGHMC must use low LR with zero-centered prior |
| What to avoid? | SGHMC-ZC-1e5 — largest drop across all metrics |

**One-line summary:** Pretrained-centered prior is the most important design choice; with it, SGHMC-PP-1e5 matches the baseline closely and SGHMC-PP-5e6 achieves the highest relevance of any model (7.70) at temp=0.3, top_k=50, samples=20.
