# Evaluation Results Comparison

## Overview
This document compares LLM-based evaluation results across different Bayesian sampling approaches and configurations for text generation.

## Models Compared

1. **SGHMC Sampler** - Stochastic Gradient Hamiltonian Monte Carlo (step_size=1e-05)
2. **BAOA Sampler (Old)** - Bayesian Optimization with Adaptive step size (step_size=1e-06)
3. **BAOA Sampler (New)** - BAOA with increased step size
4. **Baseline** - Deterministic baseline model

---

## Overall Performance Summary

| Model | Quality | Diversity | Relevance | Total Generations |
|-------|---------|-----------|-----------|-------------------|
| **SGHMC** (1e-05) | 4.455 | 5.473 | **6.866** | 56 |
| **BAOA Old** (1e-06) | **4.509** | 5.557 | **7.075** | 106 |
| **BAOA New** (higher) | 4.487 | **5.752** | 6.378 | 119 |
| **Baseline** | 4.386 | 5.436 | 6.977 | 110 |

### Key Findings

**Best Overall:** BAOA Old (1e-06)
- Highest quality (4.509)
- Highest relevance (7.075)
- Balanced diversity (5.557)

**Diversity Winner:** BAOA New
- Best diversity score (5.752)
- But at cost of relevance (-9.8% vs BAOA Old)

**Stability Champion:** Baseline
- Consistent performance across metrics
- Good relevance (6.977)
- Serves as solid reference point

---

## Detailed Configuration Breakdown

### SGHMC Sampler Results

| Configuration | Quality | Diversity | Relevance | Count |
|--------------|---------|-----------|-----------|-------|
| temp_0.3_topk_10_samples_20 | **4.679** | 5.464 | 6.679 | 14 |
| temp_0.3_topk_20_samples_20 | 4.000 | 5.000 | 6.500 | 14 |
| temp_0.8_topk_10_samples_20 | 4.536 | **5.786** | **7.214** | 14 |
| temp_0.8_topk_20_samples_20 | 4.607 | 5.643 | 7.071 | 14 |

**Analysis:**
- Higher temperature (0.8) improves relevance significantly
- Top-k=10 with temp=0.8 achieves best relevance (7.214)
- Lower temperature (0.3) with top-k=20 shows weakest performance

---

### BAOA Old (step_size=1e-06)

| Configuration | Quality | Diversity | Relevance | Count |
|--------------|---------|-----------|-----------|-------|
| temp_0.3_topk_10_samples_10 | **4.857** | **5.893** | **7.357** | 14 |
| temp_0.3_topk_10_samples_20 | 4.679 | 5.714 | 7.179 | 14 |
| temp_0.3_topk_20_samples_10 | 4.429 | 5.464 | 7.071 | 14 |
| temp_0.3_topk_20_samples_20 | 4.357 | 5.464 | 6.964 | 14 |
| temp_0.8_topk_10_samples_10 | 4.357 | 5.357 | 6.893 | 14 |
| temp_0.8_topk_10_samples_20 | 4.462 | 5.500 | 7.077 | 13 |
| temp_0.8_topk_20_samples_10 | 4.409 | 5.455 | 6.864 | 11 |
| temp_0.8_topk_20_samples_20 | 4.500 | 5.583 | 7.167 | 12 |

**Analysis:**
- Best configuration: temp=0.3, top-k=10, samples=10 (7.357 relevance!)
- Lower temperature (0.3) generally outperforms higher (0.8)
- Smaller top-k (10) shows better results
- Opposite trend to SGHMC regarding temperature

---

### BAOA New (higher step_size)

| Configuration | Quality | Diversity | Relevance | Count |
|--------------|---------|-----------|-----------|-------|
| temp_0.3_topk_10_samples_10 | **4.800** | 5.567 | 6.067 | 15 |
| temp_0.3_topk_10_samples_20 | 4.233 | 5.433 | 6.133 | 15 |
| temp_0.3_topk_20_samples_10 | 4.250 | 5.679 | 6.393 | 14 |
| temp_0.3_topk_20_samples_20 | 4.600 | **5.933** | **6.667** | 15 |
| temp_0.8_topk_10_samples_10 | 4.633 | 5.867 | 6.367 | 15 |
| temp_0.8_topk_10_samples_20 | 4.500 | **5.967** | 6.633 | 15 |
| temp_0.8_topk_20_samples_10 | 4.367 | 5.833 | 6.533 | 15 |
| temp_0.8_topk_20_samples_20 | 4.500 | 5.733 | 6.233 | 15 |

**Analysis:**
- Best diversity across all models (5.933-5.967 range)
- Relevance significantly lower than old BAOA (~0.7 points)
- Higher temperature now preferred (consistent with SGHMC)
- Step size increase shifted the model's behavior pattern

---

### Baseline Model

| Configuration | Quality | Diversity | Relevance | Count |
|--------------|---------|-----------|-----------|-------|
| temp_0.3_topk_10_samples_10 | 4.538 | 5.577 | **7.154** | 13 |
| temp_0.3_topk_10_samples_20 | **4.679** | **5.679** | 7.107 | 14 |
| temp_0.3_topk_20_samples_10 | 4.286 | 5.357 | 6.857 | 14 |
| temp_0.3_topk_20_samples_20 | 4.321 | 5.393 | 6.964 | 14 |
| temp_0.8_topk_10_samples_10 | 4.286 | 5.286 | 6.821 | 14 |
| temp_0.8_topk_10_samples_20 | 4.269 | 5.385 | 6.962 | 13 |
| temp_0.8_topk_20_samples_10 | 4.500 | 5.500 | 7.071 | 14 |
| temp_0.8_topk_20_samples_20 | 4.214 | 5.321 | 6.893 | 14 |

**Analysis:**
- Lower temperature (0.3) consistently better
- Top-k=10 with more samples performs well
- Remarkably consistent performance
- Competitive with Bayesian approaches

---

## Cross-Model Insights

### Temperature Effects
- **BAOA Old**: Prefers temp=0.3 (lower is better)
- **BAOA New**: Prefers temp=0.8 (higher is better)
- **SGHMC**: Benefits from temp=0.8
- **Baseline**: Works best with temp=0.3

**Conclusion:** Step size changes in BAOA fundamentally altered temperature sensitivity

### Top-K Effects
- **General trend:** top-k=10 generally outperforms top-k=20 for relevance
- **Exception:** BAOA New shows more mixed results
- Smaller vocabulary restriction helps maintain context

### Sample Count Effects
- Less clear pattern than temperature/top-k
- Configuration-dependent
- May require task-specific tuning

---

## Impact of BAOA Step Size Increase

### Quantitative Changes
| Metric | Old (1e-06) | New (higher) | Change |
|--------|-------------|--------------|--------|
| Quality | 4.509 | 4.487 | -0.5% |
| Diversity | 5.557 | 5.752 | **+3.5%** |
| Relevance | 7.075 | 6.378 | **-9.8%** |

### Qualitative Observations

**Positive:**
- Increased exploration (diversity up)
- More varied outputs
- Quality remains stable

**Negative:**
- Significant relevance drop
- Lost competitive advantage (was best, now worst in relevance)
- Trade-off may not be worth it

**Hypothesis:**
The higher step size causes the sampler to explore more aggressively, finding more diverse solutions but straying further from the relevant probability regions.

---

## Recommendations

### For Production Use
**Use BAOA Old (step_size=1e-06)** with:
- Configuration: temp=0.3, top-k=10, samples=10
- Achieves best balance of all metrics
- Highest relevance (7.357) with good quality (4.857)

### For High-Diversity Tasks
**Use BAOA New** with:
- Configuration: temp=0.8, top-k=10, samples=20
- Maximizes diversity (5.967)
- Accept relevance trade-off if exploration is priority

### For Stability/Reliability
**Use Baseline** with:
- Configuration: temp=0.3, top-k=10, samples=20
- Predictable, consistent performance
- Good all-around metrics

### Future Experiments
Consider testing intermediate step sizes between 1e-06 and current to find sweet spot:
- Target: Capture diversity gains without relevance loss
- Suggested range: 2e-06, 5e-06, 7e-06
- Monitor quality-diversity-relevance triangle

---

## Statistical Notes

- Evaluation counts vary (56-119 generations)
- Higher counts (BAOA New: 119) provide more reliable statistics
- Some configurations have fewer samples (11-15) due to generation issues
- Results are LLM-based evaluations (GPT-4 scoring 0-10 scale)

---

**Generated:** 2025-12-26
**Data Sources:**
- [baoa_checkpoint_gen_aggregated.json](checkpoints/samplers/baoa_sampler/run_20251224-145920/eval_results/baoa_checkpoint_gen_aggregated.json)
- Historical evaluation results from previous runs
