# Methodology

## Overview

This document describes the computational methods used to investigate variance structure in Goldbach representation counts.

## 1. Exact Enumeration (N ≤ 10⁷)

### Algorithm

```python
1. Generate primes up to N using Sieve of Eratosthenes
2. Store primes in hash set for O(1) lookup
3. For each even N:
   - Count pairs (p, N-p) where both are prime
   - Compute Hardy-Littlewood prediction
   - Record G(N), Pred(N), ω(N), S(N)
```

### Complexity
- Sieve: O(N log log N)
- Counting: O(π(N)) per N, where π(N) ~ N/ln(N)

### Data Points
- 19,001 points in [10³, 2×10⁶] (log-spaced)
- 1,801 points in [10⁶, 10⁷] (log-spaced)

## 2. Monte Carlo Sampling (N = 10¹², 10¹⁶)

### Challenge
Exact enumeration is infeasible for N > 10⁸ due to:
- Memory: Prime sieve requires ~N/8 bytes
- Time: O(π(N)) ~ 10¹³ operations for N = 10¹⁶

### Method

```python
1. Randomly sample primes p < N/2
2. Use Miller-Rabin test to check if (N - p) is prime
3. Estimate G(N) = 2 × (valid pairs) × π(N/2) / (samples)
4. Compute 95% CI from binomial proportion
```

### Parameters
- N = 10¹²: 10⁶ samples, ~60 seconds
- N = 10¹⁶: 10⁶ samples, ~90 seconds

### Error Analysis
- Sampling error: ±2.16% (95% CI) at N = 10¹⁶
- Intrinsic formula error: << sampling error

## 3. Fano Factor Computation

### Definition
```
α = Var(G) / E[G]
```

### Method
1. Partition data into log-spaced bins
2. For each bin:
   - Compute residuals: r = G(N) - Pred(N)
   - Variance: Var(r)
   - Mean prediction: ⟨Pred⟩
3. Fano factor: α = Var(r) / ⟨Pred⟩

### Extrapolation
```
α(N) = α_asymp + C / ln(N)
```
Fitted via linear regression on 1/ln(N).

## 4. Statistical Tests

### Hypothesis Testing
- H₀: α = 0.5 (GUE) → Cannot reject (p > 0.05)
- H₀: α = 1.0 (Poisson) → Strongly rejected (p < 0.001)

### Distribution Analysis
- Mean, variance, skewness, kurtosis of normalized residuals
- Comparison with Gaussian (GUE limit)

## 5. Reproducibility

### Random Seeds
- N = 10¹²: seed = 42
- N = 10¹⁶: seed = 2026

### Software
- Python 3.10+
- NumPy 1.21+, SciPy 1.7+, Pandas 1.3+

### Hardware
- Standard desktop CPU (no GPU required)
- ~4 GB RAM for N ≤ 10⁷
- ~1 GB RAM for Monte Carlo
