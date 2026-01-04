# Frequently Asked Questions

## General

### Q: What is the Goldbach conjecture?
**A:** Every even integer greater than 2 can be expressed as the sum of two primes. While unproven, it has been verified up to 4×10¹⁸.

### Q: What is G(N)?
**A:** The number of ways to write N as p + q where both p and q are prime (counting order).

### Q: What is the Fano factor?
**A:** α = Var(G)/E[G]. It measures whether fluctuations are random (α=1, Poisson) or suppressed (α<1, sub-Poissonian).

## Results

### Q: Why is α ≈ 0.5 significant?
**A:** This value is predicted by GUE (Gaussian Unitary Ensemble) random matrix theory, suggesting prime pairs exhibit "spectral rigidity" like quantum energy levels.

### Q: What is the connection to Riemann zeta zeros?
**A:** Montgomery (1973) conjectured that zeta zeros follow GUE statistics. If primes inherit this structure, we expect α ≈ 0.5 in prime-pair problems.

### Q: How confident are you in these results?
**A:** Very confident for N ≤ 10⁷ (exact computation). For 10¹², 10¹⁶, results are consistent within Monte Carlo sampling error (~2%).

## Technical

### Q: Can I run the code on my computer?
**A:** Yes! Exact computation up to N = 10⁷ requires ~4 GB RAM and ~30 minutes. Monte Carlo probes are fast (~90 seconds).

### Q: Why Monte Carlo for large N?
**A:** Exact enumeration requires O(N/ln N) operations. For N = 10¹⁶, this would take years. Monte Carlo provides estimates with quantified uncertainty.

### Q: What is the extrapolation uncertainty?
**A:** The model α = α_asymp + C/ln(N) has R² ≈ 0.39. The true asymptotic value could differ from our estimate of 0.57.

## Citation

### Q: How should I cite this work?
**A:** See CITATION.cff or the README for BibTeX format.
