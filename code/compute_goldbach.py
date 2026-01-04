#!/usr/bin/env python3
"""
Goldbach Representation Computation
====================================
Core functions for computing G(N) and Hardy-Littlewood predictions.

Author: Ruqing Chen
Email: ruqing@hotmail.com
"""

import numpy as np
from math import log, sqrt
from scipy.integrate import quad
from functools import lru_cache

# Twin prime constant C2
C2 = 0.6601618158468695739278121100145557784326233602847334133194484233354056423

def sieve_primes(n):
    """Generate all primes up to n using Sieve of Eratosthenes."""
    if n < 2:
        return np.array([], dtype=np.int64)
    
    sieve = np.ones(n + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    
    for i in range(2, int(sqrt(n)) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    
    return np.where(sieve)[0]

def prime_factors(n):
    """Return set of prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def omega(n):
    """Count distinct prime factors of n."""
    return len(prime_factors(n))

def singular_series(n):
    """
    Compute singular series S(N) for Goldbach formula.
    S(N) = prod_{p|N, p>2} (p-1)/(p-2)
    """
    s = 1.0
    for p in prime_factors(n):
        if p > 2:
            s *= (p - 1) / (p - 2)
    return s

def li2_integral(n):
    """
    Compute Li2(N) = integral from 2 to N-2 of dt/(ln(t)*ln(N-t))
    """
    if n <= 4:
        return 0.0
    
    def integrand(t):
        if t <= 2 or t >= n - 2:
            return 0.0
        ln_t = log(t)
        ln_nt = log(n - t)
        if ln_t <= 0 or ln_nt <= 0:
            return 0.0
        return 1.0 / (ln_t * ln_nt)
    
    result, _ = quad(integrand, 2.01, n - 2.01, limit=100)
    return result

def hardy_littlewood_prediction(n):
    """
    Compute Hardy-Littlewood prediction for G(N).
    G(N) ~ 2 * C2 * S(N) * Li2(N)
    """
    if n <= 4 or n % 2 != 0:
        return 0.0
    
    s = singular_series(n)
    li2 = li2_integral(n)
    
    return 2 * C2 * s * li2

def count_goldbach(n, primes=None, prime_set=None):
    """
    Count Goldbach representations G(N).
    Returns the number of ordered pairs (p, q) with p + q = N, both prime.
    """
    if n <= 4 or n % 2 != 0:
        return 0
    
    if primes is None:
        primes = sieve_primes(n)
        prime_set = set(primes)
    
    count = 0
    for p in primes:
        if p >= n:
            break
        if (n - p) in prime_set:
            count += 1
    
    return count

def compute_fano_factor(g_values, pred_values):
    """
    Compute Fano factor alpha = Var(G) / E[G]
    Using residuals: alpha = Var(G - Pred) / mean(Pred)
    """
    residuals = np.array(g_values) - np.array(pred_values)
    variance = np.var(residuals)
    mean_pred = np.mean(pred_values)
    
    if mean_pred <= 0:
        return np.nan
    
    return variance / mean_pred

if __name__ == "__main__":
    # Example usage
    print("Goldbach Computation Module")
    print("=" * 40)
    
    test_N = 1000000
    print(f"\nTest: N = {test_N:,}")
    print(f"  ω(N) = {omega(test_N)}")
    print(f"  S(N) = {singular_series(test_N):.6f}")
    print(f"  Hardy-Littlewood prediction = {hardy_littlewood_prediction(test_N):.2f}")
    
    # Small example with exact count
    small_N = 100
    primes = sieve_primes(small_N)
    prime_set = set(primes)
    g = count_goldbach(small_N, primes, prime_set)
    pred = hardy_littlewood_prediction(small_N)
    print(f"\nN = {small_N}: G(N) = {g}, Pred = {pred:.2f}")
