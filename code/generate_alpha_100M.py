#!/usr/bin/env python3
"""
Paper III: α Evolution Data Generator (优化版)
生成 N = 10^6 到 10^8 的 Fano Factor 演化数据

优化特性:
- 使用numpy向量化操作
- 支持断点续传
- 实时进度显示
- 自动α分析

使用方法:
    python generate_alpha_100M_optimized.py

输出:
    ALPHA_EVOLUTION_100M.csv
"""

import numpy as np
import pandas as pd
import time
import os
import sys

# ===== 配置参数 =====
START_N = 1_000_000      # 起始N
END_N = 100_000_000      # 终止N (1亿)
POINTS_PER_DECADE = 100  # 每个数量级的采样点数（已优化）
C2 = 0.6601618158        # 孪生素数常数
CHECKPOINT_FILE = 'alpha_checkpoint.csv'

def segmented_sieve(limit, segment_size=10**6):
    """
    分段筛法 - 内存优化版本
    返回所有小于等于limit的素数
    """
    print(f"📦 使用分段筛法生成素数 (limit={limit:,})...")
    t0 = time.time()
    
    # 首先筛出sqrt(limit)以内的素数
    sqrt_limit = int(limit**0.5) + 1
    small_sieve = np.ones(sqrt_limit, dtype=bool)
    small_sieve[0:2] = False
    
    for i in range(2, int(sqrt_limit**0.5) + 1):
        if small_sieve[i]:
            small_sieve[i*i::i] = False
    
    small_primes = np.nonzero(small_sieve)[0]
    
    # 收集所有素数
    all_primes = list(small_primes)
    
    # 分段筛
    for low in range(sqrt_limit, limit + 1, segment_size):
        high = min(low + segment_size, limit + 1)
        segment = np.ones(high - low, dtype=bool)
        
        for p in small_primes:
            if p * p > high:
                break
            # 找到segment中第一个p的倍数
            start = ((low + p - 1) // p) * p - low
            if start < 0:
                start += p
            segment[start::p] = False
        
        # 添加这个segment中的素数
        segment_primes = np.nonzero(segment)[0] + low
        all_primes.extend(segment_primes)
        
        # 进度
        if (high - sqrt_limit) % (10 * segment_size) < segment_size:
            progress = (high - sqrt_limit) / (limit - sqrt_limit) * 100
            print(f"   筛法进度: {progress:.1f}%")
    
    primes = np.array(all_primes, dtype=np.int64)
    elapsed = time.time() - t0
    print(f"   ✓ 完成! 找到 {len(primes):,} 个素数 ({elapsed:.1f}秒)")
    
    return primes

def get_primes_simple(max_n):
    """简单筛法（适用于内存充足的情况）"""
    print(f"📦 生成素数表 (最大值: {max_n:,})...")
    t0 = time.time()
    
    try:
        sieve = np.ones(max_n + 1, dtype=np.bool_)
        sieve[0:2] = False
        
        for i in range(2, int(np.sqrt(max_n)) + 1):
            if sieve[i]:
                sieve[i*i::i] = False
        
        primes = np.nonzero(sieve)[0]
        elapsed = time.time() - t0
        print(f"   ✓ 完成! 找到 {len(primes):,} 个素数 ({elapsed:.1f}秒)")
        return primes
    except MemoryError:
        print("   内存不足，切换到分段筛法...")
        return segmented_sieve(max_n)

def get_singular_series_vectorized(n_array):
    """向量化计算奇异级数"""
    results = np.ones(len(n_array))
    
    for idx, n in enumerate(n_array):
        if n % 2 != 0:
            results[idx] = 0
            continue
            
        temp = n
        while temp % 2 == 0:
            temp //= 2
        
        d = 3
        while d * d <= temp:
            if temp % d == 0:
                results[idx] *= (d - 1) / (d - 2)
                while temp % d == 0:
                    temp //= d
            d += 2
        if temp > 1:
            results[idx] *= (temp - 1) / (temp - 2)
    
    return results

def predict_simple(n):
    """简化预测（避免积分的开销）"""
    # Li2(n) ≈ n/ln²(n) × (1 + 2/ln(n) + ...)
    ln_n = np.log(n)
    return n / (ln_n ** 2) * (1 + 2/ln_n + 6/ln_n**2)

def count_goldbach_fast(n, primes, primes_set):
    """快速计算G(N)"""
    limit = n // 2
    idx = np.searchsorted(primes, limit, side='right')
    
    count = 0
    for p in primes[:idx]:
        if (n - p) in primes_set:
            count += 1
    
    # Ordered count
    g_n = count * 2
    if n % 2 == 0 and (n // 2) in primes_set:
        g_n -= 1
    
    return g_n

def load_checkpoint():
    """加载断点"""
    if os.path.exists(CHECKPOINT_FILE):
        df = pd.read_csv(CHECKPOINT_FILE)
        last_n = df['N'].max()
        print(f"📂 发现断点文件，从 N={last_n:,} 继续...")
        return df.to_dict('records'), last_n
    return [], 0

def save_checkpoint(results):
    """保存断点"""
    df = pd.DataFrame(results)
    df.to_csv(CHECKPOINT_FILE, index=False)

def run_sampling():
    """主采样函数"""
    print("=" * 70)
    print("🚀 Paper III: α Evolution Generator (Optimized)")
    print("=" * 70)
    print(f"\n目标: N = {START_N:,} → {END_N:,}")
    
    # 加载断点
    results, last_n = load_checkpoint()
    
    # 生成素数
    primes = get_primes_simple(END_N)
    primes_set = set(primes)
    
    # 生成采样点
    n_decades = np.log10(END_N / START_N)
    total_points = int(n_decades * POINTS_PER_DECADE)
    
    targets = np.logspace(np.log10(START_N), np.log10(END_N), total_points)
    targets = np.unique(targets.astype(np.int64))
    targets = [t if t % 2 == 0 else t + 1 for t in targets]
    targets = sorted(set(targets))
    
    # 跳过已完成的
    if last_n > 0:
        targets = [t for t in targets if t > last_n]
    
    print(f"\n📊 采样点: {len(targets)} (跳过已完成: {len(results)})")
    
    # 开始采样
    start_time = time.time()
    checkpoint_interval = 50
    
    for i, n in enumerate(targets):
        # 计算
        g_n = count_goldbach_fast(n, primes, primes_set)
        sn = get_singular_series_vectorized(np.array([n]))[0]
        pred_simple = 2 * C2 * sn * predict_simple(n)
        
        results.append({
            'N': n,
            'G_N': g_n,
            'S_N': sn,
            'Pred': pred_simple,
            'Residual': g_n - pred_simple,
            'Bias': (g_n - pred_simple) / pred_simple * 100 if pred_simple > 0 else 0
        })
        
        # 进度和断点
        if (i + 1) % checkpoint_interval == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(targets) - i - 1) / rate if rate > 0 else 0
            
            print(f"   [{i+1}/{len(targets)}] N={n:,} | G={g_n:,} | "
                  f"速度:{rate:.1f}/s | 剩余:{eta/60:.1f}分钟")
            
            # 保存断点
            save_checkpoint(results)
    
    # 最终保存
    df = pd.DataFrame(results)
    output_file = 'ALPHA_EVOLUTION_100M.csv'
    df.to_csv(output_file, index=False)
    
    # 删除断点文件
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
    
    total_time = time.time() - start_time
    print(f"\n✅ 完成! 总用时: {total_time/60:.1f}分钟")
    print(f"   数据已保存至: {output_file}")
    
    # 快速α分析
    analyze_alpha(df)
    
    return df

def analyze_alpha(df):
    """快速α分析"""
    print("\n" + "=" * 70)
    print("📈 α 演化分析")
    print("=" * 70)
    
    bins = np.logspace(np.log10(df['N'].min()), np.log10(df['N'].max()), 12)
    
    print(f"\n{'N范围':<25} {'α':<10} {'样本量':<10}")
    print("-" * 50)
    
    alpha_list = []
    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i+1]
        subset = df[(df['N'] >= low) & (df['N'] < high)]
        
        if len(subset) > 10:
            residuals = subset['G_N'] - subset['Pred']
            var_g = residuals.var()
            mean_g = subset['Pred'].mean()
            alpha = var_g / mean_g
            alpha_list.append(alpha)
            
            label = f"{low:.1e}-{high:.1e}"
            print(f"{label:<25} {alpha:<10.4f} {len(subset):<10}")
    
    if len(alpha_list) > 0:
        mean_alpha = np.mean(alpha_list[-5:]) if len(alpha_list) >= 5 else np.mean(alpha_list)
        print(f"\n最后几个bin的平均α: {mean_alpha:.4f}")
        
        if mean_alpha < 0.55:
            print("✅ 支持 GUE 假设 (α → 0.5)")
        elif mean_alpha > 0.7:
            print("⚠️ 可能趋向 Poisson (α → 1.0)")
        else:
            print("🔶 过渡区域，需要更大N验证")

if __name__ == "__main__":
    # 检查内存
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024**3)
        required_gb = END_N / 1e9 * 1.2
        
        print(f"\n系统信息:")
        print(f"  可用内存: {available_gb:.1f} GB")
        print(f"  预计需要: {required_gb:.1f} GB")
        
        if available_gb < required_gb:
            print(f"\n⚠️ 内存可能不足！")
            print(f"   建议: 减少END_N或使用分段筛法")
            response = input("   继续? (y/n): ")
            if response.lower() != 'y':
                sys.exit(0)
    except ImportError:
        pass
    
    df = run_sampling()
