#!/usr/bin/env python3
"""
Paper III: Complete GUE Verification Script
完整的GUE统计验证脚本

包含三重证据检验:
1. 对数修正律: α(N) = α_∞ + C/ln(N)
2. 偏度演化: γ₁ → 0
3. 间距分布: σ → 0.707 (GUE)

使用方法:
    python verify_gue_complete.py <csv_file>
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress, skew, kurtosis, norm
import sys
import os

def load_data(csv_file):
    """加载数据"""
    print(f"📂 加载: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"   样本: {len(df)}, N ∈ [{df['N'].min():,}, {df['N'].max():,}]")
    return df

def compute_statistics(df, n_bins=30):
    """计算各N范围的统计量"""
    # 确定预测列
    if 'Pred_Integral' in df.columns:
        pred_col = 'Pred_Integral'
    elif 'Pred' in df.columns:
        pred_col = 'Pred'
    else:
        raise ValueError("需要预测值列 (Pred_Integral 或 Pred)")
    
    bins = np.logspace(np.log10(df['N'].min()), np.log10(df['N'].max()), n_bins)
    results = []
    
    for i in range(len(bins)-1):
        low, high = bins[i], bins[i+1]
        subset = df[(df['N'] >= low) & (df['N'] < high)]
        
        if len(subset) > 30:
            residuals = subset['G_N'] - subset[pred_col]
            mean_pred = subset[pred_col].mean()
            
            # Fano factor
            alpha = residuals.var() / mean_pred
            
            # 归一化残差
            normalized = residuals / np.sqrt(mean_pred)
            
            results.append({
                'N': np.sqrt(low * high),
                'ln_N': np.log(np.sqrt(low * high)),
                'inv_ln_N': 1/np.log(np.sqrt(low * high)),
                'alpha': alpha,
                'skewness': skew(normalized) if len(subset) > 50 else np.nan,
                'kurtosis': kurtosis(normalized) if len(subset) > 50 else np.nan,
                'std': normalized.std(),
                'n_samples': len(subset)
            })
    
    return pd.DataFrame(results)

def test_logarithmic_law(stats_df):
    """检验对数修正律"""
    print("\n" + "=" * 60)
    print("证据1: 对数修正律 α(N) = α_∞ + C/ln(N)")
    print("=" * 60)
    
    inv_ln = stats_df['inv_ln_N'].values
    alpha = stats_df['alpha'].values
    
    slope, intercept, r, p, se = linregress(inv_ln, alpha)
    
    print(f"\n拟合结果:")
    print(f"  α_∞ = {intercept:.4f}")
    print(f"  C = {slope:.4f}")
    print(f"  R² = {r**2:.4f}")
    
    # 判断
    if 0.45 <= intercept <= 0.65:
        verdict = "✅ 支持GUE (α_∞ ≈ 0.5)"
    elif intercept > 0.8:
        verdict = "⚠️ 趋向Poisson (α_∞ → 1)"
    else:
        verdict = f"🔶 中间状态 (α_∞ ≈ {intercept:.2f})"
    
    print(f"\n判定: {verdict}")
    
    return intercept, slope, r**2

def test_skewness(stats_df):
    """检验偏度演化"""
    print("\n" + "=" * 60)
    print("证据2: 偏度演化 γ₁ → 0")
    print("=" * 60)
    
    valid = stats_df.dropna(subset=['skewness'])
    
    if len(valid) < 5:
        print("  ⚠️ 数据点不足")
        return np.nan
    
    # 按N大小分组
    small_N = valid[valid['N'] < valid['N'].median()]
    large_N = valid[valid['N'] >= valid['N'].median()]
    
    mean_small = small_N['skewness'].mean()
    mean_large = large_N['skewness'].mean()
    
    print(f"\n小N偏度: {mean_small:.4f}")
    print(f"大N偏度: {mean_large:.4f}")
    
    if abs(mean_large) < abs(mean_small):
        print(f"\n✅ 偏度趋向0 (GUE对称分布)")
    else:
        print(f"\n⚠️ 偏度未明显改善")
    
    return mean_large

def test_spacing(stats_df):
    """检验间距分布"""
    print("\n" + "=" * 60)
    print("证据3: 间距分布 σ → 0.707")
    print("=" * 60)
    
    # 大N的标准差
    large_N = stats_df[stats_df['N'] > stats_df['N'].median()]
    mean_std = large_N['std'].mean()
    
    gue_std = np.sqrt(0.5)  # ≈ 0.707
    poisson_std = 1.0
    
    gue_diff = abs(mean_std - gue_std) / gue_std * 100
    poisson_diff = abs(mean_std - poisson_std) / poisson_std * 100
    
    print(f"\n观测标准差: {mean_std:.4f}")
    print(f"GUE理论值: {gue_std:.4f} (偏差: {gue_diff:.1f}%)")
    print(f"Poisson理论值: {poisson_std:.4f} (偏差: {poisson_diff:.1f}%)")
    
    if gue_diff < poisson_diff:
        print(f"\n✅ 更接近GUE分布")
    else:
        print(f"\n⚠️ 更接近Poisson分布")
    
    return mean_std

def create_visualization(stats_df, df, output_file):
    """创建可视化"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('GUE Statistics Verification', fontsize=14, fontweight='bold')
    
    # 1. 对数律
    ax1 = axes[0, 0]
    ax1.scatter(stats_df['inv_ln_N'], stats_df['alpha'], s=60, c='purple', edgecolor='black')
    slope, intercept, r, p, se = linregress(stats_df['inv_ln_N'], stats_df['alpha'])
    x_fit = np.linspace(0, stats_df['inv_ln_N'].max()*1.1, 100)
    ax1.plot(x_fit, intercept + slope*x_fit, 'b-', linewidth=2)
    ax1.axhline(0.5, color='green', linestyle='--', label='GUE')
    ax1.axhline(1.0, color='red', linestyle=':', label='Poisson')
    ax1.set_xlabel('1/ln(N)')
    ax1.set_ylabel('α')
    ax1.set_title(f'Logarithmic Law: α_∞ = {intercept:.3f}')
    ax1.legend()
    
    # 2. 偏度
    ax2 = axes[0, 1]
    valid_skew = stats_df.dropna(subset=['skewness'])
    ax2.scatter(np.log10(valid_skew['N']), valid_skew['skewness'], s=60, c='coral', edgecolor='black')
    ax2.axhline(0, color='green', linestyle='--', label='GUE (symmetric)')
    ax2.set_xlabel('log₁₀(N)')
    ax2.set_ylabel('Skewness')
    ax2.set_title('Skewness Evolution')
    ax2.legend()
    
    # 3. 标准差
    ax3 = axes[1, 0]
    ax3.scatter(np.log10(stats_df['N']), stats_df['std'], s=60, c='steelblue', edgecolor='black')
    ax3.axhline(np.sqrt(0.5), color='green', linestyle='--', label='GUE (0.707)')
    ax3.axhline(1.0, color='red', linestyle=':', label='Poisson (1.0)')
    ax3.set_xlabel('log₁₀(N)')
    ax3.set_ylabel('Normalized Std')
    ax3.set_title('Variance Compression')
    ax3.legend()
    
    # 4. 分布直方图
    ax4 = axes[1, 1]
    pred_col = 'Pred_Integral' if 'Pred_Integral' in df.columns else 'Pred'
    large_N = df[df['N'] > df['N'].median()]
    residuals = (large_N['G_N'] - large_N[pred_col]) / np.sqrt(large_N[pred_col])
    ax4.hist(residuals, bins=40, density=True, alpha=0.6, color='purple', edgecolor='black')
    x = np.linspace(-3, 3, 200)
    ax4.plot(x, norm.pdf(x, 0, np.sqrt(0.5)), 'g-', linewidth=2, label='GUE')
    ax4.plot(x, norm.pdf(x, 0, 1.0), 'r--', linewidth=2, label='Poisson')
    ax4.set_xlabel('Normalized Residual')
    ax4.set_ylabel('Density')
    ax4.set_title('Spacing Distribution')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n🖼️ 图表已保存: {output_file}")

def main():
    if len(sys.argv) < 2:
        print("用法: python verify_gue_complete.py <csv_file>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    print("=" * 60)
    print("Paper III: Complete GUE Verification")
    print("=" * 60)
    
    # 加载数据
    df = load_data(csv_file)
    
    # 计算统计量
    stats_df = compute_statistics(df)
    
    # 三重检验
    alpha_inf, C, R2 = test_logarithmic_law(stats_df)
    skewness = test_skewness(stats_df)
    std = test_spacing(stats_df)
    
    # 总结
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    
    print(f"""
    证据1 (对数律): α_∞ = {alpha_inf:.3f}
    证据2 (偏度):   γ₁ → {skewness:.3f}
    证据3 (标准差): σ = {std:.3f} (GUE: 0.707)
    
    综合判定: {"✅ 支持GUE假设" if 0.45 <= alpha_inf <= 0.65 else "⚠️ 需要更多数据"}
    """)
    
    # 可视化
    base_name = os.path.splitext(csv_file)[0]
    output_file = f"{base_name}_gue_verification.png"
    create_visualization(stats_df, df, output_file)

if __name__ == "__main__":
    main()
