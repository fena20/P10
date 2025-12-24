#!/usr/bin/env python3
"""Generate visualizations for tail bias mitigation analysis."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def ensure_dir(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)

def plot_tail_bias_comparison(df, output_path):
    """Plot comparison of models on key metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    metrics = [
        ('wRMSE', 'Weighted RMSE', 'lower'),
        ('wMAE', 'Weighted MAE', 'lower'),
        ('wR²', 'Weighted R²', 'higher'),
        ('Top-10% Bias (%)', 'Top-10% Bias (%)', 'closer_to_zero'),
        ('Lift@10', 'Lift@10', 'higher'),
        ('NDCG', 'NDCG', 'higher')
    ]
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    
    for ax, (col, label, direction) in zip(axes.flatten(), metrics):
        bars = ax.bar(range(len(df)), df[col], color=colors[:len(df)])
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(['Baseline\n(Tweedie)', 'Quantile\n(q=0.90)', 
                           'Tail-Weighted\n(α=1)', 'Tail-Weighted\n(α=2)'], 
                          fontsize=9)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(label, fontsize=12, fontweight='bold')
        
        # Add value labels on bars
        for bar, val in zip(bars, df[col]):
            height = bar.get_height()
            ax.annotate(f'{val:.2f}' if abs(val) < 100 else f'{val:.0f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        
        # Highlight best
        if direction == 'lower':
            best_idx = df[col].idxmin()
        elif direction == 'higher':
            best_idx = df[col].idxmax()
        else:  # closer_to_zero
            best_idx = df[col].abs().idxmin()
        
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)
    
    plt.suptitle('Tail Bias Mitigation: Model Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

def plot_top10_bias_by_model(df, output_path):
    """Plot top-10% bias specifically."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ['Baseline\n(Tweedie)', 'Quantile\n(q=0.90)', 
              'Tail-Weighted\n(α=1)', 'Tail-Weighted\n(α=2)']
    bias_pct = df['Top-10% Bias (%)'].values
    
    colors = ['#e74c3c' if b < -5 else '#f39c12' if b < 0 else '#2ecc71' for b in bias_pct]
    
    bars = ax.barh(range(len(models)), bias_pct, color=colors)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.axvline(x=-5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlabel('Top-10% Bias (%)', fontsize=12)
    ax.set_title('Top-Decile Underprediction by Model\n(Negative = Underprediction)', 
                 fontsize=13, fontweight='bold')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, bias_pct)):
        ax.annotate(f'{val:.1f}%',
                   xy=(val, i),
                   xytext=(5 if val >= 0 else -5, 0),
                   textcoords="offset points",
                   ha='left' if val >= 0 else 'right',
                   va='center', fontsize=11, fontweight='bold')
    
    # Add annotation for best model
    best_idx = np.argmin(np.abs(bias_pct))
    ax.annotate('✓ Best', xy=(bias_pct[best_idx], best_idx),
               xytext=(40, 0), textcoords="offset points",
               ha='left', va='center', fontsize=10, color='green',
               fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

def plot_tail_bias_by_tech_group(df, output_path):
    """Plot tail bias metrics by technology group."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    models = df['Model'].unique()
    tech_groups = ['combustion', 'electric_heat_pump', 'electric_resistance', 'hybrid_ambiguous']
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    
    # Plot 1: Top-10% Bias by tech group
    ax = axes[0, 0]
    x = np.arange(len(tech_groups))
    width = 0.2
    for i, model in enumerate(models):
        model_df = df[df['Model'] == model]
        values = [model_df[model_df['tech_group'] == tg]['top_decile_bias_pct'].values[0] 
                  if len(model_df[model_df['tech_group'] == tg]) > 0 else 0 
                  for tg in tech_groups]
        ax.bar(x + i*width, values, width, label=model[:15], color=colors[i])
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(['Combustion', 'Heat Pump', 'Electric\nResistance', 'Hybrid'], fontsize=9)
    ax.set_ylabel('Top-10% Bias (%)')
    ax.set_title('Top-Decile Bias by Technology Group', fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.legend(fontsize=8, loc='upper right')
    
    # Plot 2: Lift@10 by tech group
    ax = axes[0, 1]
    for i, model in enumerate(models):
        model_df = df[df['Model'] == model]
        values = [model_df[model_df['tech_group'] == tg]['lift_at_k'].values[0] 
                  if len(model_df[model_df['tech_group'] == tg]) > 0 else 0 
                  for tg in tech_groups]
        ax.bar(x + i*width, values, width, label=model[:15], color=colors[i])
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(['Combustion', 'Heat Pump', 'Electric\nResistance', 'Hybrid'], fontsize=9)
    ax.set_ylabel('Lift@10')
    ax.set_title('Targeting Lift by Technology Group', fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    
    # Plot 3: NDCG by tech group
    ax = axes[1, 0]
    for i, model in enumerate(models):
        model_df = df[df['Model'] == model]
        values = [model_df[model_df['tech_group'] == tg]['ndcg'].values[0] 
                  if len(model_df[model_df['tech_group'] == tg]) > 0 else 0 
                  for tg in tech_groups]
        ax.bar(x + i*width, values, width, label=model[:15], color=colors[i])
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(['Combustion', 'Heat Pump', 'Electric\nResistance', 'Hybrid'], fontsize=9)
    ax.set_ylabel('NDCG')
    ax.set_title('Ranking Quality (NDCG) by Technology Group', fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    
    # Plot 4: Recall@10 by tech group
    ax = axes[1, 1]
    for i, model in enumerate(models):
        model_df = df[df['Model'] == model]
        values = [model_df[model_df['tech_group'] == tg]['recall_at_k'].values[0] 
                  if len(model_df[model_df['tech_group'] == tg]) > 0 else 0 
                  for tg in tech_groups]
        ax.bar(x + i*width, values, width, label=model[:15], color=colors[i])
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(['Combustion', 'Heat Pump', 'Electric\nResistance', 'Hybrid'], fontsize=9)
    ax.set_ylabel('Recall@10')
    ax.set_title('Top-User Identification (Recall@10) by Technology Group', fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    
    plt.suptitle('Tail Bias Metrics by Technology Group', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

def plot_calibration_equity(df, output_path, title_suffix=''):
    """Plot calibration equity results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Bias before vs after
    ax = axes[0]
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width/2, df['bias_before'], width, label='Before Calibration', color='#e74c3c', alpha=0.7)
    ax.bar(x + width/2, df['bias_after'], width, label='After Calibration', color='#2ecc71', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(df['group'], fontsize=9)
    ax.set_ylabel('Bias (BTU)')
    ax.set_title(f'Bias Reduction by Climate Zone{title_suffix}', fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.legend()
    
    # Add reduction annotations
    for i, (before, after, reduction) in enumerate(zip(df['bias_before'], df['bias_after'], df['bias_reduction'])):
        if abs(reduction) > 100:
            ax.annotate(f'↓{reduction:.0f}', xy=(i, max(before, after)),
                       xytext=(0, 5), textcoords="offset points",
                       ha='center', fontsize=8, color='green')
    
    # Plot 2: MAE before vs after  
    ax = axes[1]
    ax.bar(x - width/2, df['mae_before'], width, label='Before Calibration', color='#e74c3c', alpha=0.7)
    ax.bar(x + width/2, df['mae_after'], width, label='After Calibration', color='#2ecc71', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(df['group'], fontsize=9)
    ax.set_ylabel('MAE (BTU)')
    ax.set_title(f'MAE Reduction by Climate Zone{title_suffix}', fontweight='bold')
    ax.legend()
    
    plt.suptitle('Group-Conditional Calibration: Equity Impact', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

def plot_delta_ci(df, output_path):
    """Plot confidence intervals for delta metrics."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = df['Metric'].values
    estimates = df['Estimate'].values
    ci_lower = df['CI_lower'].values
    ci_upper = df['CI_upper'].values
    
    y_pos = np.arange(len(metrics))
    
    # Plot error bars
    ax.errorbar(estimates, y_pos, xerr=[estimates - ci_lower, ci_upper - estimates],
                fmt='o', markersize=10, capsize=5, capthick=2, linewidth=2,
                color='#3498db', ecolor='#3498db')
    
    # Add vertical line at zero
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics, fontsize=11)
    ax.set_xlabel('Δ = Baseline (Tweedie) - Quantile (q=0.90)', fontsize=12)
    ax.set_title('Baseline Tweedie vs Quantile q=0.90: Delta with 95% CI\n(Negative Δ = Baseline is better on that metric)', 
                 fontsize=13, fontweight='bold')
    
    # Add value annotations
    for i, (est, lo, hi) in enumerate(zip(estimates, ci_lower, ci_upper)):
        ax.annotate(f'{est:.2f}\n[{lo:.2f}, {hi:.2f}]',
                   xy=(est, i), xytext=(10, 0), textcoords="offset points",
                   ha='left', va='center', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))
    
    # Add significance indicator
    for i, (lo, hi) in enumerate(zip(ci_lower, ci_upper)):
        if lo > 0 or hi < 0:  # CI doesn't include zero
            ax.annotate('*', xy=(estimates[i], i), xytext=(-20, 0),
                       textcoords="offset points", ha='center', va='center',
                       fontsize=16, color='green', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

def plot_model_tradeoff(df, output_path):
    """Plot trade-off between accuracy and tail bias."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract data
    models = df['Model'].values
    rmse = df['wRMSE'].values
    bias = np.abs(df['Top-10% Bias (%)'].values)
    lift = df['Lift@10'].values
    
    # Normalize for marker size
    sizes = (lift - lift.min()) / (lift.max() - lift.min()) * 300 + 100
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    
    for i, (m, x, y, s, c) in enumerate(zip(models, rmse, bias, sizes, colors)):
        ax.scatter(x, y, s=s, c=c, alpha=0.7, edgecolors='black', linewidth=2)
        # Add label
        label = m.replace('Tail-Weighted', 'TW').replace('Baseline', 'Base').replace('Quantile', 'Q90')
        ax.annotate(label, xy=(x, y), xytext=(10, 10), textcoords="offset points",
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=c, alpha=0.3))
    
    ax.set_xlabel('Weighted RMSE (lower is better)', fontsize=12)
    ax.set_ylabel('|Top-10% Bias (%)| (lower is better)', fontsize=12)
    ax.set_title('Accuracy vs Tail Bias Trade-off\n(Marker size = Lift@10)', 
                 fontsize=13, fontweight='bold')
    
    # Add ideal corner annotation
    ax.annotate('← Ideal Corner\n(Low RMSE, Low Bias)', 
               xy=(rmse.min(), bias.min()),
               xytext=(rmse.min() + 1000, bias.min() + 2),
               fontsize=10, color='gray', style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

def main():
    """Generate all plots."""
    base_path = Path('/workspace/outputs_tail_bias')
    tables_path = base_path / 'tables'
    figures_path = base_path / 'figures'
    ensure_dir(figures_path)
    
    # Load data
    comparison_df = pd.read_csv(tables_path / 'tail_bias_comparison.csv')
    tech_group_df = pd.read_csv(tables_path / 'tail_bias_by_tech_group.csv')
    delta_ci_df = pd.read_csv(tables_path / 'delta_ci_split_vs_mono.csv')
    calibration_df = pd.read_csv(tables_path / 'calibration_climate_climate.csv')
    
    print("\n" + "="*60)
    print("Generating Tail Bias Mitigation Visualizations")
    print("="*60 + "\n")
    
    # Generate plots
    plot_tail_bias_comparison(comparison_df, figures_path / 'fig1_tail_bias_comparison.png')
    plot_top10_bias_by_model(comparison_df, figures_path / 'fig2_top10_bias_by_model.png')
    plot_tail_bias_by_tech_group(tech_group_df, figures_path / 'fig3_tail_bias_by_tech_group.png')
    plot_calibration_equity(calibration_df, figures_path / 'fig4_calibration_equity.png')
    plot_delta_ci(delta_ci_df, figures_path / 'fig5_delta_ci.png')
    plot_model_tradeoff(comparison_df, figures_path / 'fig6_model_tradeoff.png')
    
    print("\n" + "="*60)
    print("All visualizations generated successfully!")
    print(f"Output directory: {figures_path}")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
