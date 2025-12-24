#!/usr/bin/env python3
"""
Advanced Tail Bias Analysis for Applied Energy submission.

Implements:
- 5.1: Quantile sweep (q ∈ {0.80, 0.85, 0.90, 0.95}) with Pareto plot
- 5.2: Hybrid targeting score with λ optimization
- 5.3: CI/Significance on Top-10% bias using SDR replicate weights
- Fix: Corrected labeling for all plots
"""

import sys
import os
sys.path.insert(0, '/workspace')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from src.utils.helpers import Timer
from src.data.loader import load_recs_data
from src.data.preprocessor import RECSPreprocessor
from src.features.builder import FeatureBuilder
from src.models.tail_bias_models import QuantileLightGBMHeatingModel, TailBiasMetrics

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


@dataclass
class QuantileSweepResult:
    """Results from a single quantile model run."""
    quantile: float
    weighted_rmse: float
    weighted_mae: float
    weighted_r2: float
    weighted_bias: float
    top10_bias_pct: float
    top10_bias_pct_ci_lower: float
    top10_bias_pct_ci_upper: float
    lift_at_10: float
    lift_at_10_ci_lower: float
    lift_at_10_ci_upper: float
    ndcg: float
    ndcg_ci_lower: float
    ndcg_ci_upper: float
    precision_at_10: float
    recall_at_10: float


def compute_metric_with_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
    replicate_weights: np.ndarray,
    metric_func: callable,
    **metric_kwargs
) -> Tuple[float, float, float, float]:
    """
    Compute a metric with CI using SDR (Successive Difference Replications).
    
    Returns: (estimate, se, ci_lower, ci_upper)
    """
    # Point estimate
    point_estimate = metric_func(y_true, y_pred, weights, **metric_kwargs)
    
    # Replicate estimates
    n_replicates = replicate_weights.shape[1]
    replicate_estimates = []
    
    for r in range(n_replicates):
        rep_weights = replicate_weights[:, r]
        rep_estimate = metric_func(y_true, y_pred, rep_weights, **metric_kwargs)
        replicate_estimates.append(rep_estimate)
    
    replicate_estimates = np.array(replicate_estimates)
    
    # SDR variance estimation
    diffs = np.diff(replicate_estimates)
    variance = np.sum(diffs**2) / (2 * (n_replicates - 1))
    se = np.sqrt(variance)
    
    # 95% CI
    ci_lower = point_estimate - 1.96 * se
    ci_upper = point_estimate + 1.96 * se
    
    return point_estimate, se, ci_lower, ci_upper


def compute_top10_bias_pct(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    """Compute top-10% bias percentage."""
    # Find top 10% by true values
    threshold = np.percentile(y_true, 90)
    mask = y_true >= threshold
    
    if mask.sum() == 0:
        return 0.0
    
    top_true = y_true[mask]
    top_pred = y_pred[mask]
    top_weights = weights[mask]
    
    mean_true = np.average(top_true, weights=top_weights)
    mean_pred = np.average(top_pred, weights=top_weights)
    
    bias_pct = (mean_pred - mean_true) / mean_true * 100
    return bias_pct


def compute_lift_at_k(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray, k_pct: float = 0.10) -> float:
    """Compute Lift@k."""
    n = len(y_true)
    k = int(n * k_pct)
    
    # Top k by prediction
    pred_top_idx = np.argsort(y_pred)[-k:]
    # Top k by actual
    true_top_idx = np.argsort(y_true)[-k:]
    
    overlap = len(set(pred_top_idx) & set(true_top_idx))
    precision = overlap / k
    lift = precision / k_pct
    
    return lift


def compute_ndcg(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray, k: int = None) -> float:
    """Compute NDCG."""
    if k is None:
        k = int(len(y_true) * 0.1)
    
    # Sort by prediction
    pred_order = np.argsort(y_pred)[::-1][:k]
    
    # DCG
    relevance = y_true[pred_order]
    positions = np.arange(1, k + 1)
    dcg = np.sum(relevance / np.log2(positions + 1))
    
    # Ideal DCG
    ideal_order = np.argsort(y_true)[::-1][:k]
    ideal_relevance = y_true[ideal_order]
    idcg = np.sum(ideal_relevance / np.log2(positions + 1))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def run_quantile_sweep(
    df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    replicate_weights: np.ndarray,
    quantiles: List[float],
    n_folds: int = 3,
    random_state: int = 42
) -> List[QuantileSweepResult]:
    """Run quantile sweep with CV and compute metrics with CI."""
    
    results = []
    
    # Create stratification groups
    y_binned = pd.qcut(y, q=10, labels=False, duplicates='drop')
    
    for q in quantiles:
        print(f"\n{'='*60}")
        print(f"Running Quantile q={q}")
        print('='*60)
        
        # Cross-validation predictions
        cv_preds = np.zeros(len(y))
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y_binned)):
            print(f"  Fold {fold_idx + 1}/{n_folds}...")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            w_train = weights[train_idx]
            
            # Train quantile model
            model = QuantileLightGBMHeatingModel(
                quantile=q,
                params={
                    'n_estimators': 200,
                    'learning_rate': 0.05,
                    'max_depth': 6,
                    'num_leaves': 31,
                    'random_state': random_state + fold_idx
                }
            )
            model.fit(X_train, y_train, sample_weight=w_train)
            cv_preds[test_idx] = model.predict(X_test)
        
        # Compute metrics with CI using replicate weights
        # Top-10% Bias
        top10_bias, top10_se, top10_ci_lo, top10_ci_hi = compute_metric_with_ci(
            y, cv_preds, weights, replicate_weights,
            compute_top10_bias_pct
        )
        
        # Lift@10
        lift, lift_se, lift_ci_lo, lift_ci_hi = compute_metric_with_ci(
            y, cv_preds, weights, replicate_weights,
            compute_lift_at_k, k_pct=0.10
        )
        
        # NDCG
        ndcg, ndcg_se, ndcg_ci_lo, ndcg_ci_hi = compute_metric_with_ci(
            y, cv_preds, weights, replicate_weights,
            compute_ndcg
        )
        
        # Standard metrics (point estimates)
        from src.evaluation.metrics import WeightedMetrics
        wm = WeightedMetrics()
        rmse = wm.weighted_rmse(y, cv_preds, weights)
        mae = wm.weighted_mae(y, cv_preds, weights)
        r2 = wm.weighted_r2(y, cv_preds, weights)
        bias = wm.weighted_bias(y, cv_preds, weights)
        
        # Precision/Recall
        k = int(len(y) * 0.10)
        pred_top_idx = set(np.argsort(cv_preds)[-k:])
        true_top_idx = set(np.argsort(y)[-k:])
        overlap = len(pred_top_idx & true_top_idx)
        precision = overlap / k
        recall = overlap / k
        
        result = QuantileSweepResult(
            quantile=q,
            weighted_rmse=rmse,
            weighted_mae=mae,
            weighted_r2=r2,
            weighted_bias=bias,
            top10_bias_pct=top10_bias,
            top10_bias_pct_ci_lower=top10_ci_lo,
            top10_bias_pct_ci_upper=top10_ci_hi,
            lift_at_10=lift,
            lift_at_10_ci_lower=lift_ci_lo,
            lift_at_10_ci_upper=lift_ci_hi,
            ndcg=ndcg,
            ndcg_ci_lower=ndcg_ci_lo,
            ndcg_ci_upper=ndcg_ci_hi,
            precision_at_10=precision,
            recall_at_10=recall
        )
        results.append(result)
        
        print(f"  wRMSE: {rmse:.0f}, wR²: {r2:.3f}")
        print(f"  Top-10% Bias: {top10_bias:.1f}% [{top10_ci_lo:.1f}, {top10_ci_hi:.1f}]")
        print(f"  Lift@10: {lift:.2f} [{lift_ci_lo:.2f}, {lift_ci_hi:.2f}]")
        print(f"  NDCG: {ndcg:.3f} [{ndcg_ci_lo:.3f}, {ndcg_ci_hi:.3f}]")
    
    return results


def compute_hybrid_targeting_score(
    y_pred_mean: np.ndarray,
    y_pred_q90: np.ndarray,
    y_true: np.ndarray,
    weights: np.ndarray,
    lambda_values: np.ndarray = None
) -> Tuple[float, Dict]:
    """
    Compute hybrid targeting score and optimize λ.
    
    score = λ * rank(q90) + (1-λ) * rank(mean_pred)
    
    Returns: (optimal_lambda, results_dict)
    """
    if lambda_values is None:
        lambda_values = np.linspace(0, 1, 21)  # 0, 0.05, 0.10, ..., 1.0
    
    n = len(y_true)
    k = int(n * 0.10)
    
    # Compute ranks (higher = better for targeting)
    rank_mean = np.argsort(np.argsort(y_pred_mean))  # 0 = lowest, n-1 = highest
    rank_q90 = np.argsort(np.argsort(y_pred_q90))
    
    # True top-k
    true_top_idx = set(np.argsort(y_true)[-k:])
    
    results = []
    for lam in lambda_values:
        # Hybrid score
        hybrid_score = lam * rank_q90 + (1 - lam) * rank_mean
        
        # Top k by hybrid score
        hybrid_top_idx = set(np.argsort(hybrid_score)[-k:])
        
        # Metrics
        overlap = len(hybrid_top_idx & true_top_idx)
        precision = overlap / k
        recall = overlap / k
        lift = precision / 0.10
        
        # Compute top-10% bias using hybrid ranking
        hybrid_top_mask = np.zeros(n, dtype=bool)
        hybrid_top_mask[list(hybrid_top_idx)] = True
        
        # For bias, we need predictions - use weighted average of both
        hybrid_pred = lam * y_pred_q90 + (1 - lam) * y_pred_mean
        
        mean_true_top = np.average(y_true[hybrid_top_mask], weights=weights[hybrid_top_mask])
        mean_pred_top = np.average(hybrid_pred[hybrid_top_mask], weights=weights[hybrid_top_mask])
        top10_bias_pct = (mean_pred_top - mean_true_top) / mean_true_top * 100
        
        results.append({
            'lambda': lam,
            'precision_at_10': precision,
            'recall_at_10': recall,
            'lift_at_10': lift,
            'top10_bias_pct': top10_bias_pct,
            'abs_top10_bias_pct': abs(top10_bias_pct)
        })
    
    results_df = pd.DataFrame(results)
    
    # Find optimal λ: minimize |bias| while maintaining reasonable lift
    # Use Pareto-optimal selection
    min_acceptable_lift = results_df['lift_at_10'].max() * 0.95  # Within 5% of best lift
    valid_results = results_df[results_df['lift_at_10'] >= min_acceptable_lift]
    
    if len(valid_results) > 0:
        optimal_idx = valid_results['abs_top10_bias_pct'].idxmin()
    else:
        optimal_idx = results_df['abs_top10_bias_pct'].idxmin()
    
    optimal_lambda = results_df.loc[optimal_idx, 'lambda']
    
    return optimal_lambda, results_df


def plot_quantile_sweep_pareto(results: List[QuantileSweepResult], output_path: Path):
    """Create Pareto plot for quantile sweep."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    quantiles = [r.quantile for r in results]
    rmse = [r.weighted_rmse for r in results]
    bias = [r.top10_bias_pct for r in results]
    bias_lo = [r.top10_bias_pct_ci_lower for r in results]
    bias_hi = [r.top10_bias_pct_ci_upper for r in results]
    lift = [r.lift_at_10 for r in results]
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(quantiles)))
    
    # Plot 1: RMSE vs |Top-10% Bias|
    ax = axes[0]
    for i, (q, x, y_val) in enumerate(zip(quantiles, rmse, [abs(b) for b in bias])):
        ax.scatter(x, y_val, s=200, c=[colors[i]], edgecolors='black', linewidth=2, zorder=5)
        ax.annotate(f'q={q}', xy=(x, y_val), xytext=(8, 8), textcoords='offset points',
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))
    
    # Connect points with line
    ax.plot(rmse, [abs(b) for b in bias], 'k--', alpha=0.3, linewidth=1)
    
    ax.set_xlabel('Weighted RMSE (lower is better)', fontsize=12)
    ax.set_ylabel('|Top-10% Bias (%)| (lower is better)', fontsize=12)
    ax.set_title('Pareto Frontier: Accuracy vs Tail Bias\n(Quantile Sweep)', fontsize=13, fontweight='bold')
    
    # Ideal corner
    ax.annotate('← Ideal\n   Corner', xy=(min(rmse), min([abs(b) for b in bias])),
               xytext=(min(rmse) + 500, min([abs(b) for b in bias]) + 1),
               fontsize=10, color='green', style='italic')
    
    # Plot 2: Top-10% Bias with CI
    ax = axes[1]
    x_pos = np.arange(len(quantiles))
    
    ax.bar(x_pos, bias, color=colors, edgecolor='black', linewidth=1.5, alpha=0.7)
    ax.errorbar(x_pos, bias, 
                yerr=[np.array(bias) - np.array(bias_lo), np.array(bias_hi) - np.array(bias)],
                fmt='none', ecolor='black', capsize=5, capthick=2, elinewidth=2)
    
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.axhspan(-5, 5, alpha=0.1, color='green', label='Acceptable range (±5%)')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'q={q}' for q in quantiles], fontsize=11)
    ax.set_ylabel('Top-10% Bias (%) with 95% CI', fontsize=12)
    ax.set_title('Top-Decile Bias by Quantile\n(Negative = Underprediction)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    
    # Add significance markers
    for i, (lo, hi) in enumerate(zip(bias_lo, bias_hi)):
        if lo > 0 or hi < 0:  # CI excludes zero
            ax.annotate('*', xy=(i, bias[i]), xytext=(0, 10), textcoords='offset points',
                       ha='center', fontsize=16, fontweight='bold', color='blue')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_hybrid_targeting(results_df: pd.DataFrame, optimal_lambda: float, output_path: Path):
    """Plot hybrid targeting score optimization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Lift and Bias vs Lambda
    ax = axes[0]
    ax2 = ax.twinx()
    
    line1, = ax.plot(results_df['lambda'], results_df['lift_at_10'], 'b-o', 
                     linewidth=2, markersize=6, label='Lift@10')
    line2, = ax2.plot(results_df['lambda'], results_df['abs_top10_bias_pct'], 'r-s',
                      linewidth=2, markersize=6, label='|Bias|')
    
    ax.axvline(x=optimal_lambda, color='green', linestyle='--', linewidth=2, 
               label=f'Optimal λ={optimal_lambda:.2f}')
    
    ax.set_xlabel('λ (weight on Quantile ranking)', fontsize=12)
    ax.set_ylabel('Lift@10', color='blue', fontsize=12)
    ax2.set_ylabel('|Top-10% Bias (%)|', color='red', fontsize=12)
    ax.set_title('Hybrid Targeting Score Optimization\nλ·rank(q90) + (1-λ)·rank(mean)', 
                 fontsize=13, fontweight='bold')
    
    lines = [line1, line2]
    labels = ['Lift@10', '|Top-10% Bias (%)|']
    ax.legend(lines, labels, loc='center right')
    
    # Plot 2: Pareto curve
    ax = axes[1]
    scatter = ax.scatter(results_df['abs_top10_bias_pct'], results_df['lift_at_10'],
                        c=results_df['lambda'], cmap='viridis', s=100, edgecolors='black')
    
    # Highlight optimal
    opt_row = results_df[results_df['lambda'] == optimal_lambda].iloc[0]
    ax.scatter(opt_row['abs_top10_bias_pct'], opt_row['lift_at_10'],
              s=300, c='red', marker='*', edgecolors='black', linewidth=2, zorder=10,
              label=f'Optimal (λ={optimal_lambda:.2f})')
    
    plt.colorbar(scatter, ax=ax, label='λ')
    ax.set_xlabel('|Top-10% Bias (%)| (lower is better)', fontsize=12)
    ax.set_ylabel('Lift@10 (higher is better)', fontsize=12)
    ax.set_title('Pareto Trade-off: Targeting vs Bias', fontsize=13, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_metrics_with_ci(results: List[QuantileSweepResult], output_path: Path):
    """Plot all metrics with confidence intervals."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    quantiles = [r.quantile for r in results]
    x_pos = np.arange(len(quantiles))
    
    # Colors
    colors = plt.cm.Set2(np.linspace(0, 1, len(quantiles)))
    
    # Plot 1: Top-10% Bias with CI
    ax = axes[0, 0]
    bias = [r.top10_bias_pct for r in results]
    bias_lo = [r.top10_bias_pct_ci_lower for r in results]
    bias_hi = [r.top10_bias_pct_ci_upper for r in results]
    
    ax.bar(x_pos, bias, color=colors, edgecolor='black', alpha=0.7)
    ax.errorbar(x_pos, bias,
                yerr=[np.array(bias) - np.array(bias_lo), np.array(bias_hi) - np.array(bias)],
                fmt='none', ecolor='black', capsize=6, capthick=2)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'q={q}' for q in quantiles])
    ax.set_ylabel('Top-10% Bias (%)')
    ax.set_title('Top-10% Bias with 95% CI', fontweight='bold')
    
    # Add CI text
    for i, (b, lo, hi) in enumerate(zip(bias, bias_lo, bias_hi)):
        ax.annotate(f'{b:.1f}\n[{lo:.1f}, {hi:.1f}]', xy=(i, b),
                   xytext=(0, 10 if b >= 0 else -30), textcoords='offset points',
                   ha='center', fontsize=9)
    
    # Plot 2: Lift@10 with CI
    ax = axes[0, 1]
    lift = [r.lift_at_10 for r in results]
    lift_lo = [r.lift_at_10_ci_lower for r in results]
    lift_hi = [r.lift_at_10_ci_upper for r in results]
    
    ax.bar(x_pos, lift, color=colors, edgecolor='black', alpha=0.7)
    ax.errorbar(x_pos, lift,
                yerr=[np.array(lift) - np.array(lift_lo), np.array(lift_hi) - np.array(lift)],
                fmt='none', ecolor='black', capsize=6, capthick=2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'q={q}' for q in quantiles])
    ax.set_ylabel('Lift@10')
    ax.set_title('Lift@10 with 95% CI', fontweight='bold')
    
    # Plot 3: NDCG with CI
    ax = axes[1, 0]
    ndcg = [r.ndcg for r in results]
    ndcg_lo = [r.ndcg_ci_lower for r in results]
    ndcg_hi = [r.ndcg_ci_upper for r in results]
    
    ax.bar(x_pos, ndcg, color=colors, edgecolor='black', alpha=0.7)
    ax.errorbar(x_pos, ndcg,
                yerr=[np.array(ndcg) - np.array(ndcg_lo), np.array(ndcg_hi) - np.array(ndcg)],
                fmt='none', ecolor='black', capsize=6, capthick=2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'q={q}' for q in quantiles])
    ax.set_ylabel('NDCG')
    ax.set_title('NDCG with 95% CI', fontweight='bold')
    
    # Plot 4: wRMSE and wR² (no CI - deterministic)
    ax = axes[1, 1]
    rmse = [r.weighted_rmse for r in results]
    r2 = [r.weighted_r2 for r in results]
    
    ax2 = ax.twinx()
    bars1 = ax.bar(x_pos - 0.2, rmse, 0.35, color='steelblue', alpha=0.7, label='wRMSE')
    bars2 = ax2.bar(x_pos + 0.2, r2, 0.35, color='coral', alpha=0.7, label='wR²')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'q={q}' for q in quantiles])
    ax.set_ylabel('wRMSE', color='steelblue')
    ax2.set_ylabel('wR²', color='coral')
    ax.set_title('Accuracy Metrics', fontweight='bold')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.suptitle('Quantile Sweep: All Metrics with 95% CI (SDR)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_summary_table(results: List[QuantileSweepResult]) -> pd.DataFrame:
    """Create summary table with CI."""
    rows = []
    for r in results:
        rows.append({
            'Quantile': f'q={r.quantile}',
            'wRMSE': f'{r.weighted_rmse:.0f}',
            'wMAE': f'{r.weighted_mae:.0f}',
            'wR²': f'{r.weighted_r2:.3f}',
            'Top-10% Bias (%)': f'{r.top10_bias_pct:.1f} [{r.top10_bias_pct_ci_lower:.1f}, {r.top10_bias_pct_ci_upper:.1f}]',
            'Lift@10': f'{r.lift_at_10:.2f} [{r.lift_at_10_ci_lower:.2f}, {r.lift_at_10_ci_upper:.2f}]',
            'NDCG': f'{r.ndcg:.3f} [{r.ndcg_ci_lower:.3f}, {r.ndcg_ci_upper:.3f}]',
            'Precision@10': f'{r.precision_at_10:.3f}',
            'Recall@10': f'{r.recall_at_10:.3f}'
        })
    return pd.DataFrame(rows)


def main():
    """Run advanced tail bias analysis."""
    output_dir = Path('/workspace/outputs_tail_bias')
    figures_dir = output_dir / 'figures'
    tables_dir = output_dir / 'tables'
    ensure_dir(figures_dir)
    ensure_dir(tables_dir)
    
    print("\n" + "="*70)
    print("ADVANCED TAIL BIAS ANALYSIS")
    print("Quantile Sweep + Hybrid Targeting + CI with Replicate Weights")
    print("="*70 + "\n")
    
    # Load and preprocess data
    with Timer("Loading data"):
        df = load_recs_data('/workspace/data/raw/recs2020_public_v7.csv')
    
    with Timer("Preprocessing"):
        preprocessor = RECSPreprocessor()
        df = preprocessor.assign_technology_groups(df)
        df = preprocessor.filter_for_analysis(df, exclude_no_heating=True)
        # Remove zero energy values
        df = df[df['TOTALBTUSPH'] > 0].copy()
        df = preprocessor.create_derived_features(df)
    
    with Timer("Feature engineering"):
        feature_builder = FeatureBuilder()
        X_df = feature_builder.fit_transform(df)
        X = X_df.values
        feature_names = list(X_df.columns)
    
    # Extract arrays
    y = df['TOTALBTUSPH'].values
    weights = df['NWEIGHT'].values
    
    # Get replicate weights
    replicate_cols = [f'NWEIGHT{i}' for i in range(1, 61)]
    replicate_weights = df[replicate_cols].values
    
    print(f"\nData shape: {X.shape}")
    print(f"Replicate weights: {replicate_weights.shape}")
    
    # ===== 5.1: Quantile Sweep =====
    print("\n" + "="*70)
    print("5.1 QUANTILE SWEEP")
    print("="*70)
    
    quantiles = [0.80, 0.85, 0.90, 0.95]
    sweep_results = run_quantile_sweep(
        df, X, y, weights, replicate_weights,
        quantiles=quantiles,
        n_folds=3,
        random_state=42
    )
    
    # Create summary table
    summary_df = create_summary_table(sweep_results)
    summary_df.to_csv(tables_dir / 'quantile_sweep_summary.csv', index=False)
    print(f"\nSaved: {tables_dir / 'quantile_sweep_summary.csv'}")
    print(summary_df.to_string())
    
    # Create Pareto plot
    plot_quantile_sweep_pareto(sweep_results, figures_dir / 'fig7_quantile_sweep_pareto.png')
    
    # Create metrics with CI plot
    plot_metrics_with_ci(sweep_results, figures_dir / 'fig8_quantile_sweep_ci.png')
    
    # ===== 5.2: Hybrid Targeting Score =====
    print("\n" + "="*70)
    print("5.2 HYBRID TARGETING SCORE")
    print("="*70)
    
    # Get mean predictions (baseline) and q90 predictions
    # Use the best quantile from sweep for q90
    best_q_for_bias = min(sweep_results, key=lambda r: abs(r.top10_bias_pct))
    print(f"\nBest quantile for bias: q={best_q_for_bias.quantile} (bias={best_q_for_bias.top10_bias_pct:.1f}%)")
    
    # Re-run to get predictions
    from sklearn.model_selection import StratifiedKFold
    import lightgbm as lgb
    
    y_binned = pd.qcut(y, q=10, labels=False, duplicates='drop')
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    y_pred_mean = np.zeros(len(y))
    y_pred_q90 = np.zeros(len(y))
    
    print("\nGenerating predictions for hybrid score...")
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y_binned)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        w_train = weights[train_idx]
        
        # Mean model (Tweedie)
        model_mean = lgb.LGBMRegressor(
            objective='tweedie', tweedie_variance_power=1.5,
            n_estimators=200, learning_rate=0.05, max_depth=6,
            random_state=42 + fold_idx, verbose=-1
        )
        model_mean.fit(X_train, y_train, sample_weight=w_train)
        y_pred_mean[test_idx] = model_mean.predict(X_test)
        
        # Quantile model (q=0.90)
        model_q90 = QuantileLightGBMHeatingModel(
            quantile=0.90,
            params={
                'n_estimators': 200,
                'learning_rate': 0.05,
                'max_depth': 6,
                'random_state': 42 + fold_idx
            }
        )
        model_q90.fit(X_train, y_train, sample_weight=w_train)
        y_pred_q90[test_idx] = model_q90.predict(X_test)
    
    # Optimize hybrid score
    optimal_lambda, hybrid_results = compute_hybrid_targeting_score(
        y_pred_mean, y_pred_q90, y, weights
    )
    
    print(f"\nOptimal λ: {optimal_lambda:.2f}")
    opt_result = hybrid_results[hybrid_results['lambda'] == optimal_lambda].iloc[0]
    print(f"  Lift@10: {opt_result['lift_at_10']:.2f}")
    print(f"  Top-10% Bias: {opt_result['top10_bias_pct']:.1f}%")
    
    # Save results
    hybrid_results.to_csv(tables_dir / 'hybrid_targeting_sweep.csv', index=False)
    print(f"Saved: {tables_dir / 'hybrid_targeting_sweep.csv'}")
    
    # Plot
    plot_hybrid_targeting(hybrid_results, optimal_lambda, 
                         figures_dir / 'fig9_hybrid_targeting.png')
    
    # ===== Summary =====
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print("\nNew tables:")
    print(f"  - quantile_sweep_summary.csv")
    print(f"  - hybrid_targeting_sweep.csv")
    print("\nNew figures:")
    print(f"  - fig7_quantile_sweep_pareto.png")
    print(f"  - fig8_quantile_sweep_ci.png")
    print(f"  - fig9_hybrid_targeting.png")
    
    # Key findings
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    
    best_rmse = min(sweep_results, key=lambda r: r.weighted_rmse)
    best_bias = min(sweep_results, key=lambda r: abs(r.top10_bias_pct))
    best_lift = max(sweep_results, key=lambda r: r.lift_at_10)
    
    print(f"\n1. Best for RMSE: q={best_rmse.quantile} (wRMSE={best_rmse.weighted_rmse:.0f})")
    print(f"2. Best for Tail Bias: q={best_bias.quantile} (bias={best_bias.top10_bias_pct:.1f}%)")
    print(f"3. Best for Lift: q={best_lift.quantile} (Lift={best_lift.lift_at_10:.2f})")
    print(f"4. Optimal Hybrid λ: {optimal_lambda:.2f}")
    
    # Recommendation
    print("\n" + "-"*70)
    print("RECOMMENDATION FOR APPLIED ENERGY")
    print("-"*70)
    if abs(best_bias.top10_bias_pct) < 5:
        print(f"\n✓ Quantile q={best_bias.quantile} achieves acceptable tail bias")
        print(f"  (|bias| = {abs(best_bias.top10_bias_pct):.1f}% < 5% threshold)")
        print(f"  with CI: [{best_bias.top10_bias_pct_ci_lower:.1f}, {best_bias.top10_bias_pct_ci_upper:.1f}]")
    else:
        print(f"\n⚠ Best quantile q={best_bias.quantile} still has |bias| > 5%")
        print(f"  Consider hybrid targeting with λ={optimal_lambda:.2f}")


if __name__ == '__main__':
    main()
