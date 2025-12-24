"""
Model Comparison Module for H1 Testing

Creates outer-fold comparison tables:
- Monolithic vs Split models
- Per-fold metrics + Mean ± SD
- Paired statistical tests
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats

from src.utils.helpers import logger


def create_split_vs_monolithic_table(
    split_fold_metrics: pd.DataFrame,
    mono_fold_metrics: pd.DataFrame,
    baseline_fold_metrics: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Create outer-fold comparison table (5 rows per model + summary).
    
    Parameters
    ----------
    split_fold_metrics : DataFrame
        Per-fold metrics for split model
    mono_fold_metrics : DataFrame
        Per-fold metrics for monolithic model
    baseline_fold_metrics : DataFrame, optional
        Per-fold metrics for physics baseline
        
    Returns
    -------
    DataFrame
        Comparison table with per-fold and summary rows
    """
    rows = []
    
    metrics_cols = ['weighted_rmse', 'weighted_mae', 'weighted_r2']
    
    # Helper to add model rows
    def add_model_rows(fold_df, model_name):
        for fold_idx in range(len(fold_df)):
            row = {'Model': model_name, 'Fold': fold_idx + 1}
            for col in metrics_cols:
                if col in fold_df.columns:
                    row[col] = fold_df.iloc[fold_idx][col]
            rows.append(row)
        
        # Summary row
        summary = {'Model': model_name, 'Fold': 'Mean ± SD'}
        for col in metrics_cols:
            if col in fold_df.columns:
                mean_val = fold_df[col].mean()
                std_val = fold_df[col].std()
                if col == 'weighted_r2':
                    summary[col] = f"{mean_val:.4f} ± {std_val:.4f}"
                else:
                    summary[col] = f"{mean_val:,.0f} ± {std_val:,.0f}"
        rows.append(summary)
    
    # Add baseline if provided
    if baseline_fold_metrics is not None:
        add_model_rows(baseline_fold_metrics, 'Physics Baseline')
    
    # Add monolithic
    add_model_rows(mono_fold_metrics, 'Monolithic LightGBM')
    
    # Add split
    add_model_rows(split_fold_metrics, 'Split by Technology')
    
    result_df = pd.DataFrame(rows)
    
    # Rename columns for publication
    result_df = result_df.rename(columns={
        'weighted_rmse': 'wRMSE (kBTU)',
        'weighted_mae': 'wMAE (kBTU)',
        'weighted_r2': 'wR²'
    })
    
    return result_df


def compute_paired_comparison(
    split_fold_metrics: pd.DataFrame,
    mono_fold_metrics: pd.DataFrame,
    metric: str = 'weighted_rmse'
) -> Dict[str, Any]:
    """
    Compute paired comparison statistics between split and monolithic.
    
    Parameters
    ----------
    split_fold_metrics : DataFrame
        Per-fold metrics for split model
    mono_fold_metrics : DataFrame
        Per-fold metrics for monolithic model
    metric : str
        Metric to compare
        
    Returns
    -------
    dict
        Paired comparison statistics
    """
    split_vals = split_fold_metrics[metric].values
    mono_vals = mono_fold_metrics[metric].values
    
    if len(split_vals) != len(mono_vals):
        raise ValueError("Fold counts must match")
    
    # Paired t-test (one-sided: split < mono for error metrics)
    if 'r2' in metric:
        # For R², higher is better
        t_stat, p_two = stats.ttest_rel(split_vals, mono_vals)
        p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
        improvement = split_vals.mean() - mono_vals.mean()
        improvement_pct = improvement / mono_vals.mean() * 100
    else:
        # For error metrics, lower is better
        t_stat, p_two = stats.ttest_rel(mono_vals, split_vals)
        p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
        improvement = mono_vals.mean() - split_vals.mean()
        improvement_pct = improvement / mono_vals.mean() * 100
    
    # Differences per fold
    if 'r2' in metric:
        diffs = split_vals - mono_vals
    else:
        diffs = mono_vals - split_vals
    
    return {
        'metric': metric,
        'split_mean': split_vals.mean(),
        'split_std': split_vals.std(),
        'mono_mean': mono_vals.mean(),
        'mono_std': mono_vals.std(),
        'improvement': improvement,
        'improvement_pct': improvement_pct,
        't_statistic': t_stat,
        'p_value_two_sided': p_two,
        'p_value_one_sided': p_one,
        'significant_at_05': p_one < 0.05,
        'fold_differences': diffs,
        'all_folds_improved': all(diffs > 0)
    }


def create_h1_summary_table(
    split_fold_metrics: pd.DataFrame,
    mono_fold_metrics: pd.DataFrame
) -> pd.DataFrame:
    """
    Create H1 hypothesis test summary table.
    
    Tests: Split models reduce outer-fold error vs monolithic.
    
    Parameters
    ----------
    split_fold_metrics : DataFrame
        Per-fold metrics for split model
    mono_fold_metrics : DataFrame
        Per-fold metrics for monolithic model
        
    Returns
    -------
    DataFrame
        H1 test summary
    """
    results = []
    
    for metric in ['weighted_rmse', 'weighted_mae', 'weighted_r2']:
        if metric not in split_fold_metrics.columns:
            continue
        
        comp = compute_paired_comparison(split_fold_metrics, mono_fold_metrics, metric)
        
        metric_label = {
            'weighted_rmse': 'wRMSE',
            'weighted_mae': 'wMAE',
            'weighted_r2': 'wR²'
        }.get(metric, metric)
        
        results.append({
            'Metric': metric_label,
            'Split (Mean ± SD)': f"{comp['split_mean']:.3f} ± {comp['split_std']:.3f}" if 'r2' in metric else f"{comp['split_mean']:,.0f} ± {comp['split_std']:,.0f}",
            'Monolithic (Mean ± SD)': f"{comp['mono_mean']:.3f} ± {comp['mono_std']:.3f}" if 'r2' in metric else f"{comp['mono_mean']:,.0f} ± {comp['mono_std']:,.0f}",
            'Improvement': f"{comp['improvement']:.3f}" if 'r2' in metric else f"{comp['improvement']:,.0f}",
            'Improvement (%)': f"{comp['improvement_pct']:.1f}%",
            'p-value (one-sided)': f"{comp['p_value_one_sided']:.4f}",
            'Significant (α=0.05)': '✓' if comp['significant_at_05'] else '✗'
        })
    
    result_df = pd.DataFrame(results)
    result_df.attrs['note'] = (
        "H1 test: Split-by-technology models reduce outer-fold generalization error "
        "compared to monolithic model. Paired t-test across 5 outer folds. "
        "All metrics are weighted."
    )
    
    return result_df


def create_physics_diagnostic_summary(
    split_diagnostics: Dict[str, Any],
    mono_diagnostics: Dict[str, Any]
) -> pd.DataFrame:
    """
    Create physics diagnostic comparison table.
    
    Compares:
    - Negative prediction rate
    - HDD sensitivity direction
    - Local monotonicity violations
    
    Parameters
    ----------
    split_diagnostics : dict
        Diagnostics for split model
    mono_diagnostics : dict
        Diagnostics for monolithic model
        
    Returns
    -------
    DataFrame
        Physics diagnostic comparison
    """
    rows = []
    
    # Negative prediction rate
    rows.append({
        'Diagnostic': 'Negative Prediction Rate',
        'Split': f"{split_diagnostics.get('negative_rate', 0)*100:.2f}%",
        'Monolithic': f"{mono_diagnostics.get('negative_rate', 0)*100:.2f}%",
        'Expected': '0%',
        'Notes': 'Lower is better'
    })
    
    # HDD sensitivity
    split_hdd = split_diagnostics.get('hdd_sensitivity', {})
    mono_hdd = mono_diagnostics.get('hdd_sensitivity', {})
    
    rows.append({
        'Diagnostic': 'HDD Correlation',
        'Split': f"{split_hdd.get('correlation', 0):.3f}",
        'Monolithic': f"{mono_hdd.get('correlation', 0):.3f}",
        'Expected': '> 0 (positive)',
        'Notes': 'Higher HDD → Higher energy'
    })
    
    rows.append({
        'Diagnostic': 'Correct HDD Direction',
        'Split': '✓' if split_hdd.get('is_correct_direction', False) else '✗',
        'Monolithic': '✓' if mono_hdd.get('is_correct_direction', False) else '✗',
        'Expected': '✓',
        'Notes': 'Must be positive'
    })
    
    result_df = pd.DataFrame(rows)
    result_df.attrs['note'] = (
        "Physics plausibility diagnostics. Split models should show better "
        "physical behavior (correct HDD sensitivity, fewer negative predictions)."
    )
    
    return result_df
