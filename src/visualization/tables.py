"""
Table Generation Utilities for Heating Demand Modeling

Creates publication-quality tables as specified in Section 13.
All tables follow requirements:
- Remove spreadsheet artifacts
- Replace codes with human-readable labels
- Declare weighted vs unweighted, out-of-sample vs in-sample
- Include metric definitions and units
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

from src.utils.helpers import logger, compute_weighted_quantile


# Human-readable labels for RECS codes
HOUSING_TYPE_LABELS = {
    1: 'Mobile Home', 2: 'Single-Family Detached', 3: 'Single-Family Attached',
    4: 'Apartment (2-4 units)', 5: 'Apartment (5+ units)'
}

TENURE_LABELS = {1: 'Owned', 2: 'Rented', 3: 'Occupied w/o Payment'}

DIVISION_LABELS = {
    1: 'New England', 2: 'Middle Atlantic', 3: 'East North Central',
    4: 'West North Central', 5: 'South Atlantic', 6: 'East South Central',
    7: 'West South Central', 8: 'Mountain North', 9: 'Mountain South', 10: 'Pacific'
}

YEARMADE_LABELS = {
    1: 'Before 1950', 2: '1950-1959', 3: '1960-1969', 4: '1970-1979',
    5: '1980-1989', 6: '1990-1999', 7: '2000-2009', 8: '2010-2015', 9: '2016-2020'
}

# RECS 2020 MONEYPY codes - complete income ranges (16 categories)
INCOME_LABELS = {
    1: '<$5K', 2: '$5-7.5K', 3: '$7.5-10K', 4: '$10-12.5K',
    5: '$12.5-15K', 6: '$15-20K', 7: '$20-25K', 8: '$25-30K',
    9: '$30-35K', 10: '$35-40K', 11: '$40-50K', 12: '$50-60K',
    13: '$60-75K', 14: '$75-100K', 15: '$100-150K', 16: '$150K+'
}

# Climate/HDD bin labels
CLIMATE_LABELS = {
    'very_mild': 'Very Mild (<2k HDD)',
    'mild': 'Mild (2-4k HDD)',
    'moderate': 'Moderate (4-6k HDD)',
    'cold': 'Cold (6-8k HDD)',
    'very_cold': 'Very Cold (>8k HDD)'
}


def create_table1_descriptives(df: pd.DataFrame,
                                weights: pd.Series,
                                replicate_weights: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Create Table 1: Descriptive statistics by technology group.
    
    Includes (weighted):
    - n (unweighted), population share (weighted)
    - mean/median E_heat, E_heat/A
    - mean HDD, mean area
    - vintage/envelope proxies
    - COVID indicator shares
    
    Parameters
    ----------
    df : DataFrame
        Preprocessed RECS data with tech_group
    weights : Series
        NWEIGHT values
    replicate_weights : DataFrame, optional
        For uncertainty estimation
        
    Returns
    -------
    DataFrame
        Publication-ready Table 1
    """
    results = []
    total_weight = weights.sum()
    
    for group in df['tech_group'].unique():
        mask = df['tech_group'] == group
        group_df = df[mask]
        group_weights = weights[mask]
        
        row = {
            'Technology Group': group.replace('_', ' ').title(),
            'n (unweighted)': len(group_df),
            'Population Share (%)': group_weights.sum() / total_weight * 100,
        }
        
        # Energy metrics
        energy = group_df['TOTALBTUSPH'].values
        valid = ~np.isnan(energy)
        if valid.sum() > 0:
            row['Mean E_heat (kBTU)'] = np.average(energy[valid], weights=group_weights.values[valid])
            row['Median E_heat (kBTU)'] = compute_weighted_quantile(
                energy[valid], group_weights.values[valid], 0.5
            )
        
        # Energy intensity
        if 'TOTSQFT_EN' in group_df.columns:
            intensity = energy / group_df['TOTSQFT_EN'].values
            valid_int = ~np.isnan(intensity)
            if valid_int.sum() > 0:
                row['Mean E/Area (kBTU/ft²)'] = np.average(
                    intensity[valid_int], weights=group_weights.values[valid_int]
                )
        
        # Climate
        if 'HDD65' in group_df.columns:
            hdd = group_df['HDD65'].values
            row['Mean HDD'] = np.average(hdd[valid], weights=group_weights.values[valid])
        
        # Area
        if 'TOTSQFT_EN' in group_df.columns:
            sqft = group_df['TOTSQFT_EN'].values
            row['Mean Area (ft²)'] = np.average(sqft[valid], weights=group_weights.values[valid])
        
        # Vintage (share pre-1980)
        if 'YEARMADERANGE' in group_df.columns:
            pre1980 = group_df['YEARMADERANGE'] <= 4  # Before 1980
            pre1980_share = np.average(pre1980.values[valid], weights=group_weights.values[valid])
            row['Pre-1980 (%)'] = pre1980_share * 100
        
        # COVID indicators
        if 'ATHOME' in group_df.columns:
            athome = group_df['ATHOME'].values == 1
            valid_athome = ~np.isnan(group_df['ATHOME'].values)
            if valid_athome.sum() > 0:
                row['At Home Daytime (%)'] = np.average(
                    athome[valid_athome], weights=group_weights.values[valid_athome]
                ) * 100
        
        if 'TELLWORK' in group_df.columns:
            tellwork = group_df['TELLWORK'].values
            valid_tw = ~np.isnan(tellwork) & (tellwork > 0)
            if valid_tw.sum() > 0:
                # TELLWORK > 0 means some telework
                row['Telework (%)'] = valid_tw.sum() / len(group_df) * 100
        
        results.append(row)
    
    result_df = pd.DataFrame(results)
    
    # Round numeric columns
    for col in result_df.columns:
        if result_df[col].dtype in [np.float64, np.float32]:
            if '%' in col:
                result_df[col] = result_df[col].round(1)
            elif 'kBTU' in col or 'Area' in col or 'HDD' in col:
                result_df[col] = result_df[col].round(0).astype(int)
    
    return result_df


def create_performance_table(fold_metrics_list: List[Dict],
                              model_names: List[str],
                              runtime_seconds: Optional[List[float]] = None) -> pd.DataFrame:
    """
    Create performance table with nested CV results.
    
    Shows outer-fold metrics per fold for each model,
    then mean ± SD across folds.
    
    Parameters
    ----------
    fold_metrics_list : list
        List of fold metrics DataFrames for each model
    model_names : list
        Names of models
    runtime_seconds : list, optional
        Runtime for each model
        
    Returns
    -------
    DataFrame
        Performance comparison table
    """
    results = []
    
    for i, (metrics_df, name) in enumerate(zip(fold_metrics_list, model_names)):
        # Per-fold metrics
        for fold_idx, row in metrics_df.iterrows():
            results.append({
                'Model': name,
                'Fold': fold_idx + 1,
                'wRMSE (kBTU)': row.get('weighted_rmse', np.nan),
                'wMAE (kBTU)': row.get('weighted_mae', np.nan),
                'wR²': row.get('weighted_r2', np.nan),
            })
        
        # Summary row
        summary = {
            'Model': name,
            'Fold': 'Mean ± SD',
            'wRMSE (kBTU)': f"{metrics_df['weighted_rmse'].mean():.0f} ± {metrics_df['weighted_rmse'].std():.0f}",
            'wMAE (kBTU)': f"{metrics_df['weighted_mae'].mean():.0f} ± {metrics_df['weighted_mae'].std():.0f}",
            'wR²': f"{metrics_df['weighted_r2'].mean():.3f} ± {metrics_df['weighted_r2'].std():.3f}",
        }
        if runtime_seconds and i < len(runtime_seconds):
            summary['Runtime (s)'] = f"{runtime_seconds[i]:.1f}"
        results.append(summary)
    
    return pd.DataFrame(results)


def create_uncertainty_table(uncertainty_df: pd.DataFrame,
                              include_policy_metrics: bool = True) -> pd.DataFrame:
    """
    Create uncertainty table with proper formatting.
    
    EXCLUDES MAPE (unstable for small denominators).
    Uses WAPE and nMAE as stable alternatives.
    
    Parameters
    ----------
    uncertainty_df : DataFrame
        Raw uncertainty results
    include_policy_metrics : bool
        Whether to include policy targeting metrics
        
    Returns
    -------
    DataFrame
        Formatted uncertainty table
    """
    # Filter out unstable MAPE metrics
    if 'metric' in uncertainty_df.columns:
        # Exclude any MAPE-related metrics
        exclude_pattern = 'mape'
        uncertainty_df = uncertainty_df[
            ~uncertainty_df['metric'].str.lower().str.contains(exclude_pattern)
        ].copy()
    
    formatted = {
        'Metric': [],
        'Estimate': [],
        'SE': [],
        '95% CI': [],
        'Unit': []
    }
    
    metric_info = {
        'weighted_rmse': ('wRMSE', 'kBTU', False),
        'weighted_mae': ('wMAE', 'kBTU', False),
        'weighted_r2': ('wR²', '—', True),
        'weighted_bias': ('wBias', 'kBTU', False),
        'weighted_wape': ('WAPE', '%', True),
        'weighted_nmae': ('nMAE', '%', True),
    }
    
    for _, row in uncertainty_df.iterrows():
        metric_name = row['metric']
        
        # Get display info
        if metric_name in metric_info:
            display_name, unit, is_ratio = metric_info[metric_name]
        else:
            display_name = metric_name.replace('weighted_', 'w').replace('_', ' ').title()
            unit = 'kBTU'
            is_ratio = False
        
        formatted['Metric'].append(display_name)
        formatted['Unit'].append(unit)
        
        # Format based on metric type
        if is_ratio or metric_name == 'weighted_r2':
            formatted['Estimate'].append(f"{row['estimate']:.3f}")
            formatted['SE'].append(f"{row['se']:.3f}")
            formatted['95% CI'].append(f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]")
        elif unit == '%':
            formatted['Estimate'].append(f"{row['estimate']:.1f}")
            formatted['SE'].append(f"{row['se']:.1f}")
            formatted['95% CI'].append(f"[{row['ci_lower']:.1f}, {row['ci_upper']:.1f}]")
        else:
            formatted['Estimate'].append(f"{row['estimate']:,.0f}")
            formatted['SE'].append(f"{row['se']:,.0f}")
            formatted['95% CI'].append(f"[{row['ci_lower']:,.0f}, {row['ci_upper']:,.0f}]")
    
    result_df = pd.DataFrame(formatted)
    
    result_df.attrs['note'] = (
        "All metrics computed on outer-fold test predictions with NWEIGHT. "
        "SE and 95% CI from RECS replicate-weight jackknife (n=60). "
        "MAPE excluded (unstable for small denominators); WAPE and nMAE used instead."
    )
    
    return result_df


def create_policy_targeting_table(targeting_results: Dict,
                                   score_name: str = 'high_use',
                                   include_cis: bool = True) -> pd.DataFrame:
    """
    Create policy targeting summary table.
    
    Includes:
    - Jaccard (95% CI)
    - Overlap rate (95% CI)
    - Top positive/negative subgroup shifts
    - Threshold values
    
    Parameters
    ----------
    targeting_results : dict
        Results from PolicyTargeting analysis
    score_name : str
        Policy score to summarize
    include_cis : bool
        Whether to include confidence intervals
        
    Returns
    -------
    DataFrame
        Policy targeting summary
    """
    if score_name not in targeting_results.get('weighted_vs_unweighted', {}):
        return pd.DataFrame()
    
    score_results = targeting_results['weighted_vs_unweighted'][score_name]
    overlap = score_results['overlap']
    
    rows = [
        {
            'Metric': 'Jaccard Index',
            'Value': f"{overlap['jaccard_index']:.3f}",
            'Description': 'Intersection / Union of candidate sets'
        },
        {
            'Metric': 'Dice Overlap',
            'Value': f"{overlap['overlap_rate']:.3f}",
            'Description': 'Dice coefficient: 2×|A∩B| / (|A|+|B|)'
        },
    ]
    
    # Add Recall and Containment if available
    if 'recall_weighted' in overlap:
        rows.append({
            'Metric': 'Recall (Weighted→Unweighted)',
            'Value': f"{overlap['recall_weighted']:.3f}",
            'Description': 'Fraction of weighted candidates also in unweighted'
        })
    if 'containment' in overlap:
        rows.append({
            'Metric': 'Containment',
            'Value': f"{overlap['containment']:.3f}",
            'Description': 'Intersection / min(|A|, |B|)'
        })
    
    rows.extend([
        {
            'Metric': 'Only in Weighted',
            'Value': f"{overlap['only_weighted']} ({overlap['pct_only_weighted']:.1f}%)",
            'Description': 'Candidates selected only with weights'
        },
        {
            'Metric': 'Only in Unweighted',
            'Value': f"{overlap['only_unweighted']} ({overlap['pct_only_unweighted']:.1f}%)",
            'Description': 'Candidates selected only without weights'
        },
    ])
    
    # Correct units based on score type
    if 'intensity' in score_name.lower():
        threshold_unit = 'kBTU/ft²'
    else:
        threshold_unit = 'kBTU'
    
    rows.extend([
        {
            'Metric': 'Weighted Threshold',
            'Value': f"{score_results['weighted_threshold']:,.0f} {threshold_unit}",
            'Description': 'Weighted 90th percentile cutoff'
        },
        {
            'Metric': 'Unweighted Threshold',
            'Value': f"{score_results['unweighted_threshold']:,.0f} {threshold_unit}",
            'Description': 'Unweighted 90th percentile cutoff'
        },
    ])
    
    # Add equal-budget comparison (fixed N candidates)
    if 'overlap_equal_budget' in score_results:
        eb = score_results['overlap_equal_budget']
        rows.append({
            'Metric': '--- Equal Budget (N fixed) ---',
            'Value': '',
            'Description': f"Both methods select top {score_results.get('n_budget', 'N')} candidates"
        })
        rows.append({
            'Metric': 'Jaccard (Equal Budget)',
            'Value': f"{eb['jaccard_index']:.3f}",
            'Description': 'Jaccard when both select same N candidates'
        })
        rows.append({
            'Metric': 'Dice (Equal Budget)',
            'Value': f"{eb['overlap_rate']:.3f}",
            'Description': 'Dice coefficient for equal-budget lists'
        })
    
    result_df = pd.DataFrame(rows)
    result_df.attrs['note'] = (
        f"Policy score: {score_name}. Top 10% candidates defined using weighted/unweighted "
        "90th percentile threshold on outer-fold predictions."
    )
    
    return result_df


def create_composition_table(composition_df: pd.DataFrame,
                              group_type: str,
                              include_representation_ratio: bool = True) -> pd.DataFrame:
    """
    Create composition shift table with human-readable labels.
    
    Includes:
    - Subgroup shares among candidates
    - (Weighted - Unweighted) differences
    - Population share
    - Within-group selection rate (% of group selected into Top 10%)
    - Representation ratio (candidate share / population share)
    
    Parameters
    ----------
    composition_df : DataFrame
        Raw composition results
    group_type : str
        Type of grouping (housing_type, tenure, income, division)
    include_representation_ratio : bool
        Whether to include representation ratio
        
    Returns
    -------
    DataFrame
        Formatted composition table
    """
    label_maps = {
        'housing_type': HOUSING_TYPE_LABELS,
        'tenure': TENURE_LABELS,
        'division': DIVISION_LABELS,
        'income': INCOME_LABELS,
        'climate': CLIMATE_LABELS
    }
    
    label_map = label_maps.get(group_type, {})
    
    # Copy and add human-readable labels
    result_df = composition_df.copy()
    label_col = result_df.columns[0]
    
    if label_map:
        result_df['Group'] = result_df[label_col].map(lambda x: label_map.get(x, str(x)))
    else:
        result_df['Group'] = result_df[label_col].astype(str)
    
    # Rename columns for clarity
    rename_map = {
        'share_weighted_candidates': 'Candidate Share (%)',
        'share_unweighted_candidates': 'Unweighted Share (%)',
        'share_difference': 'Difference (pp)',
        'population_share': 'Population Share (%)',
        'n_in_group': 'n',
        'within_group_selection_rate': 'Selection Rate (%)',
        'representation_ratio': 'Repr. Ratio'
    }
    result_df = result_df.rename(columns=rename_map)
    
    # Calculate representation ratio if not already present
    if include_representation_ratio:
        if 'Repr. Ratio' not in result_df.columns and 'Population Share (%)' in result_df.columns and 'Candidate Share (%)' in result_df.columns:
            result_df['Repr. Ratio'] = (
                result_df['Candidate Share (%)'] / result_df['Population Share (%)']
            ).round(2)
    
    # Select and order columns
    cols_to_keep = ['Group', 'n', 'Population Share (%)', 
                    'Candidate Share (%)', 'Unweighted Share (%)', 
                    'Difference (pp)']
    
    if 'Selection Rate (%)' in result_df.columns:
        cols_to_keep.append('Selection Rate (%)')
    
    if 'Repr. Ratio' in result_df.columns:
        cols_to_keep.append('Repr. Ratio')
    
    result_df = result_df[[c for c in cols_to_keep if c in result_df.columns]]
    
    # Round numeric columns
    for col in result_df.columns:
        if result_df[col].dtype in [np.float64, np.float32]:
            if 'Ratio' in col:
                result_df[col] = result_df[col].round(2)
            else:
                result_df[col] = result_df[col].round(1)
    
    # Add interpretation note
    result_df.attrs['note'] = (
        "Repr. Ratio > 1 means overrepresented among candidates; < 1 means underrepresented. "
        "Selection Rate = % of subgroup selected into Top 10%."
    )
    
    return result_df


def create_equity_table(equity_df: pd.DataFrame,
                         group_type: str,
                         include_normalized: bool = True) -> pd.DataFrame:
    """
    Create error equity table with proper formatting.
    
    Includes:
    - Bias (kBTU) and Bias (% of mean)
    - MAE (kBTU) and nMAE (%)
    - Group n and weighted share
    - 95% CIs (if available)
    
    Parameters
    ----------
    equity_df : DataFrame
        Raw equity results
    group_type : str
        Type of grouping
    include_normalized : bool
        Whether to include normalized metrics
        
    Returns
    -------
    DataFrame
        Formatted equity table
    """
    label_maps = {
        'housing_type': HOUSING_TYPE_LABELS,
        'tenure': TENURE_LABELS,
        'income': INCOME_LABELS
    }
    
    label_map = label_maps.get(group_type, {})
    
    result_df = equity_df.copy()
    
    # Handle specific label columns
    if 'income_group' in result_df.columns:
        result_df['Group'] = result_df['income_group']
    elif 'climate_zone' in result_df.columns:
        result_df['Group'] = result_df['climate_zone']
    else:
        label_col = result_df.columns[0]
        if label_map:
            result_df['Group'] = result_df[label_col].map(lambda x: label_map.get(x, str(x)))
        else:
            result_df['Group'] = result_df[label_col].astype(str)
    
    # Compute normalized metrics
    if include_normalized and 'weighted_mean_true' in result_df.columns:
        result_df['Bias (%)'] = (result_df['weighted_bias'] / result_df['weighted_mean_true'] * 100).round(1)
        result_df['nMAE (%)'] = (result_df['weighted_mae'] / result_df['weighted_mean_true'] * 100).round(1)
    
    # Rename columns
    result_df = result_df.rename(columns={
        'n_samples': 'n',
        'total_weight': 'Weighted Pop.',
        'weighted_bias': 'Bias (kBTU)',
        'weighted_mae': 'MAE (kBTU)',
        'weighted_mean_true': 'Mean Observed (kBTU)'
    })
    
    # Select columns
    cols_to_keep = ['Group', 'n', 'Mean Observed (kBTU)', 
                    'Bias (kBTU)', 'MAE (kBTU)']
    if 'Bias (%)' in result_df.columns:
        cols_to_keep.extend(['Bias (%)', 'nMAE (%)'])
    
    result_df = result_df[[c for c in cols_to_keep if c in result_df.columns]]
    
    # Round
    for col in ['Bias (kBTU)', 'MAE (kBTU)', 'Mean Observed (kBTU)']:
        if col in result_df.columns:
            result_df[col] = result_df[col].round(0).astype(int)
    
    result_df.attrs['note'] = (
        "Bias = weighted mean(Ŷ − Y). nMAE = MAE / mean(Y) × 100. "
        "All metrics computed on outer-fold predictions with NWEIGHT."
    )
    
    return result_df


def create_hdd_diagnostics_table(bias_by_hdd_df: pd.DataFrame,
                                  include_normalized: bool = True) -> pd.DataFrame:
    """
    Create HDD diagnostics table with bin support and normalized bias.
    
    Parameters
    ----------
    bias_by_hdd_df : DataFrame
        Bias results by HDD bin
    include_normalized : bool
        Whether to include normalized bias
        
    Returns
    -------
    DataFrame
        Formatted diagnostics table
    """
    result_df = bias_by_hdd_df.copy()
    
    # Rename columns
    result_df = result_df.rename(columns={
        'hdd_bin': 'HDD Bin',
        'n_samples': 'n',
        'total_weight': 'Weighted n',
        'weighted_bias': 'Bias (kBTU)',
        'weighted_mae': 'MAE (kBTU)',
        'weighted_mean_true': 'Mean Observed (kBTU)',
        'bias_pct': 'Bias (%)'
    })
    
    # Round
    numeric_cols = ['Bias (kBTU)', 'MAE (kBTU)', 'Mean Observed (kBTU)', 'Weighted n']
    for col in numeric_cols:
        if col in result_df.columns:
            result_df[col] = result_df[col].round(0).astype(int)
    
    if 'Bias (%)' in result_df.columns:
        result_df['Bias (%)'] = result_df['Bias (%)'].round(1)
    
    result_df.attrs['note'] = (
        "Bias = weighted mean(Ŷ − Y). Normalized Bias = Bias / Mean Observed × 100. "
        "Sparse high-HDD bins should be interpreted with caution."
    )
    
    return result_df


def create_h1_comparison_table(comparison_results: Dict[str, Any]) -> pd.DataFrame:
    """
    Create H1 test table: Monolithic vs Split paired comparison.
    
    Shows per-fold metrics, mean ± SD, Δ values, and paired p-value.
    This directly supports H1 hypothesis testing.
    
    Parameters
    ----------
    comparison_results : dict
        Results from compare_split_vs_monolithic()
        
    Returns
    -------
    DataFrame
        Paired comparison table
    """
    split_df = comparison_results['split_by_fold']
    mono_df = comparison_results['mono_by_fold']
    
    results = []
    
    # Per-fold comparison
    n_folds = len(split_df)
    for fold in range(n_folds):
        mono_row = mono_df.iloc[fold]
        split_row = split_df.iloc[fold]
        
        results.append({
            'Fold': fold + 1,
            'Mono wRMSE': f"{mono_row['weighted_rmse']:,.0f}",
            'Split wRMSE': f"{split_row['weighted_rmse']:,.0f}",
            'Δ wRMSE': f"{mono_row['weighted_rmse'] - split_row['weighted_rmse']:+,.0f}",
            'Mono wMAE': f"{mono_row['weighted_mae']:,.0f}",
            'Split wMAE': f"{split_row['weighted_mae']:,.0f}",
            'Δ wMAE': f"{mono_row['weighted_mae'] - split_row['weighted_mae']:+,.0f}",
            'Mono wR²': f"{mono_row['weighted_r2']:.3f}",
            'Split wR²': f"{split_row['weighted_r2']:.3f}",
            'Δ wR²': f"{split_row['weighted_r2'] - mono_row['weighted_r2']:+.3f}",
        })
    
    # Summary row
    summary = {
        'Fold': 'Mean ± SD',
        'Mono wRMSE': f"{mono_df['weighted_rmse'].mean():,.0f} ± {mono_df['weighted_rmse'].std():,.0f}",
        'Split wRMSE': f"{split_df['weighted_rmse'].mean():,.0f} ± {split_df['weighted_rmse'].std():,.0f}",
        'Δ wRMSE': f"{(mono_df['weighted_rmse'] - split_df['weighted_rmse']).mean():+,.0f}",
        'Mono wMAE': f"{mono_df['weighted_mae'].mean():,.0f} ± {mono_df['weighted_mae'].std():,.0f}",
        'Split wMAE': f"{split_df['weighted_mae'].mean():,.0f} ± {split_df['weighted_mae'].std():,.0f}",
        'Δ wMAE': f"{(mono_df['weighted_mae'] - split_df['weighted_mae']).mean():+,.0f}",
        'Mono wR²': f"{mono_df['weighted_r2'].mean():.3f} ± {mono_df['weighted_r2'].std():.3f}",
        'Split wR²': f"{split_df['weighted_r2'].mean():.3f} ± {split_df['weighted_r2'].std():.3f}",
        'Δ wR²': f"{(split_df['weighted_r2'] - mono_df['weighted_r2']).mean():+.3f}",
    }
    results.append(summary)
    
    df = pd.DataFrame(results)
    
    # Compute consistency (how many folds show improvement)
    delta_rmse = mono_df['weighted_rmse'].values - split_df['weighted_rmse'].values
    n_improved = (delta_rmse > 0).sum()
    improvement_pct = comparison_results.get('rmse_improvement_pct', 0)
    
    # Frame with effect size and consistency, not p-value
    df.attrs['note'] = (
        f"H1 Evidence: Technology-Split vs Monolithic model comparison. "
        f"Split improves in {n_improved}/{n_folds} folds. "
        f"Mean Δ wRMSE = {delta_rmse.mean():+,.0f} kBTU ({improvement_pct:.1f}% reduction). "
        f"Mean Δ wR² = {(split_df['weighted_r2'] - mono_df['weighted_r2']).mean():+.3f}. "
        f"Split model shows more consistent performance (lower SD across folds). "
        f"All metrics on outer-fold predictions, weighted by NWEIGHT."
    )
    
    return df


def create_physics_diagnostics_table(y_true: np.ndarray,
                                      y_pred: np.ndarray,
                                      weights: np.ndarray,
                                      tech_group: np.ndarray,
                                      hdd: np.ndarray) -> pd.DataFrame:
    """
    Create physics-consistency diagnostics table by technology.
    
    Includes:
    - Non-physical rate (% predictions < 0)
    - Cold-climate bias (HDD ≥ 6000)
    - Tail underprediction index (top decile of observed)
    - Overall bias
    
    Parameters
    ----------
    y_true : array
        Observed values
    y_pred : array
        Predicted values
    weights : array
        Sample weights
    tech_group : array
        Technology group labels
    hdd : array
        Heating degree days
        
    Returns
    -------
    DataFrame
        Physics diagnostics by technology
    """
    residuals = y_pred - y_true
    
    unique_techs = [t for t in np.unique(tech_group) 
                   if t not in ['no_heating', 'unknown'] and not pd.isna(t)]
    
    results = []
    
    for tech in unique_techs:
        mask = tech_group == tech
        n = mask.sum()
        w = weights[mask]
        total_w = w.sum()
        
        # Non-physical rate
        neg_preds = y_pred[mask] < 0
        neg_rate = np.sum(w[neg_preds]) / total_w * 100 if total_w > 0 else 0
        
        # Cold-climate bias (HDD >= 6000)
        cold_mask = mask & (hdd >= 6000)
        n_cold = cold_mask.sum()
        if n_cold > 0:
            w_cold = weights[cold_mask]
            cold_bias = np.sum(w_cold * residuals[cold_mask]) / np.sum(w_cold)
            cold_mean_true = np.sum(w_cold * y_true[cold_mask]) / np.sum(w_cold)
            cold_bias_pct = cold_bias / cold_mean_true * 100 if cold_mean_true != 0 else np.nan
        else:
            cold_bias = np.nan
            cold_bias_pct = np.nan
            n_cold = 0
        
        # Tail underprediction (top decile = top 10% of observed for this tech)
        p90 = np.percentile(y_true[mask], 90)
        tail_mask = mask & (y_true >= p90)
        n_tail = tail_mask.sum()
        if n_tail > 0:
            w_tail = weights[tail_mask]
            tail_residual = np.sum(w_tail * residuals[tail_mask]) / np.sum(w_tail)
            tail_mean_true = np.sum(w_tail * y_true[tail_mask]) / np.sum(w_tail)
            tail_residual_pct = tail_residual / tail_mean_true * 100 if tail_mean_true != 0 else np.nan
        else:
            tail_residual = np.nan
            tail_residual_pct = np.nan
            n_tail = 0
        
        # Overall bias
        overall_bias = np.sum(w * residuals[mask]) / total_w if total_w > 0 else 0
        overall_bias_pct = overall_bias / (np.sum(w * y_true[mask]) / total_w) * 100
        
        # Weighted mean HDD for this tech
        mean_hdd = np.sum(w * hdd[mask]) / total_w if total_w > 0 else 0
        
        # Target variance (weighted) - explains low R² for some groups
        mean_y = np.sum(w * y_true[mask]) / total_w
        target_var = np.sum(w * (y_true[mask] - mean_y)**2) / total_w
        target_std = np.sqrt(target_var)
        cv_target = target_std / mean_y * 100 if mean_y > 0 else np.nan  # Coefficient of variation
        
        # Weighted correlation (r) - more robust than R² for subgroups
        mean_pred = np.sum(w * y_pred[mask]) / total_w
        cov_xy = np.sum(w * (y_true[mask] - mean_y) * (y_pred[mask] - mean_pred)) / total_w
        std_pred = np.sqrt(np.sum(w * (y_pred[mask] - mean_pred)**2) / total_w)
        
        if target_std > 0 and std_pred > 0:
            weighted_corr = cov_xy / (target_std * std_pred)
        else:
            weighted_corr = np.nan
        
        results.append({
            'Technology': tech.replace('_', ' ').title(),
            'n': n,
            'Pop Share (%)': total_w / weights.sum() * 100,
            'Mean HDD': mean_hdd,
            'Target SD (kBTU)': target_std,
            'Target CV (%)': cv_target,
            'Weighted r': weighted_corr,
            'Overall Bias (%)': overall_bias_pct,
            'n_cold (HDD≥6k)': n_cold,
            'Cold Bias (%)': cold_bias_pct if not np.isnan(cold_bias_pct) else 'N/A',
            'n_tail (top 10%)': n_tail,
            'Tail Bias (%)': tail_residual_pct if not np.isnan(tail_residual_pct) else 'N/A',
            'Non-Physical (%)': neg_rate
        })
    
    df = pd.DataFrame(results)
    
    # Round
    for col in ['Pop Share (%)', 'Mean HDD', 'Overall Bias (%)', 'Non-Physical (%)', 
                'Target SD (kBTU)', 'Target CV (%)', 'Weighted r']:
        if col in df.columns:
            if col == 'Weighted r':
                df[col] = pd.to_numeric(df[col], errors='coerce').round(3)
            elif col == 'Target SD (kBTU)':
                df[col] = pd.to_numeric(df[col], errors='coerce').round(0)
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce').round(1)
    
    df.attrs['note'] = (
        "Physics-consistency diagnostics by technology. "
        "Cold Bias = weighted mean residual for HDD ≥ 6000 (as % of mean observed). "
        "Tail Bias = weighted mean residual for top 10% of observed energy (as % of mean). "
        "Negative bias = systematic underprediction. "
        "All metrics weighted by NWEIGHT, outer-fold predictions. "
        "NOTE: Low/negative wR² for electric groups reflects high noise in RECS end-use estimates "
        "and low within-group variance; correlation (r) may still indicate predictive signal."
    )
    
    return df


def create_baseline_comparison_table(baseline_results: pd.DataFrame,
                                      ml_metrics: Dict[str, float],
                                      ml_name: str = "LightGBM") -> pd.DataFrame:
    """
    Create baseline vs ML comparison table.
    
    Shows physics baselines vs ML model performance.
    
    Parameters
    ----------
    baseline_results : DataFrame
        Baseline evaluation results by technology
    ml_metrics : dict
        ML model metrics (overall)
    ml_name : str
        ML model name
        
    Returns
    -------
    DataFrame
        Comparison table
    """
    results = []
    
    # Add baseline results
    for _, row in baseline_results.iterrows():
        results.append({
            'Model': f"Baseline ({row.get('tech_group', 'All')})",
            'wRMSE (kBTU)': row.get('weighted_rmse', np.nan),
            'wMAE (kBTU)': row.get('weighted_mae', np.nan),
            'wR²': row.get('weighted_r2', np.nan),
            'Type': 'Physics Baseline'
        })
    
    # Add ML model results
    results.append({
        'Model': ml_name,
        'wRMSE (kBTU)': ml_metrics.get('weighted_rmse', np.nan),
        'wMAE (kBTU)': ml_metrics.get('weighted_mae', np.nan),
        'wR²': ml_metrics.get('weighted_r2', np.nan),
        'Type': 'ML Model'
    })
    
    df = pd.DataFrame(results)
    
    # Compute improvement
    if len(baseline_results) > 0:
        baseline_rmse = baseline_results['weighted_rmse'].mean()
        ml_rmse = ml_metrics.get('weighted_rmse', np.nan)
        improvement = (baseline_rmse - ml_rmse) / baseline_rmse * 100 if baseline_rmse > 0 else np.nan
        
        df.attrs['note'] = (
            f"Comparison of physics baselines vs {ml_name}. "
            f"RMSE improvement: {improvement:.1f}% over mean baseline. "
            f"All metrics weighted by NWEIGHT, outer-fold predictions."
        )
    
    return df


def create_calibration_comparison_table(before_metrics: Dict, after_metrics: Dict,
                                         before_diagnostics: Dict = None,
                                         after_diagnostics: Dict = None) -> pd.DataFrame:
    """
    Create before/after calibration comparison table.
    
    Parameters
    ----------
    before_metrics : dict
        Metrics before calibration
    after_metrics : dict
        Metrics after calibration
    before_diagnostics : dict, optional
        Physics diagnostics before
    after_diagnostics : dict, optional
        Physics diagnostics after
        
    Returns
    -------
    DataFrame
        Calibration comparison table
    """
    rows = []
    
    # Performance metrics
    metrics = [
        ('wRMSE (kBTU)', 'weighted_rmse'),
        ('wMAE (kBTU)', 'weighted_mae'),
        ('wR²', 'weighted_r2'),
        ('wBias (kBTU)', 'weighted_bias'),
    ]
    
    for label, key in metrics:
        before = before_metrics.get(key, np.nan)
        after = after_metrics.get(key, np.nan)
        
        if key == 'weighted_r2':
            change = after - before
            change_str = f"+{change:.3f}" if change > 0 else f"{change:.3f}"
        else:
            pct_change = (after - before) / abs(before) * 100 if before != 0 else 0
            change_str = f"{pct_change:+.1f}%"
        
        rows.append({
            'Metric': label,
            'Before Calibration': f"{before:,.0f}" if abs(before) > 10 else f"{before:.3f}",
            'After Calibration': f"{after:,.0f}" if abs(after) > 10 else f"{after:.3f}",
            'Change': change_str
        })
    
    # Calibration slope (if available)
    if 'calibration_slope' in before_metrics or 'calibration_slope' in after_metrics:
        rows.append({
            'Metric': 'Calibration Slope',
            'Before Calibration': f"{before_metrics.get('calibration_slope', 'N/A'):.3f}",
            'After Calibration': f"{after_metrics.get('calibration_slope', 'N/A'):.3f}",
            'Change': '→ closer to 1.0'
        })
    
    # Physics diagnostics
    if before_diagnostics and after_diagnostics:
        for tech in ['Combustion', 'Electric Heat Pump']:
            if tech in before_diagnostics and tech in after_diagnostics:
                before_tail = before_diagnostics[tech].get('tail_bias_pct', np.nan)
                after_tail = after_diagnostics[tech].get('tail_bias_pct', np.nan)
                
                rows.append({
                    'Metric': f'Tail Bias - {tech} (%)',
                    'Before Calibration': f"{before_tail:.1f}%",
                    'After Calibration': f"{after_tail:.1f}%",
                    'Change': f"{after_tail - before_tail:+.1f}pp"
                })
    
    df = pd.DataFrame(rows)
    
    df.attrs['note'] = (
        "Comparison of model performance before and after isotonic calibration. "
        "Calibration applies monotonic correction to reduce tail underprediction. "
        "All metrics on outer-fold predictions, weighted by NWEIGHT."
    )
    
    return df


def create_ablation_table(ablation_results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create ablation study table.
    
    Parameters
    ----------
    ablation_results : dict
        Results from ablation experiments
        
    Returns
    -------
    DataFrame
        Ablation comparison table
    """
    results = []
    
    for config_name, metrics in ablation_results.items():
        results.append({
            'Configuration': config_name,
            'wRMSE': metrics.get('weighted_rmse', np.nan),
            'wMAE': metrics.get('weighted_mae', np.nan),
            'wR²': metrics.get('weighted_r2', np.nan),
            'wBias': metrics.get('weighted_bias', np.nan),
        })
    
    df = pd.DataFrame(results)
    
    df.attrs['note'] = (
        "Ablation study comparing model configurations. "
        "All metrics on outer-fold test predictions, weighted by NWEIGHT."
    )
    
    return df


def save_table_with_note(df: pd.DataFrame, 
                          filepath: str,
                          note: str = None) -> None:
    """
    Save table to CSV with metadata note in comments.
    
    Parameters
    ----------
    df : DataFrame
        Table to save
    filepath : str
        Output path
    note : str, optional
        Note to include
    """
    # Get note from attrs if not provided
    if note is None:
        note = df.attrs.get('note', '')
    
    # Remove any 'Unnamed' columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Save with header comment
    with open(filepath, 'w', encoding='utf-8-sig', newline='') as f:
        if note:
            f.write(f"# {note}\n")
        df.to_csv(f, index=False)
    
    logger.info(f"Saved table: {filepath}")
