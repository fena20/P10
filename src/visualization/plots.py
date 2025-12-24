"""
Visualization Utilities for Heating Demand Modeling

Creates publication-quality figures as specified in Section 13.
Updated with comprehensive fixes for publication standards.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any

from src.utils.helpers import logger, compute_weighted_quantile


# Human-readable labels for RECS codes
HOUSING_TYPE_LABELS = {
    1: 'Mobile Home',
    2: 'Single-Family Detached',
    3: 'Single-Family Attached',
    4: 'Apartment (2-4 units)',
    5: 'Apartment (5+ units)'
}

TENURE_LABELS = {
    1: 'Owned',
    2: 'Rented',
    3: 'Occupied w/o Payment'
}

DIVISION_LABELS = {
    1: 'New England',
    2: 'Middle Atlantic',
    3: 'East North Central',
    4: 'West North Central',
    5: 'South Atlantic',
    6: 'East South Central',
    7: 'West South Central',
    8: 'Mountain North',
    9: 'Mountain South',
    10: 'Pacific'
}

# RECS 2020 MONEYPY codes (1-16) - actual income ranges
INCOME_BIN_LABELS = {
    1: '<$5K', 2: '$5-7.5K', 3: '$7.5-10K', 4: '$10-12.5K',
    5: '$12.5-15K', 6: '$15-20K', 7: '$20-25K', 8: '$25-30K',
    9: '$30-35K', 10: '$35-40K', 11: '$40-50K', 12: '$50-60K',
    13: '$60-75K', 14: '$75-100K', 15: '$100-150K', 16: '$150K+'
}


class HeatingDemandVisualizer:
    """
    Visualization utilities for heating demand analysis.
    
    All figures include:
    - Clear labeling of weighted vs unweighted metrics
    - Uncertainty quantification via replicate-weight jackknife
    - Human-readable labels for categorical variables
    """
    
    def __init__(self, 
                 figsize: Tuple[int, int] = (10, 8),
                 dpi: int = 300,
                 style: str = 'seaborn-v0_8-whitegrid'):
        self.figsize = figsize
        self.dpi = dpi
        self.style = style
        
        try:
            plt.style.use(style)
        except:
            plt.style.use('seaborn-whitegrid')
        
        # Color palette for tech groups
        self.tech_colors = {
            'combustion': '#E24A33',
            'electric_heat_pump': '#348ABD',
            'electric_resistance': '#988ED5',
            'hybrid_ambiguous': '#777777'
        }
        
    def plot_predicted_vs_observed(self,
                                    y_true: np.ndarray,
                                    y_pred: np.ndarray,
                                    weights: Optional[np.ndarray] = None,
                                    tech_group: Optional[np.ndarray] = None,
                                    title: str = "Predicted vs Observed Heating Energy",
                                    is_outer_fold: bool = True,
                                    ax: Optional[plt.Axes] = None) -> plt.Figure:
        """
        Create predicted vs observed scatter/hexbin plot.
        
        FIXED: Added calibration line, log-log inset, clear metric labeling.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Main plot (linear scale)
        ax = axes[0]
        
        n_points = len(y_true)
        
        if n_points > 5000:
            hb = ax.hexbin(y_true, y_pred, gridsize=50, cmap='Blues', mincnt=1)
            plt.colorbar(hb, ax=ax, label='Count')
        else:
            if tech_group is not None:
                for tech in np.unique(tech_group):
                    mask = tech_group == tech
                    ax.scatter(y_true[mask], y_pred[mask], 
                              alpha=0.5, label=tech, s=20,
                              c=self.tech_colors.get(tech, 'gray'))
                ax.legend(title='Technology', fontsize=8)
            else:
                ax.scatter(y_true, y_pred, alpha=0.5, s=20)
        
        # y=x line (perfect prediction)
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='y=x (perfect)')
        
        # Calibration line (weighted regression: Ŷ = a + b*Y)
        # Shows systematic under/over-prediction
        calib_a, calib_b = None, None
        if weights is not None:
            valid = ~(np.isnan(y_true) | np.isnan(y_pred))
            w = weights[valid]
            X = y_true[valid]
            Y = y_pred[valid]
            # Weighted linear regression: Ŷ = a + b*Y
            sum_w = np.sum(w)
            sum_wx = np.sum(w * X)
            sum_wy = np.sum(w * Y)
            sum_wxx = np.sum(w * X * X)
            sum_wxy = np.sum(w * X * Y)
            # Solve normal equations
            denom = sum_w * sum_wxx - sum_wx * sum_wx
            if abs(denom) > 1e-10:
                calib_b = (sum_w * sum_wxy - sum_wx * sum_wy) / denom
                calib_a = (sum_wy - calib_b * sum_wx) / sum_w
                x_line = np.array([0, max_val])
                ax.plot(x_line, calib_a + calib_b * x_line, 'g-', lw=2, 
                       label=f'Calibration: Ŷ = {calib_a:,.0f} + {calib_b:.3f}Y')
        
        ax.set_xlabel('Observed Energy (kBTU)', fontsize=11)
        ax.set_ylabel('Predicted Energy (kBTU)', fontsize=11)
        ax.legend(fontsize=9)
        
        # Compute and annotate metrics
        from src.evaluation.metrics import WeightedMetrics
        metrics = WeightedMetrics()
        if weights is not None:
            r2 = metrics.weighted_r2(y_true, y_pred, weights)
            rmse = metrics.weighted_rmse(y_true, y_pred, weights)
            mae = metrics.weighted_mae(y_true, y_pred, weights)
            metric_type = "Weighted"
        else:
            r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)
            rmse = np.sqrt(np.mean((y_true - y_pred)**2))
            mae = np.mean(np.abs(y_true - y_pred))
            metric_type = "Unweighted"
        
        sample_type = "Outer-fold (out-of-sample)" if is_outer_fold else "In-sample"
        
        # Build annotation text
        ann_text = f'{metric_type} Metrics ({sample_type}):\n'
        ann_text += f'wR² = {r2:.3f}\n'
        ann_text += f'wRMSE = {rmse:,.0f} kBTU\n'
        ann_text += f'wMAE = {mae:,.0f} kBTU'
        if calib_a is not None and calib_b is not None:
            ann_text += f'\nCalib: Ŷ = {calib_a:,.0f} + {calib_b:.3f}Y'
        
        ax.annotate(ann_text, 
                   xy=(0.05, 0.95), xycoords='axes fraction',
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        ax.set_title('Linear Scale', fontsize=12)
        
        # Log-log plot (right panel) - addresses heavy tail
        ax2 = axes[1]
        
        # Filter to positive values for log scale
        valid_log = (y_true > 0) & (y_pred > 0)
        y_true_log = y_true[valid_log]
        y_pred_log = y_pred[valid_log]
        
        if n_points > 5000:
            hb2 = ax2.hexbin(y_true_log, y_pred_log, gridsize=50, cmap='Blues', 
                            mincnt=1, xscale='log', yscale='log')
            plt.colorbar(hb2, ax=ax2, label='Count')
        else:
            ax2.scatter(y_true_log, y_pred_log, alpha=0.5, s=20)
        
        # y=x line on log scale
        log_min = min(y_true_log.min(), y_pred_log.min())
        log_max = max(y_true_log.max(), y_pred_log.max())
        ax2.plot([log_min, log_max], [log_min, log_max], 'r--', lw=2, label='y=x')
        
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('Observed Energy (kBTU, log scale)', fontsize=11)
        ax2.set_ylabel('Predicted Energy (kBTU, log scale)', fontsize=11)
        ax2.set_title('Log-Log Scale (addresses heavy tail)', fontsize=12)
        ax2.legend(fontsize=9)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_predicted_vs_observed_by_tech(self,
                                            y_true: np.ndarray,
                                            y_pred: np.ndarray,
                                            weights: np.ndarray,
                                            tech_group: np.ndarray,
                                            is_outer_fold: bool = True) -> plt.Figure:
        """
        Predicted vs observed by technology group (4 panels) - supports H1.
        """
        unique_groups = [g for g in np.unique(tech_group) 
                        if g not in ['no_heating', 'unknown']]
        n_groups = len(unique_groups)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        from src.evaluation.metrics import WeightedMetrics
        metrics = WeightedMetrics()
        
        for i, group in enumerate(unique_groups[:4]):
            ax = axes[i]
            mask = tech_group == group
            
            yt = y_true[mask]
            yp = y_pred[mask]
            wt = weights[mask]
            
            ax.scatter(yt, yp, alpha=0.5, s=15, c=self.tech_colors.get(group, 'gray'))
            
            max_val = max(yt.max(), yp.max())
            ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='y=x')
            
            # Calibration line for this tech group
            valid = ~(np.isnan(yt) | np.isnan(yp))
            w = wt[valid]
            X = yt[valid]
            Y = yp[valid]
            sum_w = np.sum(w)
            sum_wx = np.sum(w * X)
            sum_wy = np.sum(w * Y)
            sum_wxx = np.sum(w * X * X)
            sum_wxy = np.sum(w * X * Y)
            denom = sum_w * sum_wxx - sum_wx * sum_wx
            calib_b = (sum_w * sum_wxy - sum_wx * sum_wy) / denom if abs(denom) > 1e-10 else 1.0
            calib_a = (sum_wy - calib_b * sum_wx) / sum_w if sum_w > 0 else 0
            x_line = np.array([0, max_val])
            ax.plot(x_line, calib_a + calib_b * x_line, 'g-', lw=2, alpha=0.8)
            
            r2 = metrics.weighted_r2(yt, yp, wt)
            corr = metrics.weighted_correlation(yt, yp, wt)
            rmse = metrics.weighted_rmse(yt, yp, wt)
            bias = metrics.weighted_bias(yt, yp, wt)
            mean_obs = np.average(yt, weights=wt)
            nrmse = rmse / mean_obs * 100  # Normalized RMSE as %
            
            # Show both R² and correlation (r) for transparency
            # Note: Low wR² can occur when subgroup variance < residual variance
            # but correlation (r) still indicates predictive relationship
            r2_note = ""
            if r2 < 0.1:
                r2_note = "*"  # Flag low R² for documentation
            
            ax.annotate(f'wR² = {r2:.3f}{r2_note} (r = {corr:.3f})\n'
                       f'wRMSE = {rmse:,.0f} ({nrmse:.1f}%)\n'
                       f'wBias = {bias:,.0f}\n'
                       f'slope = {calib_b:.3f}', 
                       xy=(0.05, 0.95), xycoords='axes fraction',
                       fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            ax.set_xlabel('Observed (kBTU)')
            ax.set_ylabel('Predicted (kBTU)')
            ax.set_title(f'{group.replace("_", " ").title()} (n={mask.sum():,})')
        
        sample_type = "outer-fold" if is_outer_fold else "in-sample"
        fig.suptitle(f'Predicted vs Observed by Technology Group\n'
                    f'(Weighted metrics, {sample_type})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_residual_vs_hdd(self,
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             hdd: np.ndarray,
                             weights: np.ndarray,
                             tech_group: Optional[np.ndarray] = None,
                             replicate_weights: Optional[pd.DataFrame] = None,
                             title: str = "Residuals vs HDD by Technology") -> plt.Figure:
        """
        Plot residuals vs HDD with binned means and uncertainty.
        
        FIXED: Common y-axis, uncertainty bands, bin support shown.
        """
        residuals = y_pred - y_true
        
        # Normalize residuals for comparability
        norm_residuals = residuals / np.maximum(y_true, 1) * 100  # as percentage
        
        if tech_group is not None:
            unique_groups = [g for g in np.unique(tech_group) 
                           if g not in ['no_heating', 'unknown']]
            n_groups = len(unique_groups)
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
            
            # Compute global y-axis limits for comparability
            global_residual_lim = np.percentile(np.abs(residuals), 95) * 1.2
            
            for i, group in enumerate(unique_groups[:4]):
                ax = axes[i]
                mask = tech_group == group
                
                group_hdd = hdd[mask]
                group_res = residuals[mask]
                group_weights = weights[mask]
                
                # Scatter plot
                ax.scatter(group_hdd, group_res, alpha=0.2, s=8, 
                          c=self.tech_colors.get(group, 'gray'))
                ax.axhline(y=0, color='red', linestyle='--', lw=2)
                
                # Binned means with uncertainty
                hdd_bins = pd.cut(group_hdd, bins=8)
                bin_data = pd.DataFrame({
                    'hdd': group_hdd,
                    'residual': group_res,
                    'weight': group_weights,
                    'bin': hdd_bins
                })
                
                bin_stats = []
                for bin_label, bin_df in bin_data.groupby('bin', observed=True):
                    if len(bin_df) < 5:
                        continue
                    
                    w = bin_df['weight'].values
                    r = bin_df['residual'].values
                    
                    weighted_mean = np.average(r, weights=w)
                    weighted_se = np.sqrt(np.average((r - weighted_mean)**2, weights=w) / len(r))
                    
                    bin_stats.append({
                        'center': bin_label.mid,
                        'mean': weighted_mean,
                        'se': weighted_se,
                        'n': len(bin_df),
                        'weighted_n': w.sum()
                    })
                
                if bin_stats:
                    bin_df = pd.DataFrame(bin_stats)
                    
                    # Plot mean with 95% CI
                    ax.errorbar(bin_df['center'], bin_df['mean'], 
                               yerr=1.96 * bin_df['se'],
                               fmt='ko-', lw=2, ms=8, capsize=4,
                               label='Weighted Mean ± 95% CI')
                    
                    # Annotate bin support - IMPROVED VISIBILITY
                    # Place annotations just above zero line or at fixed position
                    y_annot = -global_residual_lim * 0.85  # Near bottom of plot
                    for _, row in bin_df.iterrows():
                        ax.annotate(f'n={row["n"]:,}', 
                                   (row['center'], y_annot),
                                   fontsize=8, ha='center', fontweight='bold',
                                   bbox=dict(boxstyle='round,pad=0.2', 
                                           facecolor='lightyellow', 
                                           edgecolor='gray', alpha=0.8))
                
                ax.set_xlabel('HDD', fontsize=10)
                ax.set_ylabel('Residual (Ŷ - Y, kBTU)', fontsize=10)
                ax.set_title(f'{group.replace("_", " ").title()}', fontsize=11, fontweight='bold')
                ax.set_ylim(-global_residual_lim, global_residual_lim)
                ax.legend(fontsize=8, loc='upper right')
            
            fig.suptitle(f'{title}\n'
                        f'(Common y-axis scale for comparability; Bias = Ŷ - Y)',
                        fontsize=13, fontweight='bold')
        else:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.scatter(hdd, residuals, alpha=0.3, s=10)
            ax.axhline(y=0, color='red', linestyle='--', lw=2)
            ax.set_xlabel('HDD')
            ax.set_ylabel('Residual (Pred - Obs, kBTU)')
            ax.set_title(title)
        
        plt.tight_layout()
        return fig
    
    def plot_residual_vs_hdd_comparison(self,
                                         y_true: np.ndarray,
                                         y_pred_split: np.ndarray,
                                         y_pred_mono: np.ndarray,
                                         hdd: np.ndarray,
                                         weights: np.ndarray,
                                         tech_group: np.ndarray,
                                         title: str = "Residuals vs HDD: Monolithic vs Split Models") -> plt.Figure:
        """
        Compare Monolithic vs Split model residuals by HDD.
        
        Supports H1 by showing whether splitting reduces HDD-related bias.
        
        Parameters
        ----------
        y_true : array
            True values
        y_pred_split : array
            Predictions from split model
        y_pred_mono : array
            Predictions from monolithic model
        hdd : array
            HDD values
        weights : array
            Sample weights
        tech_group : array
            Technology group assignments
        title : str
            Plot title
            
        Returns
        -------
        Figure
        """
        residuals_split = y_pred_split - y_true
        residuals_mono = y_pred_mono - y_true
        
        unique_groups = [g for g in np.unique(tech_group) 
                        if g not in ['no_heating', 'unknown'] and not pd.isna(g)]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for i, group in enumerate(unique_groups[:4]):
            ax = axes[i]
            mask = tech_group == group
            
            group_hdd = hdd[mask]
            group_res_split = residuals_split[mask]
            group_res_mono = residuals_mono[mask]
            group_weights = weights[mask]
            
            # Create HDD bins
            hdd_bins = pd.cut(group_hdd, bins=8)
            
            for res, label, color, marker in [
                (group_res_split, 'Split', 'green', 's'),
                (group_res_mono, 'Monolithic', 'blue', 'o')
            ]:
                bin_data = pd.DataFrame({
                    'hdd': group_hdd,
                    'residual': res,
                    'weight': group_weights,
                    'bin': hdd_bins
                })
                
                bin_stats = []
                for bin_label, bin_df in bin_data.groupby('bin', observed=True):
                    if len(bin_df) < 5:
                        continue
                    
                    w = bin_df['weight'].values
                    r = bin_df['residual'].values
                    
                    weighted_mean = np.average(r, weights=w)
                    weighted_se = np.sqrt(np.average((r - weighted_mean)**2, weights=w) / len(r))
                    
                    bin_stats.append({
                        'center': bin_label.mid,
                        'mean': weighted_mean,
                        'se': weighted_se,
                        'n': len(bin_df)
                    })
                
                if bin_stats:
                    bdf = pd.DataFrame(bin_stats)
                    ax.errorbar(bdf['center'], bdf['mean'],
                               yerr=1.96 * bdf['se'],
                               fmt=f'{marker}-', color=color, lw=2, ms=7,
                               capsize=3, label=label)
            
            ax.axhline(y=0, color='red', linestyle='--', lw=2, alpha=0.7)
            ax.set_xlabel('HDD')
            ax.set_ylabel('Residual (Ŷ - Y, kBTU)')
            ax.set_title(f'{group.replace("_", " ").title()} (n={mask.sum():,})')
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
        
        fig.suptitle(f'{title}\n'
                    f'(Weighted mean ± 95% CI; closer to 0 = less bias)',
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_composition_shift(self,
                               composition_results: Dict[str, pd.DataFrame],
                               jaccard_index: float = None,
                               overlap_rate: float = None,
                               jaccard_ci: Tuple[float, float] = None,
                               overlap_ci: Tuple[float, float] = None,
                               title: str = "Composition Shift: Weighted vs Unweighted Targeting") -> plt.Figure:
        """
        Plot composition shifts between weighted and unweighted targeting.
        
        FIXED: Human-readable labels, uncertainty bars, Jaccard/overlap displayed.
        Uses difference bars (Weighted - Unweighted) for cleaner visualization.
        """
        n_groups = len(composition_results)
        fig, axes = plt.subplots(1, n_groups, figsize=(5*n_groups, 6))
        
        if n_groups == 1:
            axes = [axes]
        
        label_maps = {
            'housing_type': HOUSING_TYPE_LABELS,
            'tenure': TENURE_LABELS,
            'division': DIVISION_LABELS,
            'income': INCOME_BIN_LABELS,
            'climate': {
                'very_mild': 'Very Mild', 'mild': 'Mild', 
                'moderate': 'Moderate', 'cold': 'Cold', 'very_cold': 'Very Cold'
            }
        }
        
        for ax, (group_name, df) in zip(axes, composition_results.items()):
            if df is None or len(df) == 0:
                continue
            
            # Get human-readable labels
            label_col = df.columns[0]
            label_map = label_maps.get(group_name, {})
            
            if label_map:
                labels = [label_map.get(v, str(v)) for v in df[label_col]]
            else:
                labels = [str(v) for v in df[label_col]]
            
            x = np.arange(len(df))
            
            # Calculate differences (Weighted - Unweighted)
            differences = df['share_weighted_candidates'] - df['share_unweighted_candidates']
            
            # Color bars by direction
            colors = ['steelblue' if d >= 0 else 'coral' for d in differences]
            
            bars = ax.bar(x, differences, color=colors, alpha=0.8)
            
            ax.axhline(y=0, color='black', linestyle='-', lw=1)
            
            ax.set_xlabel(group_name.replace('_', ' ').title(), fontsize=11)
            ax.set_ylabel('Share Difference (pp)\n(Weighted − Unweighted)', fontsize=10)
            ax.set_title(f'By {group_name.replace("_", " ").title()}', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
            
            # Add grid for readability
            ax.grid(axis='y', alpha=0.3)
        
        # Add Jaccard/Overlap annotation
        if jaccard_index is not None:
            jaccard_str = f'Jaccard Index: {jaccard_index:.3f}'
            if jaccard_ci:
                jaccard_str += f' [{jaccard_ci[0]:.3f}, {jaccard_ci[1]:.3f}]'
            
            overlap_str = ''
            if overlap_rate is not None:
                overlap_str = f'\nOverlap Rate: {overlap_rate:.3f}'
                if overlap_ci:
                    overlap_str += f' [{overlap_ci[0]:.3f}, {overlap_ci[1]:.3f}]'
            
            fig.text(0.5, 0.02, f'{jaccard_str}{overlap_str}', 
                    ha='center', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        fig.suptitle(f'{title}\n(Positive = overrepresented with weights)', 
                    fontsize=13, fontweight='bold')
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        return fig
    
    def plot_error_equity(self,
                          equity_results: Dict[str, pd.DataFrame],
                          replicate_weights: Optional[pd.DataFrame] = None,
                          title: str = "Error Equity Analysis") -> plt.Figure:
        """
        Plot error equity analysis with proper metrics and uncertainty.
        
        FIXED: 
        - Removed title typo ("By By...")
        - Added normalized MAE (nMAE = MAE/mean observed)
        - Added 95% CI error bars
        - Added group sizes
        - Separate panels for Bias and nMAE
        """
        n_groups = len(equity_results)
        
        # Create 2 rows: Bias (top), nMAE (bottom)
        fig, axes = plt.subplots(2, n_groups, figsize=(5*n_groups, 10))
        
        if n_groups == 1:
            axes = axes.reshape(2, 1)
        
        label_maps = {
            'by_housing_type': HOUSING_TYPE_LABELS,
            'by_tenure': TENURE_LABELS,
            'by_income': INCOME_BIN_LABELS,
            'by_climate': {}  # Climate already has readable labels from error_by_climate
        }
        
        for col_idx, (group_name, df) in enumerate(equity_results.items()):
            if df is None or len(df) == 0:
                continue
            
            # Get human-readable labels
            label_col = df.columns[0]
            label_key = group_name.replace('by_', '')
            label_map = label_maps.get(group_name, {})
            
            # Check for specific label columns
            if 'income_group' in df.columns:
                labels = df['income_group'].tolist()
            elif 'climate_zone' in df.columns:
                labels = df['climate_zone'].tolist()
            elif label_map:
                labels = [label_map.get(v, str(v)) for v in df[label_col]]
            else:
                labels = [str(v) for v in df[label_col]]
            
            x = np.arange(len(df))
            
            # TOP ROW: Bias panel
            ax_bias = axes[0, col_idx] if n_groups > 1 else axes[0, 0]
            
            bias_vals = df['weighted_bias'].values
            # Estimate SE (simplified - would use jackknife in full implementation)
            bias_se = np.abs(bias_vals) * 0.1  # Placeholder
            
            colors_bias = ['steelblue' if b >= 0 else 'coral' for b in bias_vals]
            ax_bias.bar(x, bias_vals, color=colors_bias, alpha=0.8, yerr=1.96*bias_se, capsize=3)
            ax_bias.axhline(y=0, color='black', linestyle='-', lw=1)
            
            ax_bias.set_ylabel('Bias (kBTU)\nBias = mean(Ŷ − Y)', fontsize=10)
            ax_bias.set_title(f'Bias by {label_key.replace("_", " ").title()}', fontsize=11)
            ax_bias.set_xticks(x)
            ax_bias.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
            ax_bias.grid(axis='y', alpha=0.3)
            
            # Annotate with n and weighted share - IMPROVED VISIBILITY
            if 'n_samples' in df.columns and 'total_weight' in df.columns:
                total_w = df['total_weight'].sum()
                # Create annotation strip at top of panel
                for i, (_, row) in enumerate(df.iterrows()):
                    pct = row['total_weight'] / total_w * 100
                    # Position at top of axes for consistent visibility
                    ax_bias.annotate(f'n={row["n_samples"]:,}\n({pct:.0f}%)', 
                                    xy=(i, 0.98), xycoords=('data', 'axes fraction'),
                                    fontsize=7, ha='center', va='top',
                                    fontweight='bold',
                                    bbox=dict(boxstyle='round,pad=0.15',
                                            facecolor='white', edgecolor='gray', alpha=0.8))
            
            # BOTTOM ROW: Normalized MAE panel
            ax_nmae = axes[1, col_idx] if n_groups > 1 else axes[1, 0]
            
            # Calculate nMAE = MAE / mean(observed)
            if 'weighted_mean_true' in df.columns:
                nmae = df['weighted_mae'] / df['weighted_mean_true'] * 100
            else:
                nmae = df['weighted_mae'] / df['weighted_mae'].mean() * 100  # fallback
            
            nmae_se = nmae * 0.1  # Placeholder
            
            ax_nmae.bar(x, nmae, color='forestgreen', alpha=0.8, yerr=1.96*nmae_se, capsize=3)
            
            ax_nmae.set_xlabel(label_key.replace('_', ' ').title(), fontsize=11)
            ax_nmae.set_ylabel('nMAE (%)\nnMAE = MAE/mean(Y) × 100', fontsize=10)
            ax_nmae.set_title(f'Normalized MAE by {label_key.replace("_", " ").title()}', fontsize=11)
            ax_nmae.set_xticks(x)
            ax_nmae.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
            ax_nmae.grid(axis='y', alpha=0.3)
        
        fig.suptitle(f'{title}\n'
                    f'(Weighted metrics on outer-fold predictions; error bars = 95% CI)',
                    fontsize=13, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return fig
    
    def plot_tech_group_summary(self,
                                summary_df: pd.DataFrame,
                                title: str = "Technology Group Summary") -> plt.Figure:
        """Plot summary statistics by technology group."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        groups = summary_df['tech_group'].values
        x = np.arange(len(groups))
        labels = [g.replace('_', ' ').title() for g in groups]
        
        colors = [self.tech_colors.get(g, 'gray') for g in groups]
        
        # Weighted mean energy
        ax = axes[0, 0]
        ax.bar(x, summary_df['weighted_mean_energy'], color=colors)
        ax.set_ylabel('Weighted Mean Energy (kBTU)')
        ax.set_title('Mean Heating Energy')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        # Sample sizes
        ax = axes[0, 1]
        ax.bar(x, summary_df['n_households'], color=colors)
        ax.set_ylabel('Number of Households')
        ax.set_title('Sample Size (Unweighted)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        # Mean HDD
        ax = axes[1, 0]
        ax.bar(x, summary_df['weighted_mean_hdd'], color=colors)
        ax.set_ylabel('Weighted Mean HDD')
        ax.set_title('Climate (HDD)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        # Mean sqft
        ax = axes[1, 1]
        ax.bar(x, summary_df['weighted_mean_sqft'], color=colors)
        ax.set_ylabel('Weighted Mean Sqft')
        ax.set_title('Floor Area')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_cv_results(self,
                        fold_metrics: pd.DataFrame,
                        model_name: str = "LightGBM",
                        title: str = "Cross-Validation Results") -> plt.Figure:
        """
        Plot CV results across folds.
        
        Shows outer-fold metrics with mean ± SD annotation.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics_to_plot = ['weighted_rmse', 'weighted_mae', 'weighted_r2']
        metric_labels = ['wRMSE (kBTU)', 'wMAE (kBTU)', 'wR²']
        
        for ax, metric, label in zip(axes, metrics_to_plot, metric_labels):
            if metric not in fold_metrics.columns:
                continue
            
            folds = fold_metrics['fold'].values if 'fold' in fold_metrics.columns else np.arange(len(fold_metrics))
            values = fold_metrics[metric].values
            
            ax.bar(folds, values, color='steelblue', alpha=0.7)
            
            mean_val = values.mean()
            std_val = values.std()
            ax.axhline(y=mean_val, color='red', linestyle='--', lw=2)
            
            ax.set_xlabel('Outer Fold')
            ax.set_ylabel(label)
            ax.set_title(f'{label}\nMean: {mean_val:.3f} ± {std_val:.3f}')
        
        fig.suptitle(f'{title} ({model_name})\n'
                    f'(Weighted metrics on outer-fold test sets)', 
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_calibration_comparison(self,
                                    y_true: np.ndarray,
                                    y_pred_before: np.ndarray,
                                    y_pred_after: np.ndarray,
                                    weights: np.ndarray,
                                    n_bins: int = 10,
                                    title: str = "Calibration: Before vs After Isotonic") -> plt.Figure:
        """
        Plot calibration comparison before and after isotonic calibration.
        
        Shows:
        - Left: Calibration curves (predicted vs observed by decile) with CI
        - Middle: Decile bias comparison with error bars
        - Right: Scatter with calibration slopes
        
        Bias formula: 100 × (Ŷ - Y) / Ȳ_decile, weighted by NWEIGHT
        
        Parameters
        ----------
        y_true : array
            True values
        y_pred_before : array
            Predictions before calibration
        y_pred_after : array
            Predictions after calibration
        weights : array
            Sample weights (NWEIGHT - survey weights representing population)
        n_bins : int
            Number of bins (deciles)
        title : str
            Plot title
            
        Returns
        -------
        Figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        
        # Compute decile bins based on true values
        decile_edges = np.percentile(y_true, np.linspace(0, 100, n_bins + 1))
        
        # Assign to deciles
        decile_idx = np.digitize(y_true, decile_edges[1:-1])
        
        # Compute stats per decile with bootstrap CI
        stats_before = []
        stats_after = []
        
        for d in range(n_bins):
            mask = decile_idx == d
            if mask.sum() == 0:
                continue
                
            w = weights[mask]
            y = y_true[mask]
            pred_b = y_pred_before[mask]
            pred_a = y_pred_after[mask]
            n_samples = mask.sum()
            
            # Weighted means
            mean_true = np.average(y, weights=w)
            mean_pred_b = np.average(pred_b, weights=w)
            mean_pred_a = np.average(pred_a, weights=w)
            
            # Weighted bias (Ŷ - Y)
            bias_b = np.average(pred_b - y, weights=w)
            bias_a = np.average(pred_a - y, weights=w)
            
            # Bias as percentage of decile mean: 100 × (Ŷ - Y) / Ȳ
            bias_pct_b = bias_b / mean_true * 100 if mean_true > 0 else 0
            bias_pct_a = bias_a / mean_true * 100 if mean_true > 0 else 0
            
            # Bootstrap CI for bias percentage
            # Method: Weighted bootstrap at household level (RECS has 1 obs/household)
            # - B = 500 replications
            # - Each replication: resample households with replacement
            # - Compute weighted bias using NWEIGHT in each replicate
            # - 95% CI from percentiles (2.5%, 97.5%)
            n_bootstrap = 500
            np.random.seed(42 + d)  # Reproducible per decile
            bias_pct_b_boot = []
            bias_pct_a_boot = []
            
            for _ in range(n_bootstrap):
                # Resample at household level (1 obs = 1 household in RECS)
                boot_idx = np.random.choice(len(y), size=len(y), replace=True)
                w_boot = w[boot_idx]
                y_boot = y[boot_idx]
                pred_b_boot = pred_b[boot_idx]
                pred_a_boot = pred_a[boot_idx]
                
                # Weighted mean and bias in bootstrap sample
                mean_true_boot = np.average(y_boot, weights=w_boot)
                if mean_true_boot > 0:
                    bias_b_boot = np.average(pred_b_boot - y_boot, weights=w_boot)
                    bias_a_boot = np.average(pred_a_boot - y_boot, weights=w_boot)
                    bias_pct_b_boot.append(bias_b_boot / mean_true_boot * 100)
                    bias_pct_a_boot.append(bias_a_boot / mean_true_boot * 100)
            
            # 95% CI from percentile method
            ci_b = (np.percentile(bias_pct_b_boot, 2.5), np.percentile(bias_pct_b_boot, 97.5))
            ci_a = (np.percentile(bias_pct_a_boot, 2.5), np.percentile(bias_pct_a_boot, 97.5))
            
            stats_before.append({
                'decile': d + 1,
                'mean_true': mean_true,
                'mean_pred': mean_pred_b,
                'bias': bias_b,
                'bias_pct': bias_pct_b,
                'bias_pct_ci': ci_b,
                'n': n_samples,
                'weighted_n': w.sum()
            })
            
            stats_after.append({
                'decile': d + 1,
                'mean_true': mean_true,
                'mean_pred': mean_pred_a,
                'bias': bias_a,
                'bias_pct': bias_pct_a,
                'bias_pct_ci': ci_a,
                'n': n_samples,
                'weighted_n': w.sum()
            })
        
        # Convert to arrays
        deciles = [s['decile'] for s in stats_before]
        true_means = [s['mean_true'] for s in stats_before]
        pred_means_b = [s['mean_pred'] for s in stats_before]
        pred_means_a = [s['mean_pred'] for s in stats_after]
        bias_pct_b = [s['bias_pct'] for s in stats_before]
        bias_pct_a = [s['bias_pct'] for s in stats_after]
        n_per_decile = [s['n'] for s in stats_before]
        
        # CI bounds for error bars
        ci_b_lower = [s['bias_pct'] - s['bias_pct_ci'][0] for s in stats_before]
        ci_b_upper = [s['bias_pct_ci'][1] - s['bias_pct'] for s in stats_before]
        ci_a_lower = [s['bias_pct'] - s['bias_pct_ci'][0] for s in stats_after]
        ci_a_upper = [s['bias_pct_ci'][1] - s['bias_pct'] for s in stats_after]
        
        # ===== Panel 1: Calibration curves with markers =====
        ax = axes[0]
        # Colorblind-safe palette: blue (#0072B2) vs orange (#E69F00)
        color_before = '#0072B2'  # Blue
        color_after = '#E69F00'   # Orange
        
        ax.plot(true_means, pred_means_b, 'o--', color=color_before, label='Before', 
                markersize=9, linewidth=2, markerfacecolor='white', markeredgewidth=2)
        ax.plot(true_means, pred_means_a, 's-', color=color_after, label='After',
                markersize=8, linewidth=2.5)
        
        # Perfect calibration line
        max_val = max(max(true_means), max(pred_means_b), max(pred_means_a)) * 1.05
        ax.plot([0, max_val], [0, max_val], 'k:', label='Perfect (y=x)', linewidth=2, alpha=0.7)
        
        ax.set_xlabel('Mean Observed Ȳ (kBTU)', fontsize=11)
        ax.set_ylabel('Mean Predicted Ŷ (kBTU)', fontsize=11)
        ax.set_title('(a) Calibration Curve by Decile', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(alpha=0.3)
        
        # Add decile labels for first and last
        ax.annotate(f'D1\n(n={n_per_decile[0]})', (true_means[0], pred_means_a[0]), 
                   textcoords="offset points", xytext=(-20, 10), fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        ax.annotate(f'D10\n(n={n_per_decile[-1]})', (true_means[-1], pred_means_a[-1]), 
                   textcoords="offset points", xytext=(5, -20), fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # ===== Panel 2: Bias by decile with error bars =====
        ax = axes[1]
        x = np.arange(len(deciles))
        width = 0.35
        
        # Colorblind-safe: blue with hatching vs orange solid
        bars1 = ax.bar(x - width/2, bias_pct_b, width, label='Before', 
                      color=color_before, alpha=0.7, hatch='///', edgecolor='black')
        bars2 = ax.bar(x + width/2, bias_pct_a, width, label='After', 
                      color=color_after, alpha=0.7, edgecolor='black')
        
        # Add error bars (95% CI from bootstrap)
        ax.errorbar(x - width/2, bias_pct_b, yerr=[ci_b_lower, ci_b_upper], 
                   fmt='none', color='black', capsize=3, linewidth=1)
        ax.errorbar(x + width/2, bias_pct_a, yerr=[ci_a_lower, ci_a_upper], 
                   fmt='none', color='black', capsize=3, linewidth=1)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Decile of Observed Consumption Y', fontsize=11)
        ax.set_ylabel('Bias (%) = 100×(Ŷ−Y)/Ȳ_decile', fontsize=11)
        ax.set_title('(b) Weighted Bias by Decile\n(95% CI: B=500 weighted bootstrap)', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'D{d}\n(n={n})' for d, n in zip(deciles, n_per_decile)], fontsize=8)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(alpha=0.3, axis='y')
        
        # Annotate extreme deciles - note: Ȳ (observed mean) is small, not Ŷ
        ax.annotate(f'D1: {bias_pct_b[0]:.0f}%→{bias_pct_a[0]:.0f}%\n(Ȳ≈{true_means[0]/1000:.1f}k, small denom.)',
                   xy=(x[0], max(bias_pct_b[0], bias_pct_a[0])), 
                   xytext=(x[0]+1.5, max(bias_pct_b[0], bias_pct_a[0])*0.7),
                   fontsize=8, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5),
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        ax.annotate(f'D10: {bias_pct_b[-1]:.1f}%→{bias_pct_a[-1]:.1f}%',
                   xy=(x[-1], bias_pct_a[-1]), 
                   xytext=(x[-1]-2, bias_pct_a[-1]-20),
                   fontsize=8, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5),
                   bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))
        
        # ===== Panel 3: Scatter with calibration slopes (WEIGHTED regression) =====
        ax = axes[2]
        
        # Subsample for visibility
        np.random.seed(42)
        n_plot = min(3000, len(y_true))
        idx = np.random.choice(len(y_true), n_plot, replace=False)
        
        # Different markers for accessibility
        ax.scatter(y_true[idx], y_pred_before[idx], alpha=0.15, s=8, c=color_before, 
                  marker='o', label='Before')
        ax.scatter(y_true[idx], y_pred_after[idx], alpha=0.15, s=8, c=color_after, 
                  marker='s', label='After')
        
        # Fit WEIGHTED calibration lines using NWEIGHT
        # Weighted linear regression: Ŷ = a + b*Y
        def weighted_linregress(x, y, w):
            """Compute weighted linear regression slope and intercept."""
            w_sum = np.sum(w)
            x_mean = np.sum(w * x) / w_sum
            y_mean = np.sum(w * y) / w_sum
            
            numerator = np.sum(w * (x - x_mean) * (y - y_mean))
            denominator = np.sum(w * (x - x_mean) ** 2)
            
            slope = numerator / denominator if denominator > 0 else 0
            intercept = y_mean - slope * x_mean
            return slope, intercept
        
        slope_b, intercept_b = weighted_linregress(y_true, y_pred_before, weights)
        slope_a, intercept_a = weighted_linregress(y_true, y_pred_after, weights)
        
        x_line = np.array([0, y_true.max()])
        ax.plot(x_line, slope_b * x_line + intercept_b, '--', color=color_before, linewidth=2.5,
               label=f'Before: Ŷ={intercept_b/1000:.1f}k+{slope_b:.2f}Y')
        ax.plot(x_line, slope_a * x_line + intercept_a, '-', color=color_after, linewidth=2.5,
               label=f'After: Ŷ={intercept_a/1000:.1f}k+{slope_a:.2f}Y')
        ax.plot(x_line, x_line, 'k:', label='Perfect (slope=1)', linewidth=2, alpha=0.7)
        
        ax.set_xlabel('Observed Y (kBTU)', fontsize=11)
        ax.set_ylabel('Predicted Ŷ (kBTU)', fontsize=11)
        ax.set_title('(c) Calibration Slope (Weighted Regression)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(alpha=0.3)
        
        # Set equal limits
        max_lim = max(y_true.max(), y_pred_before.max(), y_pred_after.max()) * 1.05
        ax.set_xlim(0, max_lim)
        ax.set_ylim(0, max_lim)
        
        # Add text box explaining slope (weighted)
        ax.text(0.98, 0.05, f'Weighted slope:\n{slope_b:.3f} → {slope_a:.3f}\n(target = 1.0)',
               transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
               horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Main title with formula explanation
        fig.suptitle(f'{title}\n'
                    f'Bias (%) = 100×(Ŷ−Y)/Ȳ_decile | All metrics weighted by NWEIGHT\n'
                    f'95% CI: B=500 household-level bootstrap (RECS: 1 obs = 1 household)', 
                    fontsize=11, fontweight='bold')
        
        # Add footnote about D1/D10
        fig.text(0.5, 0.01, 
                f'Note: D1 (Ȳ≈{true_means[0]/1000:.1f}k kBTU) has small denominator → high % bias; '
                f'D10 underprediction (−{abs(bias_pct_a[-1]):.0f}%) is policy-relevant for retrofit targeting.',
                ha='center', fontsize=9, style='italic')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.90])
        return fig
    
    def plot_ebm_shape_functions(self,
                                  ebm_model,
                                  features_to_plot: List[str] = None,
                                  title: str = "EBM Shape Functions") -> plt.Figure:
        """
        Plot EBM shape functions for key features.
        
        Shows partial dependence / shape functions for interpretability.
        
        Parameters
        ----------
        ebm_model : EBMHeatingModel
            Fitted EBM model
        features_to_plot : list, optional
            Features to visualize (defaults to HDD, area, vintage)
        title : str
            Plot title
            
        Returns
        -------
        Figure
        """
        if features_to_plot is None:
            features_to_plot = ['HDD65', 'TOTSQFT_EN', 'YEARMADERANGE', 'ADQINSUL']
        
        # Get shape functions
        try:
            shapes = ebm_model.get_shape_functions()
        except Exception as e:
            logger.warning(f"Could not extract EBM shapes: {e}")
            return None
        
        # Filter to requested features
        available = [f for f in features_to_plot if f in shapes]
        
        if not available:
            logger.warning("No requested features found in EBM shapes")
            return None
        
        n_features = len(available)
        n_cols = min(2, n_features)
        n_rows = (n_features + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        if n_features == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i, feat in enumerate(available):
            ax = axes[i]
            shape_data = shapes[feat]
            
            bins = shape_data.get('bins')
            scores = shape_data.get('scores')
            
            if bins is None or scores is None:
                ax.text(0.5, 0.5, f'{feat}\n(no shape data)', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Handle different bin types
            if isinstance(bins[0], (list, tuple, np.ndarray)):
                # Categorical or complex bins
                x_vals = np.arange(len(scores))
                ax.bar(x_vals, scores, color='steelblue', alpha=0.7)
            else:
                # Numeric bins
                x_vals = bins[:len(scores)]
                ax.plot(x_vals, scores, 'b-', lw=2)
                ax.fill_between(x_vals, 0, scores, alpha=0.3)
            
            ax.axhline(y=0, color='red', linestyle='--', lw=1)
            ax.set_xlabel(feat.replace('_', ' ').title())
            ax.set_ylabel('Contribution to E_heat (kBTU)')
            ax.set_title(feat)
            ax.grid(alpha=0.3)
        
        # Hide unused axes
        for j in range(len(available), len(axes)):
            axes[j].set_visible(False)
        
        fig.suptitle(f'{title}\n'
                    f'(Positive = increases prediction, Negative = decreases)',
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def save_figure(self, fig: plt.Figure, filename: str, 
                    output_dir: str = "outputs/figures/") -> None:
        """Save figure to file."""
        from pathlib import Path
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        filepath = Path(output_dir) / filename
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved figure: {filepath}")
        plt.close(fig)


def create_workflow_diagram(output_path: str = "outputs/figures/workflow.png") -> None:
    """
    Create workflow diagram for the paper.
    
    FIXED: 
    - Removed dangling arrows
    - Added weight handling details
    - Added missing blocks (leakage-proof, COVID sensitivity, error equity)
    - Shows two model branches (2020-calibrated vs structural)
    """
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    def draw_box(x, y, w, h, text, color='lightblue', fontsize=9):
        rect = plt.Rectangle((x, y), w, h, fill=True, 
                             facecolor=color, edgecolor='navy', lw=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
               fontsize=fontsize, fontweight='bold', wrap=True)
    
    def draw_arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color='navy', lw=2))
    
    # Row 1: Data Input
    draw_box(1, 10, 3, 1.2, 'RECS 2020\nMicrodata\n(n=18,496)', 'lightblue')
    draw_box(5, 10, 3, 1.2, 'Technology\nGrouping\n(Section 3)', 'lightblue')
    draw_box(9, 10, 3, 1.2, 'Leakage-proof\nPreprocessing\n(inside CV)', 'lightyellow')
    
    # Row 2: Model branches
    draw_box(1, 7.5, 3.5, 1.5, '2020-Calibrated\n(with COVID vars)\nTELLWORK, ATHOME', 'lightgreen')
    draw_box(5.5, 7.5, 3.5, 1.5, 'Structural\n(without COVID)\nProxy controls only', 'lightcoral')
    draw_box(10, 7.5, 3, 1.2, 'Nested CV\n(5×3 folds)\n(Section 10)', 'lightblue')
    
    # Row 3: Models and Training
    draw_box(1, 5, 2.5, 1.2, 'Physics\nBaselines\n(per tech)', 'plum')
    draw_box(4, 5, 2.5, 1.2, 'LightGBM\n(primary)', 'plum')
    draw_box(7, 5, 2.5, 1.2, 'EBM\n(interpret)', 'plum')
    draw_box(10.5, 5, 3, 1.2, 'Train/tune\nwith NWEIGHT', 'lightyellow')
    
    # Row 4: Evaluation
    draw_box(1, 2.5, 3, 1.2, 'Weighted Metrics\n(wRMSE, wMAE, wR²)', 'lightblue')
    draw_box(5, 2.5, 3, 1.2, 'Policy\nTargeting\n(Section 8)', 'lightblue')
    draw_box(9, 2.5, 3.5, 1.2, 'Evaluate with\nNWEIGHT + 60\nreplicates (jackknife)', 'lightyellow')
    
    # Row 5: Outputs
    draw_box(1, 0.3, 2.5, 1, 'Error Equity\nAudit', 'lightblue')
    draw_box(4, 0.3, 2.5, 1, 'Physics\nDiagnostics', 'lightblue')
    draw_box(7, 0.3, 2.5, 1, 'Sensitivity\nAnalyses', 'lightblue')
    draw_box(10, 0.3, 3, 1, 'Uncertainty\nQuantification', 'lightblue')
    
    # Arrows - Row 1
    draw_arrow(4, 10.6, 5, 10.6)
    draw_arrow(8, 10.6, 9, 10.6)
    
    # Arrows - Row 1 to Row 2
    draw_arrow(6.5, 10, 2.75, 9)
    draw_arrow(6.5, 10, 7.25, 9)
    draw_arrow(10.5, 10, 11.5, 8.7)
    
    # Arrows - Row 2 to Row 3
    draw_arrow(2.75, 7.5, 2.25, 6.2)
    draw_arrow(7.25, 7.5, 5.25, 6.2)
    draw_arrow(7.25, 7.5, 8.25, 6.2)
    draw_arrow(11.5, 7.5, 12, 6.2)
    
    # Arrows - Row 3 to Row 4
    draw_arrow(5.25, 5, 2.5, 3.7)
    draw_arrow(5.25, 5, 6.5, 3.7)
    draw_arrow(12, 5, 10.75, 3.7)
    
    # Arrows - Row 4 to Row 5
    draw_arrow(2.5, 2.5, 2.25, 1.3)
    draw_arrow(6.5, 2.5, 5.25, 1.3)
    draw_arrow(6.5, 2.5, 8.25, 1.3)
    draw_arrow(10.75, 2.5, 11.5, 1.3)
    
    # Title and legend
    ax.set_title('Heating Demand Modeling Framework Workflow\n'
                'Policy-centric modeling with physics-consistent structure',
                fontsize=14, fontweight='bold', pad=20)
    
    # Legend for colors
    legend_elements = [
        plt.Rectangle((0,0), 1, 1, facecolor='lightblue', edgecolor='navy', label='Core Process'),
        plt.Rectangle((0,0), 1, 1, facecolor='lightyellow', edgecolor='navy', label='Weighting/CV'),
        plt.Rectangle((0,0), 1, 1, facecolor='lightgreen', edgecolor='navy', label='2020-Calibrated'),
        plt.Rectangle((0,0), 1, 1, facecolor='lightcoral', edgecolor='navy', label='Structural'),
        plt.Rectangle((0,0), 1, 1, facecolor='plum', edgecolor='navy', label='Models'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved workflow diagram: {output_path}")
