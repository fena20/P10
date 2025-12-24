"""
Uncertainty Quantification using Jackknife Replicate Weights

Implements Section 9: Uncertainty via replicate weights + refit-sensitivity

9.1 Design-based uncertainty:
- Compute metrics with NWEIGHT, then recompute with NWEIGHT1...60
- Apply RECS jackknife variance formula

9.2 Limitations:
- Does not include variance from re-training under each replicate

9.3 Refit-sensitivity check:
- Refit model on subset of replicates (10/60)
- Quantify gap between metric-only and full refit uncertainty
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from sklearn.base import clone
import warnings

from src.utils.helpers import logger, Timer
from src.evaluation.metrics import WeightedMetrics


class JackknifeUncertainty:
    """
    Jackknife variance estimation using RECS replicate weights.
    
    RECS uses jackknife variance estimation:
    Var(theta) = (n-1)/n * sum_{i=1}^{n} (theta_i - theta_bar)^2
    
    where n = 60 replicate weights, theta_i is the estimate using
    replicate weight i, and theta_bar is the mean across replicates.
    """
    
    def __init__(self, 
                 n_replicates: int = 60,
                 confidence_level: float = 0.95):
        """
        Initialize jackknife uncertainty estimator.
        
        Parameters
        ----------
        n_replicates : int
            Number of replicate weights (RECS has 60)
        confidence_level : float
            Confidence level for intervals
        """
        self.n_replicates = n_replicates
        self.confidence_level = confidence_level
        self.z_score = self._get_z_score(confidence_level)
        
    def _get_z_score(self, confidence_level: float) -> float:
        """Get z-score for confidence interval."""
        from scipy import stats
        alpha = 1 - confidence_level
        return stats.norm.ppf(1 - alpha / 2)
    
    def compute_metric_uncertainty(self,
                                    y_true: np.ndarray,
                                    y_pred: np.ndarray,
                                    main_weights: np.ndarray,
                                    replicate_weights: pd.DataFrame,
                                    metric_func: Callable[[np.ndarray, np.ndarray, np.ndarray], float]) -> Dict[str, float]:
        """
        Compute uncertainty for a single metric using jackknife.
        
        Parameters
        ----------
        y_true : array
            True values
        y_pred : array
            Predicted values
        main_weights : array
            Main survey weights (NWEIGHT)
        replicate_weights : DataFrame
            Replicate weights (NWEIGHT1-NWEIGHT60)
        metric_func : callable
            Function(y_true, y_pred, weights) -> metric value
            
        Returns
        -------
        dict
            Estimate, SE, and CI
        """
        # Main estimate
        main_estimate = metric_func(y_true, y_pred, main_weights)
        
        # Replicate estimates
        rep_estimates = []
        for col in replicate_weights.columns[:self.n_replicates]:
            rep_w = replicate_weights[col].values
            rep_estimate = metric_func(y_true, y_pred, rep_w)
            rep_estimates.append(rep_estimate)
        
        rep_estimates = np.array(rep_estimates)
        
        # Jackknife variance
        n = len(rep_estimates)
        variance = (n - 1) / n * np.sum((rep_estimates - rep_estimates.mean()) ** 2)
        se = np.sqrt(variance)
        
        # Confidence interval
        ci_lower = main_estimate - self.z_score * se
        ci_upper = main_estimate + self.z_score * se
        
        return {
            'estimate': main_estimate,
            'se': se,
            'variance': variance,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'replicate_mean': rep_estimates.mean(),
            'replicate_std': rep_estimates.std(),
            'n_replicates': n
        }
    
    def compute_all_metrics_uncertainty(self,
                                         y_true: np.ndarray,
                                         y_pred: np.ndarray,
                                         main_weights: np.ndarray,
                                         replicate_weights: pd.DataFrame) -> pd.DataFrame:
        """
        Compute uncertainty for all standard metrics.
        
        Parameters
        ----------
        y_true : array
            True values
        y_pred : array
            Predicted values
        main_weights : array
            Main survey weights
        replicate_weights : DataFrame
            Replicate weights
            
        Returns
        -------
        DataFrame
            Uncertainty estimates for all metrics
        """
        metrics = WeightedMetrics()
        
        metric_funcs = {
            'weighted_rmse': metrics.weighted_rmse,
            'weighted_mae': metrics.weighted_mae,
            'weighted_r2': metrics.weighted_r2,
            'weighted_mape': metrics.weighted_mape,
            'weighted_bias': metrics.weighted_bias
        }
        
        results = []
        for metric_name, func in metric_funcs.items():
            try:
                uncertainty = self.compute_metric_uncertainty(
                    y_true, y_pred, main_weights, replicate_weights, func
                )
                uncertainty['metric'] = metric_name
                results.append(uncertainty)
            except Exception as e:
                logger.warning(f"Error computing uncertainty for {metric_name}: {e}")
        
        return pd.DataFrame(results)
    
    def compute_delta_metrics_ci(self,
                                  y_true: np.ndarray,
                                  y_pred_a: np.ndarray,
                                  y_pred_b: np.ndarray,
                                  main_weights: np.ndarray,
                                  replicate_weights: pd.DataFrame) -> Dict[str, Dict]:
        """
        Compute CI for difference in metrics between two models.
        
        Δ = metric(model_a) - metric(model_b)
        
        This is more appropriate than t-test on 5 folds, as it uses
        the full cross-fitted predictions and replicate weights.
        
        Parameters
        ----------
        y_true : array
            True values
        y_pred_a : array
            Predictions from model A (e.g., Monolithic)
        y_pred_b : array
            Predictions from model B (e.g., Split)
        main_weights : array
            Main survey weights
        replicate_weights : DataFrame
            Replicate weights (NWEIGHT1-60)
            
        Returns
        -------
        dict
            Delta metrics with CIs
        """
        metrics = WeightedMetrics()
        
        results = {}
        
        for metric_name, func in [('delta_rmse', metrics.weighted_rmse),
                                   ('delta_mae', metrics.weighted_mae),
                                   ('delta_r2', metrics.weighted_r2),
                                   ('delta_bias', metrics.weighted_bias)]:
            
            # Main estimate
            main_a = func(y_true, y_pred_a, main_weights)
            main_b = func(y_true, y_pred_b, main_weights)
            
            if metric_name == 'delta_r2':
                # For R², higher is better, so delta = B - A
                main_delta = main_b - main_a
            else:
                # For RMSE/MAE/Bias, lower is better, so delta = A - B
                main_delta = main_a - main_b
            
            # Replicate estimates
            rep_deltas = []
            for col in replicate_weights.columns[:self.n_replicates]:
                rep_w = replicate_weights[col].values
                rep_a = func(y_true, y_pred_a, rep_w)
                rep_b = func(y_true, y_pred_b, rep_w)
                
                if metric_name == 'delta_r2':
                    rep_delta = rep_b - rep_a
                else:
                    rep_delta = rep_a - rep_b
                rep_deltas.append(rep_delta)
            
            rep_deltas = np.array(rep_deltas)
            n = len(rep_deltas)
            
            # Jackknife variance
            variance = (n - 1) / n * np.sum((rep_deltas - rep_deltas.mean()) ** 2)
            se = np.sqrt(variance)
            
            results[metric_name] = {
                'estimate': main_delta,
                'se': se,
                'ci_lower': main_delta - self.z_score * se,
                'ci_upper': main_delta + self.z_score * se,
                'model_a': main_a,
                'model_b': main_b
            }
        
        return results


class RefitSensitivity:
    """
    Refit-sensitivity analysis (Section 9.3).
    
    Tests whether "metric-only jackknife" underestimates variability
    by refitting the model under a subset of replicate weights.
    """
    
    def __init__(self,
                 n_refit_replicates: int = 10,
                 refit_indices: Optional[List[int]] = None):
        """
        Initialize refit sensitivity analysis.
        
        Parameters
        ----------
        n_refit_replicates : int
            Number of replicates to refit (default 10 of 60)
        refit_indices : list, optional
            Specific replicate indices to use (1-indexed)
        """
        self.n_refit_replicates = n_refit_replicates
        
        # Default: evenly spaced indices
        if refit_indices is None:
            self.refit_indices = [1, 7, 13, 19, 25, 31, 37, 43, 49, 55]
        else:
            self.refit_indices = refit_indices
    
    def run_refit_sensitivity(self,
                               X_train: pd.DataFrame,
                               y_train: np.ndarray,
                               X_test: pd.DataFrame,
                               y_test: np.ndarray,
                               main_weights: np.ndarray,
                               replicate_weights_train: pd.DataFrame,
                               replicate_weights_test: pd.DataFrame,
                               model_factory: Callable,
                               model_params: Dict = None) -> Dict[str, Any]:
        """
        Run refit sensitivity analysis.
        
        For each selected replicate:
        1. Refit model using replicate training weights
        2. Compute predictions
        3. Compute metrics using replicate test weights
        
        Compare variance from refit vs metric-only jackknife.
        
        Parameters
        ----------
        X_train : DataFrame
            Training features
        y_train : array
            Training targets
        X_test : DataFrame
            Test features
        y_test : array
            Test targets
        main_weights : array
            Main test weights
        replicate_weights_train : DataFrame
            Training replicate weights
        replicate_weights_test : DataFrame
            Test replicate weights
        model_factory : callable
            Function to create model
        model_params : dict
            Model parameters
            
        Returns
        -------
        dict
            Sensitivity analysis results
        """
        model_params = model_params or {}
        metrics = WeightedMetrics()
        
        with Timer(f"Running refit sensitivity on {len(self.refit_indices)} replicates"):
            
            # Metric-only estimates (same predictions, different weights)
            # First fit with main weights to get base predictions
            base_model = model_factory(**model_params)
            main_train_weights = main_weights[:len(y_train)] if len(main_weights) > len(y_train) else main_weights
            
            # Handle different model interfaces
            if hasattr(base_model, 'fit'):
                try:
                    base_model.fit(X_train, y_train, sample_weight=main_train_weights[:len(X_train)])
                except TypeError:
                    base_model.fit(X_train, y_train)
            
            base_predictions = base_model.predict(X_test)
            
            # Metric-only jackknife (same predictions, varying test weights)
            metric_only_estimates = []
            for idx in self.refit_indices:
                col = f'NWEIGHT{idx}'
                if col in replicate_weights_test.columns:
                    rep_test_w = replicate_weights_test[col].values
                    rmse = metrics.weighted_rmse(y_test, base_predictions, rep_test_w)
                    metric_only_estimates.append(rmse)
            
            metric_only_estimates = np.array(metric_only_estimates)
            n_metric = len(metric_only_estimates)
            metric_only_var = (n_metric - 1) / n_metric * np.sum(
                (metric_only_estimates - metric_only_estimates.mean()) ** 2
            ) if n_metric > 1 else 0
            
            # Full refit estimates (refit model with replicate weights)
            refit_estimates = []
            refit_predictions = []
            
            for idx in self.refit_indices:
                train_col = f'NWEIGHT{idx}'
                test_col = f'NWEIGHT{idx}'
                
                if train_col not in replicate_weights_train.columns:
                    continue
                
                rep_train_w = replicate_weights_train[train_col].values
                rep_test_w = replicate_weights_test[test_col].values
                
                # Refit model
                refit_model = model_factory(**model_params)
                try:
                    if hasattr(refit_model, 'fit'):
                        try:
                            refit_model.fit(X_train, y_train, sample_weight=rep_train_w[:len(X_train)])
                        except TypeError:
                            refit_model.fit(X_train, y_train)
                    
                    rep_predictions = refit_model.predict(X_test)
                    refit_predictions.append(rep_predictions)
                    
                    # Compute metric with replicate test weights
                    rmse = metrics.weighted_rmse(y_test, rep_predictions, rep_test_w)
                    refit_estimates.append(rmse)
                except Exception as e:
                    logger.warning(f"Error refitting with replicate {idx}: {e}")
                    continue
            
            refit_estimates = np.array(refit_estimates)
            n_refit = len(refit_estimates)
            
            if n_refit > 1:
                refit_var = (n_refit - 1) / n_refit * np.sum(
                    (refit_estimates - refit_estimates.mean()) ** 2
                )
            else:
                refit_var = 0
            
            # Compare variances
            variance_ratio = refit_var / metric_only_var if metric_only_var > 0 else np.inf
            
            # Prediction variability across refits
            if refit_predictions:
                pred_array = np.array(refit_predictions)
                pred_std_per_sample = pred_array.std(axis=0)
                mean_pred_std = pred_std_per_sample.mean()
            else:
                mean_pred_std = 0
            
            return {
                'metric_only': {
                    'estimates': metric_only_estimates,
                    'variance': metric_only_var,
                    'se': np.sqrt(metric_only_var),
                    'mean': metric_only_estimates.mean() if len(metric_only_estimates) > 0 else np.nan
                },
                'full_refit': {
                    'estimates': refit_estimates,
                    'variance': refit_var,
                    'se': np.sqrt(refit_var),
                    'mean': refit_estimates.mean() if len(refit_estimates) > 0 else np.nan
                },
                'comparison': {
                    'variance_ratio': variance_ratio,
                    'se_ratio': np.sqrt(variance_ratio) if variance_ratio != np.inf else np.inf,
                    'underestimation_factor': np.sqrt(variance_ratio) - 1 if variance_ratio > 1 else 0,
                    'mean_prediction_std': mean_pred_std
                },
                'n_replicates_used': n_refit,
                'refit_indices': self.refit_indices[:n_refit]
            }


class CombinedUncertainty:
    """
    Combined uncertainty quantification framework.
    
    Combines:
    - Jackknife variance from replicate weights
    - Refit sensitivity adjustment (if gap is non-negligible)
    """
    
    def __init__(self,
                 n_replicates: int = 60,
                 confidence_level: float = 0.95,
                 n_refit_for_sensitivity: int = 10):
        self.jackknife = JackknifeUncertainty(n_replicates, confidence_level)
        self.refit_sensitivity = RefitSensitivity(n_refit_for_sensitivity)
        self.confidence_level = confidence_level
        
    def compute_uncertainty_with_adjustment(self,
                                             y_true: np.ndarray,
                                             y_pred: np.ndarray,
                                             main_weights: np.ndarray,
                                             replicate_weights: pd.DataFrame,
                                             refit_results: Optional[Dict] = None) -> pd.DataFrame:
        """
        Compute uncertainty with optional refit adjustment.
        
        If refit_results show metric-only underestimates variance,
        adjust SEs accordingly.
        
        Parameters
        ----------
        y_true : array
            True values
        y_pred : array
            Predicted values
        main_weights : array
            Main weights
        replicate_weights : DataFrame
            Replicate weights
        refit_results : dict, optional
            Results from RefitSensitivity.run_refit_sensitivity()
            
        Returns
        -------
        DataFrame
            Adjusted uncertainty estimates
        """
        # Base jackknife uncertainty
        base_uncertainty = self.jackknife.compute_all_metrics_uncertainty(
            y_true, y_pred, main_weights, replicate_weights
        )
        
        # Apply adjustment if refit results available
        if refit_results is not None:
            adjustment_factor = refit_results['comparison'].get('se_ratio', 1.0)
            
            if adjustment_factor > 1.0:
                logger.info(f"Applying refit adjustment factor: {adjustment_factor:.2f}")
                
                # Adjust SEs and CIs
                base_uncertainty['se_adjusted'] = base_uncertainty['se'] * adjustment_factor
                
                z = self.jackknife.z_score
                base_uncertainty['ci_lower_adjusted'] = (
                    base_uncertainty['estimate'] - z * base_uncertainty['se_adjusted']
                )
                base_uncertainty['ci_upper_adjusted'] = (
                    base_uncertainty['estimate'] + z * base_uncertainty['se_adjusted']
                )
                base_uncertainty['adjustment_factor'] = adjustment_factor
            else:
                base_uncertainty['adjustment_factor'] = 1.0
                base_uncertainty['se_adjusted'] = base_uncertainty['se']
                base_uncertainty['ci_lower_adjusted'] = base_uncertainty['ci_lower']
                base_uncertainty['ci_upper_adjusted'] = base_uncertainty['ci_upper']
        else:
            base_uncertainty['adjustment_factor'] = np.nan
            base_uncertainty['se_adjusted'] = base_uncertainty['se']
            base_uncertainty['ci_lower_adjusted'] = base_uncertainty['ci_lower']
            base_uncertainty['ci_upper_adjusted'] = base_uncertainty['ci_upper']
        
        return base_uncertainty


def format_uncertainty_table(uncertainty_df: pd.DataFrame,
                              include_adjusted: bool = True) -> pd.DataFrame:
    """
    Format uncertainty results for publication.
    
    Parameters
    ----------
    uncertainty_df : DataFrame
        Results from uncertainty computation
    include_adjusted : bool
        Whether to include adjusted estimates
        
    Returns
    -------
    DataFrame
        Formatted table
    """
    formatted = {
        'Metric': [],
        'Estimate': [],
        'SE': [],
        '95% CI': []
    }
    
    for _, row in uncertainty_df.iterrows():
        formatted['Metric'].append(row['metric'])
        formatted['Estimate'].append(f"{row['estimate']:.3f}")
        
        if include_adjusted and 'se_adjusted' in row and not pd.isna(row.get('adjustment_factor')):
            formatted['SE'].append(f"{row['se_adjusted']:.3f}")
            formatted['95% CI'].append(f"[{row['ci_lower_adjusted']:.3f}, {row['ci_upper_adjusted']:.3f}]")
        else:
            formatted['SE'].append(f"{row['se']:.3f}")
            formatted['95% CI'].append(f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]")
    
    return pd.DataFrame(formatted)
