"""
Evaluation Metrics for Heating Demand Modeling

Implements Section 8: Evaluation metrics (scientific + policy)
Including weighted metrics and physics diagnostics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.utils.helpers import logger, compute_weighted_quantile


class WeightedMetrics:
    """
    Weighted evaluation metrics for survey data.
    
    Implements:
    - Weighted RMSE, MAE, R², MAPE
    - Support for replicate weights (jackknife SE)
    """
    
    def __init__(self):
        self.metrics_cache_ = {}
        
    @staticmethod
    def weighted_rmse(y_true: np.ndarray, y_pred: np.ndarray, 
                      weights: np.ndarray) -> float:
        """
        Compute weighted Root Mean Squared Error.
        
        RMSE_w = sqrt(sum(w_i * (y_i - y_hat_i)^2) / sum(w_i))
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        weights = np.asarray(weights)
        
        # Handle missing values
        valid = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isnan(weights))
        y_true = y_true[valid]
        y_pred = y_pred[valid]
        weights = weights[valid]
        
        if len(y_true) == 0:
            return np.nan
        
        squared_errors = (y_true - y_pred) ** 2
        weighted_mse = np.sum(weights * squared_errors) / np.sum(weights)
        return np.sqrt(weighted_mse)
    
    @staticmethod
    def weighted_mae(y_true: np.ndarray, y_pred: np.ndarray,
                     weights: np.ndarray) -> float:
        """
        Compute weighted Mean Absolute Error.
        
        MAE_w = sum(w_i * |y_i - y_hat_i|) / sum(w_i)
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        weights = np.asarray(weights)
        
        valid = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isnan(weights))
        y_true = y_true[valid]
        y_pred = y_pred[valid]
        weights = weights[valid]
        
        if len(y_true) == 0:
            return np.nan
        
        abs_errors = np.abs(y_true - y_pred)
        return np.sum(weights * abs_errors) / np.sum(weights)
    
    @staticmethod
    def weighted_r2(y_true: np.ndarray, y_pred: np.ndarray,
                    weights: np.ndarray) -> float:
        """
        Compute weighted R-squared using SUBGROUP mean.
        
        R²_w = 1 - sum(w_i * (y_i - y_hat_i)^2) / sum(w_i * (y_i - y_bar_w)^2)
        
        Note: For subgroup analysis, SS_tot uses the subgroup's weighted mean,
        not the global mean. This can result in lower R² for subgroups with
        lower variance than the full population.
        
        For within-subgroup performance, consider using weighted correlation (r)
        or normalized RMSE (NRMSE = RMSE/mean) as complementary metrics.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        weights = np.asarray(weights)
        
        valid = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isnan(weights))
        y_true = y_true[valid]
        y_pred = y_pred[valid]
        weights = weights[valid]
        
        if len(y_true) == 0:
            return np.nan
        
        # Use SUBGROUP weighted mean for SS_tot
        weighted_mean = np.sum(weights * y_true) / np.sum(weights)
        
        ss_res = np.sum(weights * (y_true - y_pred) ** 2)
        ss_tot = np.sum(weights * (y_true - weighted_mean) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        return 1 - (ss_res / ss_tot)
    
    @staticmethod
    def weighted_correlation(y_true: np.ndarray, y_pred: np.ndarray,
                             weights: np.ndarray) -> float:
        """
        Compute weighted Pearson correlation coefficient.
        
        More robust than R² for subgroup analysis as it measures
        the strength of linear relationship, not predictive variance explained.
        
        Returns r (not r²) - square it for explained correlation.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        weights = np.asarray(weights)
        
        valid = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isnan(weights))
        y_true = y_true[valid]
        y_pred = y_pred[valid]
        weights = weights[valid]
        
        if len(y_true) < 2:
            return np.nan
        
        w_sum = np.sum(weights)
        mean_true = np.sum(weights * y_true) / w_sum
        mean_pred = np.sum(weights * y_pred) / w_sum
        
        cov = np.sum(weights * (y_true - mean_true) * (y_pred - mean_pred)) / w_sum
        std_true = np.sqrt(np.sum(weights * (y_true - mean_true)**2) / w_sum)
        std_pred = np.sqrt(np.sum(weights * (y_pred - mean_pred)**2) / w_sum)
        
        if std_true * std_pred == 0:
            return 0.0
        
        return cov / (std_true * std_pred)
    
    @staticmethod
    def weighted_mape(y_true: np.ndarray, y_pred: np.ndarray,
                      weights: np.ndarray, epsilon: float = 1.0) -> float:
        """
        DEPRECATED: Use weighted_wape or weighted_nmae instead.
        
        MAPE explodes for small denominators (low-HDD/low-use cases).
        """
        # Return NaN to signal this metric should not be used
        return np.nan
    
    @staticmethod
    def weighted_wape(y_true: np.ndarray, y_pred: np.ndarray,
                      weights: np.ndarray) -> float:
        """
        Compute Weighted Absolute Percentage Error (WAPE).
        
        WAPE = sum(w_i * |y_i - y_hat_i|) / sum(w_i * |y_i|) * 100
        
        More stable than MAPE because it uses aggregate weighted sums.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        weights = np.asarray(weights)
        
        valid = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isnan(weights))
        y_true = y_true[valid]
        y_pred = y_pred[valid]
        weights = weights[valid]
        
        if len(y_true) == 0:
            return np.nan
        
        weighted_abs_error = np.sum(weights * np.abs(y_true - y_pred))
        weighted_abs_actual = np.sum(weights * np.abs(y_true))
        
        if weighted_abs_actual == 0:
            return np.nan
        
        return weighted_abs_error / weighted_abs_actual * 100
    
    @staticmethod
    def weighted_nmae(y_true: np.ndarray, y_pred: np.ndarray,
                      weights: np.ndarray) -> float:
        """
        Compute Normalized MAE (nMAE).
        
        nMAE = MAE / weighted_mean(y_true) * 100
        
        Scale-free metric that doesn't explode for small values.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        weights = np.asarray(weights)
        
        valid = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isnan(weights))
        y_true = y_true[valid]
        y_pred = y_pred[valid]
        weights = weights[valid]
        
        if len(y_true) == 0:
            return np.nan
        
        weighted_mae = np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)
        weighted_mean = np.sum(weights * y_true) / np.sum(weights)
        
        if weighted_mean == 0:
            return np.nan
        
        return weighted_mae / weighted_mean * 100
    
    @staticmethod
    def weighted_bias(y_true: np.ndarray, y_pred: np.ndarray,
                      weights: np.ndarray) -> float:
        """
        Compute weighted mean bias (mean error).
        
        Bias_w = sum(w_i * (y_hat_i - y_i)) / sum(w_i)
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        weights = np.asarray(weights)
        
        valid = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isnan(weights))
        y_true = y_true[valid]
        y_pred = y_pred[valid]
        weights = weights[valid]
        
        if len(y_true) == 0:
            return np.nan
        
        errors = y_pred - y_true
        return np.sum(weights * errors) / np.sum(weights)
    
    def compute_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                            weights: np.ndarray) -> Dict[str, float]:
        """
        Compute all weighted metrics.
        
        Parameters
        ----------
        y_true : array
            True values
        y_pred : array
            Predicted values
        weights : array
            Sample weights
            
        Returns
        -------
        dict
            Dictionary of metric name -> value
            
        Note: MAPE is excluded (unstable for small denominators).
        WAPE and nMAE are used as stable alternatives.
        """
        return {
            'weighted_rmse': self.weighted_rmse(y_true, y_pred, weights),
            'weighted_mae': self.weighted_mae(y_true, y_pred, weights),
            'weighted_r2': self.weighted_r2(y_true, y_pred, weights),
            'weighted_wape': self.weighted_wape(y_true, y_pred, weights),
            'weighted_nmae': self.weighted_nmae(y_true, y_pred, weights),
            'weighted_bias': self.weighted_bias(y_true, y_pred, weights),
            'n_samples': len(y_true[~np.isnan(y_true)])
        }
    
    def compute_metrics_by_group(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  weights: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
        """
        Compute metrics stratified by groups.
        
        Parameters
        ----------
        y_true : array
            True values
        y_pred : array
            Predicted values
        weights : array
            Sample weights
        groups : array
            Group labels
            
        Returns
        -------
        DataFrame
            Metrics by group
        """
        results = []
        
        for group in np.unique(groups):
            mask = groups == group
            metrics = self.compute_all_metrics(
                y_true[mask], y_pred[mask], weights[mask]
            )
            metrics['group'] = group
            results.append(metrics)
        
        return pd.DataFrame(results)


class PhysicsDiagnostics:
    """
    Physics sanity checks for heating demand predictions.
    
    Implements Section 8 physics diagnostics:
    - Fraction of test cases with negative predictions
    - Fraction with wrong-direction HDD sensitivity
    - Residual bias vs HDD bins
    """
    
    HDD_RESIDUAL_BINS = [0, 1000, 2000, 3000, 4000, 5000, 6000, 8000, float('inf')]
    HDD_BIN_LABELS = ['0-1k', '1-2k', '2-3k', '3-4k', '4-5k', '5-6k', '6-8k', '8k+']
    
    def __init__(self):
        pass
    
    @staticmethod
    def negative_prediction_rate(y_pred: np.ndarray, 
                                  weights: Optional[np.ndarray] = None) -> float:
        """
        Compute fraction of predictions that are negative.
        
        Parameters
        ----------
        y_pred : array
            Predicted values
        weights : array, optional
            Sample weights (for weighted rate)
            
        Returns
        -------
        float
            Fraction of negative predictions (0-1)
        """
        y_pred = np.asarray(y_pred)
        
        if weights is None:
            return np.mean(y_pred < 0)
        
        weights = np.asarray(weights)
        valid = ~np.isnan(y_pred) & ~np.isnan(weights)
        return np.sum(weights[valid] * (y_pred[valid] < 0)) / np.sum(weights[valid])
    
    @staticmethod
    def hdd_sensitivity_direction(y_pred: np.ndarray, 
                                   hdd: np.ndarray,
                                   weights: Optional[np.ndarray] = None,
                                   expected_direction: str = 'positive') -> Dict[str, Any]:
        """
        Check if HDD sensitivity has expected direction.
        
        For space heating, higher HDD should lead to higher energy use.
        
        Parameters
        ----------
        y_pred : array
            Predicted values
        hdd : array
            HDD values
        weights : array, optional
            Sample weights
        expected_direction : str
            Expected direction ('positive' or 'negative')
            
        Returns
        -------
        dict
            Analysis results including correlation and violation rate
        """
        y_pred = np.asarray(y_pred)
        hdd = np.asarray(hdd)
        
        valid = ~(np.isnan(y_pred) | np.isnan(hdd))
        y_valid = y_pred[valid]
        hdd_valid = hdd[valid]
        
        if weights is not None:
            weights = np.asarray(weights)[valid]
        
        # Compute correlation
        if weights is not None:
            # Weighted correlation
            w_mean_y = np.average(y_valid, weights=weights)
            w_mean_hdd = np.average(hdd_valid, weights=weights)
            
            cov = np.sum(weights * (y_valid - w_mean_y) * (hdd_valid - w_mean_hdd)) / np.sum(weights)
            std_y = np.sqrt(np.sum(weights * (y_valid - w_mean_y)**2) / np.sum(weights))
            std_hdd = np.sqrt(np.sum(weights * (hdd_valid - w_mean_hdd)**2) / np.sum(weights))
            
            correlation = cov / (std_y * std_hdd) if std_y * std_hdd > 0 else 0
        else:
            correlation = np.corrcoef(y_valid, hdd_valid)[0, 1]
        
        # Check direction
        is_correct_direction = (
            (expected_direction == 'positive' and correlation > 0) or
            (expected_direction == 'negative' and correlation < 0)
        )
        
        # Estimate local monotonicity violations
        # Sort by HDD and check for decreasing predicted values where increase expected
        sorted_idx = np.argsort(hdd_valid)
        y_sorted = y_valid[sorted_idx]
        
        # Count local violations (simple diff-based check)
        if expected_direction == 'positive':
            violations = np.sum(np.diff(y_sorted) < 0)
        else:
            violations = np.sum(np.diff(y_sorted) > 0)
        
        violation_rate = violations / (len(y_sorted) - 1) if len(y_sorted) > 1 else 0
        
        return {
            'correlation': correlation,
            'is_correct_direction': is_correct_direction,
            'local_violation_rate': violation_rate,
            'expected_direction': expected_direction,
            'n_samples': len(y_valid)
        }
    
    def residual_bias_by_hdd(self, y_true: np.ndarray, y_pred: np.ndarray,
                             hdd: np.ndarray, weights: np.ndarray) -> pd.DataFrame:
        """
        Compute residual bias stratified by HDD bins.
        
        Parameters
        ----------
        y_true : array
            True values
        y_pred : array
            Predicted values
        hdd : array
            HDD values
        weights : array
            Sample weights
            
        Returns
        -------
        DataFrame
            Bias statistics by HDD bin
        """
        residuals = y_pred - y_true
        
        # Bin HDD
        hdd_bins = pd.cut(hdd, bins=self.HDD_RESIDUAL_BINS, labels=self.HDD_BIN_LABELS)
        
        results = []
        for bin_label in self.HDD_BIN_LABELS:
            mask = hdd_bins == bin_label
            if mask.sum() == 0:
                continue
            
            bin_residuals = residuals[mask]
            bin_weights = weights[mask]
            bin_true = y_true[mask]
            
            valid = ~np.isnan(bin_residuals)
            if valid.sum() == 0:
                continue
            
            weighted_bias = np.sum(bin_weights[valid] * bin_residuals[valid]) / np.sum(bin_weights[valid])
            weighted_mae = np.sum(bin_weights[valid] * np.abs(bin_residuals[valid])) / np.sum(bin_weights[valid])
            weighted_mean_true = np.sum(bin_weights[valid] * bin_true[valid]) / np.sum(bin_weights[valid])
            
            results.append({
                'hdd_bin': bin_label,
                'n_samples': mask.sum(),
                'total_weight': bin_weights.sum(),
                'weighted_bias': weighted_bias,
                'weighted_mae': weighted_mae,
                'weighted_mean_true': weighted_mean_true,
                'bias_pct': (weighted_bias / weighted_mean_true * 100) if weighted_mean_true != 0 else np.nan
            })
        
        return pd.DataFrame(results)
    
    def residual_bias_by_division(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   division: np.ndarray, weights: np.ndarray) -> pd.DataFrame:
        """
        Compute residual bias stratified by Census Division.
        
        Parameters
        ----------
        y_true : array
            True values
        y_pred : array
            Predicted values
        division : array
            Census Division values
        weights : array
            Sample weights
            
        Returns
        -------
        DataFrame
            Bias statistics by division
        """
        residuals = y_pred - y_true
        
        results = []
        for div in np.unique(division):
            mask = division == div
            if mask.sum() == 0:
                continue
            
            div_residuals = residuals[mask]
            div_weights = weights[mask]
            div_true = y_true[mask]
            
            valid = ~np.isnan(div_residuals)
            if valid.sum() == 0:
                continue
            
            weighted_bias = np.sum(div_weights[valid] * div_residuals[valid]) / np.sum(div_weights[valid])
            weighted_mae = np.sum(div_weights[valid] * np.abs(div_residuals[valid])) / np.sum(div_weights[valid])
            weighted_mean_true = np.sum(div_weights[valid] * div_true[valid]) / np.sum(div_weights[valid])
            
            results.append({
                'division': div,
                'n_samples': mask.sum(),
                'total_weight': div_weights.sum(),
                'weighted_bias': weighted_bias,
                'weighted_mae': weighted_mae,
                'weighted_mean_true': weighted_mean_true,
                'bias_pct': (weighted_bias / weighted_mean_true * 100) if weighted_mean_true != 0 else np.nan
            })
        
        return pd.DataFrame(results)
    
    def run_all_diagnostics(self, y_true: np.ndarray, y_pred: np.ndarray,
                            hdd: np.ndarray, weights: np.ndarray,
                            division: Optional[np.ndarray] = None,
                            tech_group: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Run all physics diagnostics.
        
        Parameters
        ----------
        y_true : array
            True values
        y_pred : array
            Predicted values
        hdd : array
            HDD values
        weights : array
            Sample weights
        division : array, optional
            Census Division values
        tech_group : array, optional
            Technology group labels
            
        Returns
        -------
        dict
            All diagnostic results
        """
        results = {}
        
        # Negative prediction rate
        results['negative_rate'] = self.negative_prediction_rate(y_pred, weights)
        
        # HDD sensitivity
        results['hdd_sensitivity'] = self.hdd_sensitivity_direction(
            y_pred, hdd, weights, expected_direction='positive'
        )
        
        # Residual bias by HDD
        results['bias_by_hdd'] = self.residual_bias_by_hdd(y_true, y_pred, hdd, weights)
        
        # Residual bias by division
        if division is not None:
            results['bias_by_division'] = self.residual_bias_by_division(
                y_true, y_pred, division, weights
            )
        
        # Tech-group specific diagnostics
        if tech_group is not None:
            tech_results = {}
            for group in np.unique(tech_group):
                mask = tech_group == group
                tech_results[group] = {
                    'negative_rate': self.negative_prediction_rate(y_pred[mask], weights[mask]),
                    'hdd_sensitivity': self.hdd_sensitivity_direction(
                        y_pred[mask], hdd[mask], weights[mask]
                    )
                }
            results['by_tech_group'] = tech_results
        
        return results


class ErrorEquityAnalysis:
    """
    Error equity analysis ("fairness audit") as per Section 11.
    
    Analyzes weighted error distributions across:
    - Income deciles
    - Housing types
    - Renters vs owners
    - Regions
    
    All metrics include:
    - Bias = weighted mean(Ŷ - Y)
    - nMAE = MAE / mean(Y) × 100 (normalized, scale-free)
    - Group n and weighted share
    """
    
    def __init__(self):
        pass
    
    def error_by_income(self, y_true: np.ndarray, y_pred: np.ndarray,
                        income: np.ndarray, weights: np.ndarray,
                        n_deciles: int = 10) -> pd.DataFrame:
        """
        Analyze errors by income categories (MONEYPY codes).
        
        Uses MONEYPY codes directly (1-16) rather than creating arbitrary deciles,
        ensuring consistency with policy targeting composition tables.
        
        Parameters
        ----------
        y_true : array
            True values
        y_pred : array
            Predicted values
        income : array
            MONEYPY income category codes (1-16)
        weights : array
            Sample weights
        n_deciles : int
            Ignored (kept for backward compatibility). Uses actual MONEYPY codes.
            
        Returns
        -------
        DataFrame
            Error metrics by income group
        """
        residuals = y_pred - y_true
        abs_errors = np.abs(residuals)
        
        # RECS 2020 MONEYPY income labels (16 categories)
        income_labels = {
            1: '<$5K', 2: '$5-7.5K', 3: '$7.5-10K', 4: '$10-12.5K',
            5: '$12.5-15K', 6: '$15-20K', 7: '$20-25K', 8: '$25-30K',
            9: '$30-35K', 10: '$35-40K', 11: '$40-50K', 12: '$50-60K',
            13: '$60-75K', 14: '$75-100K', 15: '$100-150K', 16: '$150K+'
        }
        
        results = []
        for income_code in sorted(np.unique(income)):
            if pd.isna(income_code):
                continue
            
            income_code = int(income_code)
            mask = income == income_code
            if mask.sum() == 0:
                continue
            
            grp_weights = weights[mask]
            grp_residuals = residuals[mask]
            grp_abs_errors = abs_errors[mask]
            grp_true = y_true[mask]
            
            valid = ~np.isnan(grp_residuals)
            
            weighted_bias = np.sum(grp_weights[valid] * grp_residuals[valid]) / np.sum(grp_weights[valid])
            weighted_mae = np.sum(grp_weights[valid] * grp_abs_errors[valid]) / np.sum(grp_weights[valid])
            weighted_rmse = np.sqrt(
                np.sum(grp_weights[valid] * grp_residuals[valid]**2) / np.sum(grp_weights[valid])
            )
            weighted_mean_true = np.sum(grp_weights[valid] * grp_true[valid]) / np.sum(grp_weights[valid])
            
            results.append({
                'income_code': income_code,
                'income_group': income_labels.get(income_code, f'Code {income_code}'),
                'n_samples': mask.sum(),
                'total_weight': grp_weights.sum(),
                'weighted_bias': weighted_bias,
                'weighted_mae': weighted_mae,
                'weighted_rmse': weighted_rmse,
                'weighted_mean_true': weighted_mean_true,
                'mae_pct': (weighted_mae / weighted_mean_true * 100) if weighted_mean_true != 0 else np.nan
            })
        
        return pd.DataFrame(results)
    
    def error_by_housing_type(self, y_true: np.ndarray, y_pred: np.ndarray,
                               housing_type: np.ndarray, weights: np.ndarray) -> pd.DataFrame:
        """
        Analyze errors by housing type.
        """
        residuals = y_pred - y_true
        
        # TYPEHUQ labels
        type_labels = {
            1: 'Mobile home',
            2: 'Single-family detached',
            3: 'Single-family attached',
            4: 'Apartment (2-4 units)',
            5: 'Apartment (5+ units)'
        }
        
        results = []
        for htype in np.unique(housing_type):
            mask = housing_type == htype
            if mask.sum() == 0:
                continue
            
            type_weights = weights[mask]
            type_residuals = residuals[mask]
            type_true = y_true[mask]
            
            valid = ~np.isnan(type_residuals)
            
            weighted_bias = np.sum(type_weights[valid] * type_residuals[valid]) / np.sum(type_weights[valid])
            weighted_mae = np.sum(type_weights[valid] * np.abs(type_residuals[valid])) / np.sum(type_weights[valid])
            weighted_mean_true = np.sum(type_weights[valid] * type_true[valid]) / np.sum(type_weights[valid])
            
            results.append({
                'housing_type': type_labels.get(htype, f'Unknown ({htype})'),
                'housing_code': htype,
                'n_samples': mask.sum(),
                'total_weight': type_weights.sum(),
                'weighted_bias': weighted_bias,
                'weighted_mae': weighted_mae,
                'weighted_mean_true': weighted_mean_true,
                'bias_pct': (weighted_bias / weighted_mean_true * 100) if weighted_mean_true != 0 else np.nan
            })
        
        return pd.DataFrame(results)
    
    def error_by_tenure(self, y_true: np.ndarray, y_pred: np.ndarray,
                        tenure: np.ndarray, weights: np.ndarray) -> pd.DataFrame:
        """
        Analyze errors by tenure (renter vs owner).
        """
        residuals = y_pred - y_true
        
        tenure_labels = {
            1: 'Owned',
            2: 'Rented',
            3: 'Occupied without payment'
        }
        
        results = []
        for ten in np.unique(tenure):
            mask = tenure == ten
            if mask.sum() == 0:
                continue
            
            ten_weights = weights[mask]
            ten_residuals = residuals[mask]
            ten_true = y_true[mask]
            
            valid = ~np.isnan(ten_residuals)
            
            weighted_bias = np.sum(ten_weights[valid] * ten_residuals[valid]) / np.sum(ten_weights[valid])
            weighted_mae = np.sum(ten_weights[valid] * np.abs(ten_residuals[valid])) / np.sum(ten_weights[valid])
            weighted_mean_true = np.sum(ten_weights[valid] * ten_true[valid]) / np.sum(ten_weights[valid])
            
            results.append({
                'tenure': tenure_labels.get(ten, f'Unknown ({ten})'),
                'tenure_code': ten,
                'n_samples': mask.sum(),
                'total_weight': ten_weights.sum(),
                'weighted_bias': weighted_bias,
                'weighted_mae': weighted_mae,
                'weighted_mean_true': weighted_mean_true,
                'bias_pct': (weighted_bias / weighted_mean_true * 100) if weighted_mean_true != 0 else np.nan
            })
        
        return pd.DataFrame(results)
    
    def run_full_audit(self, y_true: np.ndarray, y_pred: np.ndarray,
                       weights: np.ndarray, metadata: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Run full error equity audit.
        
        Parameters
        ----------
        y_true : array
            True values
        y_pred : array
            Predicted values
        weights : array
            Sample weights
        metadata : DataFrame
            DataFrame with TYPEHUQ, KOWNRENT, MONEYPY, DIVISION columns
            
        Returns
        -------
        dict
            Dictionary of audit DataFrames
        """
        results = {}
        
        # By housing type
        if 'TYPEHUQ' in metadata.columns:
            results['by_housing_type'] = self.error_by_housing_type(
                y_true, y_pred, metadata['TYPEHUQ'].values, weights
            )
        
        # By tenure
        if 'KOWNRENT' in metadata.columns:
            results['by_tenure'] = self.error_by_tenure(
                y_true, y_pred, metadata['KOWNRENT'].values, weights
            )
        
        # By income
        if 'MONEYPY' in metadata.columns:
            results['by_income'] = self.error_by_income(
                y_true, y_pred, metadata['MONEYPY'].values, weights
            )
        
        # By HDD/climate (if available)
        if 'HDD65' in metadata.columns:
            results['by_climate'] = self.error_by_climate(
                y_true, y_pred, metadata['HDD65'].values, weights
            )
        
        return results
    
    def error_by_climate(self, y_true: np.ndarray, y_pred: np.ndarray,
                         hdd: np.ndarray, weights: np.ndarray) -> pd.DataFrame:
        """
        Analyze errors by climate zone (HDD bins).
        
        Parameters
        ----------
        y_true : array
            True values
        y_pred : array
            Predicted values
        hdd : array
            Heating degree days
        weights : array
            Sample weights
            
        Returns
        -------
        DataFrame
            Error metrics by climate zone
        """
        residuals = y_pred - y_true
        abs_errors = np.abs(residuals)
        
        # Define climate zones based on HDD
        climate_bins = [
            ('Very Mild (<2k)', 0, 2000),
            ('Mild (2-4k)', 2000, 4000),
            ('Moderate (4-6k)', 4000, 6000),
            ('Cold (6-8k)', 6000, 8000),
            ('Very Cold (>8k)', 8000, float('inf'))
        ]
        
        results = []
        
        for label, hdd_min, hdd_max in climate_bins:
            mask = (hdd >= hdd_min) & (hdd < hdd_max)
            if mask.sum() == 0:
                continue
            
            grp_weights = weights[mask]
            grp_residuals = residuals[mask]
            grp_abs_errors = abs_errors[mask]
            grp_true = y_true[mask]
            
            valid = ~np.isnan(grp_residuals)
            
            weighted_bias = np.sum(grp_weights[valid] * grp_residuals[valid]) / np.sum(grp_weights[valid])
            weighted_mae = np.sum(grp_weights[valid] * grp_abs_errors[valid]) / np.sum(grp_weights[valid])
            weighted_mean_true = np.sum(grp_weights[valid] * grp_true[valid]) / np.sum(grp_weights[valid])
            
            results.append({
                'climate_zone': label,
                'n_samples': mask.sum(),
                'total_weight': grp_weights.sum(),
                'weighted_bias': weighted_bias,
                'weighted_mae': weighted_mae,
                'weighted_mean_true': weighted_mean_true,
                'bias_pct': (weighted_bias / weighted_mean_true * 100) if weighted_mean_true != 0 else np.nan,
                'mae_pct': (weighted_mae / weighted_mean_true * 100) if weighted_mean_true != 0 else np.nan
            })
        
        return pd.DataFrame(results)
