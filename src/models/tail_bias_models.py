"""
Tail Bias Mitigation Models for Heating Demand Prediction

Implements Priority 1 interventions for addressing top-decile underprediction:
1.1 Quantile Regression (q=0.90) - policy-aligned model for upper-tail targeting
1.2 Tail-weighted training - reweighting strategy to emphasize high-consumption cases

These models address the critical limitation of systematic underprediction in the
upper tail, which makes mean-oriented models unsafe for benefit sizing or resource allocation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.base import BaseEstimator, RegressorMixin
import warnings

from src.utils.helpers import logger, Timer, clip_predictions, compute_weighted_quantile


class QuantileLightGBMHeatingModel(BaseEstimator, RegressorMixin):
    """
    LightGBM Quantile Regression model for heating demand prediction.
    
    Uses objective="quantile" with alpha=0.90 (or configurable) to produce
    predictions that better capture the upper tail of the consumption distribution.
    
    Key benefits:
    - Reduces top-decile underprediction
    - More appropriate for policy targeting where underestimation is costly
    - Maintains survey weight support
    
    Note: This model will typically have higher RMSE/MAE than mean-oriented models,
    but that is the expected tradeoff for better tail prediction.
    """
    
    DEFAULT_PARAMS = {
        'objective': 'quantile',
        'alpha': 0.90,  # Default: 90th percentile (upper tail)
        'metric': 'quantile',  # LightGBM quantile metric
        'boosting_type': 'gbdt',
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1
    }
    
    def __init__(self,
                 params: Optional[Dict] = None,
                 quantile: float = 0.90,
                 feature_names: Optional[List[str]] = None,
                 early_stopping_rounds: int = 50):
        """
        Initialize Quantile LightGBM model.
        
        Parameters
        ----------
        params : dict, optional
            LightGBM parameters (merged with defaults)
        quantile : float
            Target quantile (0.90 = 90th percentile)
        feature_names : list, optional
            Feature names
        early_stopping_rounds : int
            Early stopping patience
        """
        self.params = params or {}
        self.quantile = quantile
        self.feature_names = feature_names
        self.early_stopping_rounds = early_stopping_rounds
        
        self.model_ = None
        self.feature_importances_ = None
        self.n_clipped_ = 0
        self.clip_rate_ = 0.0
        
    def _get_params(self) -> Dict:
        """Get merged parameters with quantile objective."""
        merged = self.DEFAULT_PARAMS.copy()
        merged.update(self.params)
        # Ensure quantile objective is set
        merged['objective'] = 'quantile'
        merged['alpha'] = self.quantile
        return merged
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            y: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
            eval_set: Optional[List[Tuple]] = None) -> 'QuantileLightGBMHeatingModel':
        """
        Fit the quantile regression model.
        
        Parameters
        ----------
        X : DataFrame or array
            Features
        y : array
            Target values
        sample_weight : array, optional
            Sample weights
        eval_set : list, optional
            Evaluation sets for early stopping
            
        Returns
        -------
        self
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm is required. Install with: pip install lightgbm")
        
        params = self._get_params()
        
        # Get feature names
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
            X_array = X.values
        else:
            feature_names = self.feature_names or [f'f{i}' for i in range(X.shape[1])]
            X_array = X
        
        # Quantile regression works with positive and zero values
        y_adjusted = np.maximum(y, 0)
        
        def _train_lgbm(local_params: Dict, local_fit_params: Dict):
            """Train LightGBM with GPU->CPU fallback."""
            try:
                model = lgb.LGBMRegressor(**local_params)
                model.fit(X_array, y_adjusted, **local_fit_params)
                return model
            except Exception as e:
                dev = str(local_params.get("device_type", "")).lower()
                if dev == "gpu":
                    logger.warning(f"LightGBM GPU training failed ({e}); falling back to CPU.")
                    cpu_params = dict(local_params)
                    for k in ["device_type", "gpu_platform_id", "gpu_device_id"]:
                        cpu_params.pop(k, None)
                    model = lgb.LGBMRegressor(**cpu_params)
                    model.fit(X_array, y_adjusted, **local_fit_params)
                    return model
                raise
        
        # Fit params
        fit_params = {}
        if sample_weight is not None:
            fit_params['sample_weight'] = sample_weight
        
        if eval_set is not None:
            fit_params['eval_set'] = eval_set
            fit_params['callbacks'] = [lgb.early_stopping(self.early_stopping_rounds, verbose=False)]
        
        self.model_ = _train_lgbm(params, fit_params)
        
        # Store feature importances
        self.feature_importances_ = dict(zip(
            feature_names,
            self.model_.feature_importances_
        ))
        
        logger.info(f"Fitted quantile ({self.quantile}) LightGBM model")
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict heating demand at the specified quantile.
        
        Parameters
        ----------
        X : DataFrame or array
            Features
            
        Returns
        -------
        array
            Predictions (non-negative)
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        predictions = self.model_.predict(X_array)
        
        # Clip negative predictions
        predictions, self.clip_rate_ = clip_predictions(predictions, min_val=0)
        
        if self.clip_rate_ > 0:
            logger.warning(f"Clipped {self.clip_rate_*100:.2f}% of quantile predictions to non-negative")
        
        return predictions
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """Get feature importance."""
        if self.model_ is None:
            raise ValueError("Model not fitted.")
        
        if importance_type == 'gain':
            importance = self.model_.booster_.feature_importance(importance_type='gain')
        else:
            importance = self.model_.feature_importances_
        
        feature_names = (list(self.feature_importances_.keys())
                        if self.feature_importances_ else
                        [f'f{i}' for i in range(len(importance))])
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        df['importance_pct'] = df['importance'] / df['importance'].sum() * 100
        return df


class TailWeightedLightGBMHeatingModel(BaseEstimator, RegressorMixin):
    """
    LightGBM model with tail-weighted training for heating demand prediction.
    
    Implements Priority 1.2: Reweighting strategy to emphasize high-consumption cases.
    
    Training weight formula:
        w_train = survey_weight * tail_weight
        tail_weight = clip((y / median_y) ** alpha, 1, cap)
    
    This approach is more stable than custom objectives and easier to defend to reviewers.
    
    Parameters
    ----------
    tail_weighting_mode : str
        Mode: 'none', 'power', or 'quantile_step'
        - 'none': No additional tail weighting (just survey weights)
        - 'power': w = clip((y/median)^alpha, 1, cap)
        - 'quantile_step': Step function at specified quantile
    tail_alpha : float
        Power parameter for 'power' mode (default 1.0)
    tail_cap : float
        Maximum tail weight multiplier (default 20.0)
    tail_quantile : float
        Quantile threshold for 'quantile_step' mode (default 0.90)
    tail_step_multiplier : float
        Weight multiplier for points above quantile (default 3.0)
    """
    
    DEFAULT_PARAMS = {
        'objective': 'tweedie',
        'tweedie_variance_power': 1.5,
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1
    }
    
    def __init__(self,
                 params: Optional[Dict] = None,
                 tail_weighting_mode: str = 'power',
                 tail_alpha: float = 1.0,
                 tail_cap: float = 20.0,
                 tail_quantile: float = 0.90,
                 tail_step_multiplier: float = 3.0,
                 feature_names: Optional[List[str]] = None,
                 early_stopping_rounds: int = 50,
                 apply_isotonic_calibration: bool = True):
        """
        Initialize tail-weighted LightGBM model.
        
        Parameters
        ----------
        params : dict, optional
            LightGBM parameters (merged with defaults)
        tail_weighting_mode : str
            'none', 'power', or 'quantile_step'
        tail_alpha : float
            Power for tail weighting
        tail_cap : float
            Maximum tail weight
        tail_quantile : float
            Quantile threshold for step mode
        tail_step_multiplier : float
            Multiplier for step mode
        feature_names : list, optional
            Feature names
        early_stopping_rounds : int
            Early stopping patience
        apply_isotonic_calibration : bool
            Whether to apply isotonic calibration after training
        """
        self.params = params or {}
        self.tail_weighting_mode = tail_weighting_mode
        self.tail_alpha = tail_alpha
        self.tail_cap = tail_cap
        self.tail_quantile = tail_quantile
        self.tail_step_multiplier = tail_step_multiplier
        self.feature_names = feature_names
        self.early_stopping_rounds = early_stopping_rounds
        self.apply_isotonic_calibration = apply_isotonic_calibration
        
        self.model_ = None
        self.feature_importances_ = None
        self.isotonic_calibrator_ = None
        self.tail_weight_summary_ = {}
        self.n_clipped_ = 0
        self.clip_rate_ = 0.0
        
    def _get_params(self) -> Dict:
        """Get merged parameters."""
        merged = self.DEFAULT_PARAMS.copy()
        merged.update(self.params)
        return merged
    
    def _compute_tail_weights(self, y: np.ndarray, 
                              base_weights: np.ndarray) -> np.ndarray:
        """
        Compute tail-weighted training weights.
        
        Parameters
        ----------
        y : array
            Target values
        base_weights : array
            Survey weights (NWEIGHT)
            
        Returns
        -------
        array
            Combined weights (survey_weight * tail_weight)
        """
        if self.tail_weighting_mode == 'none':
            self.tail_weight_summary_ = {'mode': 'none'}
            return base_weights.copy()
        
        y_pos = np.maximum(y, 1.0)  # Avoid division by zero
        
        if self.tail_weighting_mode == 'power':
            # Power weighting: w = clip((y/median)^alpha, 1, cap)
            median_y = np.median(y_pos)
            tail_weight = np.power(y_pos / median_y, self.tail_alpha)
            tail_weight = np.clip(tail_weight, 1.0, self.tail_cap)
            
            self.tail_weight_summary_ = {
                'mode': 'power',
                'alpha': self.tail_alpha,
                'cap': self.tail_cap,
                'median_y': float(median_y),
                'max_tail_weight': float(tail_weight.max()),
                'mean_tail_weight': float(tail_weight.mean()),
                'fraction_capped': float(np.mean(tail_weight >= self.tail_cap))
            }
            
        elif self.tail_weighting_mode == 'quantile_step':
            # Step function at quantile threshold
            threshold = np.quantile(y_pos, self.tail_quantile)
            tail_mask = y_pos >= threshold
            tail_weight = np.ones_like(y_pos, dtype=float)
            tail_weight[tail_mask] = self.tail_step_multiplier
            
            self.tail_weight_summary_ = {
                'mode': 'quantile_step',
                'quantile': self.tail_quantile,
                'threshold': float(threshold),
                'multiplier': self.tail_step_multiplier,
                'fraction_upweighted': float(tail_mask.mean())
            }
        else:
            raise ValueError(f"Unknown tail_weighting_mode: {self.tail_weighting_mode}")
        
        return base_weights * tail_weight
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            y: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
            eval_set: Optional[List[Tuple]] = None) -> 'TailWeightedLightGBMHeatingModel':
        """
        Fit the tail-weighted model.
        
        Parameters
        ----------
        X : DataFrame or array
            Features
        y : array
            Target values
        sample_weight : array, optional
            Survey weights (NWEIGHT)
        eval_set : list, optional
            Evaluation sets for early stopping
            
        Returns
        -------
        self
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm is required. Install with: pip install lightgbm")
        
        params = self._get_params()
        
        # Get feature names
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
            X_array = X.values
        else:
            feature_names = self.feature_names or [f'f{i}' for i in range(X.shape[1])]
            X_array = X
        
        # Handle gamma/tweedie with zero values
        y_adjusted = np.maximum(y, 1.0)
        
        # Compute tail-weighted training weights
        base_weights = sample_weight if sample_weight is not None else np.ones(len(y_adjusted))
        train_weights = self._compute_tail_weights(y_adjusted, base_weights)
        
        logger.info(f"Tail weighting: {self.tail_weight_summary_}")
        
        def _train_lgbm(local_params: Dict, local_fit_params: Dict):
            """Train LightGBM with GPU->CPU fallback."""
            try:
                model = lgb.LGBMRegressor(**local_params)
                model.fit(X_array, y_adjusted, **local_fit_params)
                return model
            except Exception as e:
                dev = str(local_params.get("device_type", "")).lower()
                if dev == "gpu":
                    logger.warning(f"LightGBM GPU training failed ({e}); falling back to CPU.")
                    cpu_params = dict(local_params)
                    for k in ["device_type", "gpu_platform_id", "gpu_device_id"]:
                        cpu_params.pop(k, None)
                    model = lgb.LGBMRegressor(**cpu_params)
                    model.fit(X_array, y_adjusted, **local_fit_params)
                    return model
                raise
        
        fit_params = {'sample_weight': train_weights}
        
        if eval_set is not None:
            fit_params['eval_set'] = eval_set
            fit_params['callbacks'] = [lgb.early_stopping(self.early_stopping_rounds, verbose=False)]
        
        self.model_ = _train_lgbm(params, fit_params)
        
        # Store feature importances
        self.feature_importances_ = dict(zip(
            feature_names,
            self.model_.feature_importances_
        ))
        
        # Fit isotonic calibration on training data if requested
        if self.apply_isotonic_calibration:
            try:
                from sklearn.isotonic import IsotonicRegression
                
                # Get OOF predictions for calibration
                y_pred_train = self.model_.predict(X_array)
                
                self.isotonic_calibrator_ = IsotonicRegression(
                    y_min=0, y_max=None, out_of_bounds='clip'
                )
                # Use original survey weights for calibration
                self.isotonic_calibrator_.fit(y_pred_train, y_adjusted, sample_weight=base_weights)
                
                logger.info("Fitted isotonic calibration on training predictions")
            except Exception as e:
                logger.warning(f"Isotonic calibration failed: {e}")
                self.isotonic_calibrator_ = None
        
        logger.info(f"Fitted tail-weighted ({self.tail_weighting_mode}) LightGBM model")
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray],
                apply_calibration: bool = True) -> np.ndarray:
        """
        Predict heating demand.
        
        Parameters
        ----------
        X : DataFrame or array
            Features
        apply_calibration : bool
            Whether to apply isotonic calibration
            
        Returns
        -------
        array
            Predictions (non-negative)
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        predictions = self.model_.predict(X_array)
        
        # Apply isotonic calibration
        if apply_calibration and self.isotonic_calibrator_ is not None:
            predictions = self.isotonic_calibrator_.predict(predictions)
        
        # Clip negative predictions
        predictions, self.clip_rate_ = clip_predictions(predictions, min_val=0)
        
        if self.clip_rate_ > 0:
            logger.warning(f"Clipped {self.clip_rate_*100:.2f}% of predictions to non-negative")
        
        return predictions
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """Get feature importance."""
        if self.model_ is None:
            raise ValueError("Model not fitted.")
        
        if importance_type == 'gain':
            importance = self.model_.booster_.feature_importance(importance_type='gain')
        else:
            importance = self.model_.feature_importances_
        
        feature_names = (list(self.feature_importances_.keys())
                        if self.feature_importances_ else
                        [f'f{i}' for i in range(len(importance))])
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        df['importance_pct'] = df['importance'] / df['importance'].sum() * 100
        return df


class TailBiasMetrics:
    """
    Metrics for evaluating tail bias and policy-targeting performance.
    
    Computes:
    - Top-decile underprediction (must report)
    - Lift@10 (ranking quality)
    - NDCG (ranking quality)
    - Precision/Recall@10 (targeting accuracy)
    
    All metrics stratified by technology group as required.
    """
    
    def __init__(self, k_percentile: float = 10):
        """
        Initialize tail bias metrics.
        
        Parameters
        ----------
        k_percentile : float
            Top percentage to target (default 10 = top 10%)
        """
        self.k_percentile = k_percentile
        self.k_quantile = 1 - k_percentile / 100
    
    def compute_top_decile_underprediction(self,
                                            y_true: np.ndarray,
                                            y_pred: np.ndarray,
                                            weights: np.ndarray) -> Dict[str, float]:
        """
        Compute top-decile underprediction metrics.
        
        This is the critical metric for policy applications.
        
        Parameters
        ----------
        y_true : array
            True consumption values
        y_pred : array
            Predicted values
        weights : array
            Sample weights
            
        Returns
        -------
        dict
            Underprediction metrics
        """
        # Define true top decile
        threshold = compute_weighted_quantile(y_true, weights, self.k_quantile)
        top_decile_mask = y_true >= threshold
        
        if not np.any(top_decile_mask):
            return {
                'top_decile_mean_true': np.nan,
                'top_decile_mean_pred': np.nan,
                'top_decile_bias': np.nan,
                'top_decile_bias_pct': np.nan,
                'top_decile_n': 0
            }
        
        # Weighted means in top decile
        w_top = weights[top_decile_mask]
        y_true_top = y_true[top_decile_mask]
        y_pred_top = y_pred[top_decile_mask]
        
        mean_true = np.sum(w_top * y_true_top) / np.sum(w_top)
        mean_pred = np.sum(w_top * y_pred_top) / np.sum(w_top)
        
        bias = mean_pred - mean_true  # Negative = underprediction
        bias_pct = bias / mean_true * 100
        
        # Also compute RMSE and MAE in top decile
        rmse = np.sqrt(np.sum(w_top * (y_pred_top - y_true_top)**2) / np.sum(w_top))
        mae = np.sum(w_top * np.abs(y_pred_top - y_true_top)) / np.sum(w_top)
        
        return {
            'top_decile_mean_true': float(mean_true),
            'top_decile_mean_pred': float(mean_pred),
            'top_decile_bias': float(bias),
            'top_decile_bias_pct': float(bias_pct),
            'top_decile_rmse': float(rmse),
            'top_decile_mae': float(mae),
            'top_decile_n': int(top_decile_mask.sum()),
            'top_decile_weight_sum': float(np.sum(w_top)),
            'top_decile_threshold': float(threshold)
        }
    
    def compute_lift_at_k(self,
                          y_true: np.ndarray,
                          y_pred: np.ndarray,
                          weights: np.ndarray) -> Dict[str, float]:
        """
        Compute Lift@k for targeting quality.
        
        Lift@k = (precision@k) / (base_rate)
        where base_rate = k/100
        
        Parameters
        ----------
        y_true : array
            True consumption values
        y_pred : array
            Predicted values
        weights : array
            Sample weights
            
        Returns
        -------
        dict
            Lift metrics
        """
        # Define true and predicted top-k
        true_threshold = compute_weighted_quantile(y_true, weights, self.k_quantile)
        pred_threshold = compute_weighted_quantile(y_pred, weights, self.k_quantile)
        
        true_top = y_true >= true_threshold
        pred_top = y_pred >= pred_threshold
        
        # Precision: of predicted top, fraction truly high
        w_pred_top = np.sum(weights[pred_top])
        w_both = np.sum(weights[pred_top & true_top])
        
        precision = w_both / w_pred_top if w_pred_top > 0 else 0
        
        # Recall: of truly high, fraction predicted high
        w_true_top = np.sum(weights[true_top])
        recall = w_both / w_true_top if w_true_top > 0 else 0
        
        # Base rate
        base_rate = self.k_percentile / 100
        
        # Lift
        lift = precision / base_rate if base_rate > 0 else 0
        
        # F1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Jaccard
        w_union = np.sum(weights[pred_top | true_top])
        jaccard = w_both / w_union if w_union > 0 else 0
        
        return {
            'precision_at_k': float(precision),
            'recall_at_k': float(recall),
            'f1_at_k': float(f1),
            'jaccard_at_k': float(jaccard),
            'lift_at_k': float(lift),
            'base_rate': float(base_rate)
        }
    
    def compute_ndcg(self,
                     y_true: np.ndarray,
                     y_pred: np.ndarray,
                     weights: np.ndarray,
                     k: Optional[int] = None) -> float:
        """
        Compute Normalized Discounted Cumulative Gain.
        
        Measures ranking quality: do high true values rank high in predictions?
        
        Parameters
        ----------
        y_true : array
            True values
        y_pred : array
            Predicted values
        weights : array
            Sample weights
        k : int, optional
            Only consider top-k positions
            
        Returns
        -------
        float
            NDCG score (0-1)
        """
        n = len(y_true)
        k = k or int(n * self.k_percentile / 100)
        
        # Rank by predictions (descending)
        pred_order = np.argsort(-y_pred)[:k]
        ideal_order = np.argsort(-y_true)[:k]
        
        # DCG
        dcg = 0
        for i, idx in enumerate(pred_order):
            rel = y_true[idx] * weights[idx]
            dcg += rel / np.log2(i + 2)
        
        # Ideal DCG
        idcg = 0
        for i, idx in enumerate(ideal_order):
            rel = y_true[idx] * weights[idx]
            idcg += rel / np.log2(i + 2)
        
        return float(dcg / idcg) if idcg > 0 else 0.0
    
    def compute_all_metrics(self,
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            weights: np.ndarray) -> Dict[str, float]:
        """
        Compute all tail bias metrics.
        
        Parameters
        ----------
        y_true : array
            True consumption values
        y_pred : array
            Predicted values
        weights : array
            Sample weights
            
        Returns
        -------
        dict
            All metrics
        """
        results = {}
        
        # Top-decile underprediction (critical metric)
        underpred = self.compute_top_decile_underprediction(y_true, y_pred, weights)
        results.update(underpred)
        
        # Targeting metrics
        lift_metrics = self.compute_lift_at_k(y_true, y_pred, weights)
        results.update(lift_metrics)
        
        # Ranking quality
        results['ndcg'] = self.compute_ndcg(y_true, y_pred, weights)
        
        return results
    
    def compute_metrics_by_tech_group(self,
                                       y_true: np.ndarray,
                                       y_pred: np.ndarray,
                                       weights: np.ndarray,
                                       tech_group: np.ndarray) -> pd.DataFrame:
        """
        Compute tail bias metrics stratified by technology group.
        
        This is required for reviewer reporting as tail bias is particularly
        severe for some groups (e.g., combustion/hybrid).
        
        Parameters
        ----------
        y_true : array
            True consumption values
        y_pred : array
            Predicted values
        weights : array
            Sample weights
        tech_group : array
            Technology group labels
            
        Returns
        -------
        DataFrame
            Metrics by technology group
        """
        results = []
        
        for group in np.unique(tech_group):
            mask = tech_group == group
            if mask.sum() < 10:
                continue
            
            metrics = self.compute_all_metrics(
                y_true[mask],
                y_pred[mask],
                weights[mask]
            )
            metrics['tech_group'] = group
            metrics['n_samples'] = int(mask.sum())
            metrics['weight_sum'] = float(weights[mask].sum())
            
            results.append(metrics)
        
        df = pd.DataFrame(results)
        
        # Reorder columns
        first_cols = ['tech_group', 'n_samples', 'weight_sum', 
                      'top_decile_bias_pct', 'lift_at_k', 'ndcg',
                      'precision_at_k', 'recall_at_k']
        other_cols = [c for c in df.columns if c not in first_cols]
        df = df[first_cols + other_cols]
        
        return df


def compare_quantile_vs_mean_models(y_true: np.ndarray,
                                     y_pred_mean: np.ndarray,
                                     y_pred_q90: np.ndarray,
                                     weights: np.ndarray,
                                     tech_group: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Compare quantile regression vs mean-oriented model predictions.
    
    Produces the required tradeoff table showing:
    - Quantile model may worsen RMSE/MAE
    - Quantile model should improve top-10% underprediction
    - Targeting metrics (Lift@10, NDCG) comparison
    
    Parameters
    ----------
    y_true : array
        True consumption values
    y_pred_mean : array
        Mean-model predictions
    y_pred_q90 : array
        Quantile (q=0.90) model predictions
    weights : array
        Sample weights
    tech_group : array, optional
        Technology groups for stratified analysis
        
    Returns
    -------
    DataFrame
        Comparison table
    """
    from src.evaluation.metrics import WeightedMetrics
    
    wm = WeightedMetrics()
    tbm = TailBiasMetrics(k_percentile=10)
    
    rows = []
    
    # Overall comparison
    for model_name, y_pred in [('Mean (Tweedie)', y_pred_mean), 
                                ('Quantile (q=0.90)', y_pred_q90)]:
        # Standard metrics
        std_metrics = wm.compute_all_metrics(y_true, y_pred, weights)
        
        # Tail bias metrics
        tail_metrics = tbm.compute_all_metrics(y_true, y_pred, weights)
        
        row = {
            'Model': model_name,
            'tech_group': 'Overall',
            'wRMSE': std_metrics['weighted_rmse'],
            'wMAE': std_metrics['weighted_mae'],
            'wR²': std_metrics['weighted_r2'],
            'wBias': std_metrics['weighted_bias'],
            'Top-10% Bias (%)': tail_metrics['top_decile_bias_pct'],
            'Lift@10': tail_metrics['lift_at_k'],
            'NDCG': tail_metrics['ndcg'],
            'Precision@10': tail_metrics['precision_at_k'],
            'Recall@10': tail_metrics['recall_at_k'],
            'n': len(y_true)
        }
        rows.append(row)
    
    # By technology group if provided
    if tech_group is not None:
        for group in np.unique(tech_group):
            mask = tech_group == group
            if mask.sum() < 50:
                continue
            
            for model_name, y_pred in [('Mean (Tweedie)', y_pred_mean),
                                        ('Quantile (q=0.90)', y_pred_q90)]:
                std_metrics = wm.compute_all_metrics(
                    y_true[mask], y_pred[mask], weights[mask]
                )
                tail_metrics = tbm.compute_all_metrics(
                    y_true[mask], y_pred[mask], weights[mask]
                )
                
                row = {
                    'Model': model_name,
                    'tech_group': group,
                    'wRMSE': std_metrics['weighted_rmse'],
                    'wMAE': std_metrics['weighted_mae'],
                    'wR²': std_metrics['weighted_r2'],
                    'wBias': std_metrics['weighted_bias'],
                    'Top-10% Bias (%)': tail_metrics['top_decile_bias_pct'],
                    'Lift@10': tail_metrics['lift_at_k'],
                    'NDCG': tail_metrics['ndcg'],
                    'Precision@10': tail_metrics['precision_at_k'],
                    'Recall@10': tail_metrics['recall_at_k'],
                    'n': int(mask.sum())
                }
                rows.append(row)
    
    df = pd.DataFrame(rows)
    df.attrs['note'] = (
        "Quantile model (q=0.90) trades off RMSE/MAE for better top-decile prediction. "
        "Top-10% Bias: 0 = perfect, negative = underprediction. "
        "Lift@10: higher is better (>1 = better than random). "
        "NDCG: 1.0 = perfect ranking."
    )
    
    return df
