"""
Main Models for Heating Demand Prediction

Implements Section 6.2: Main models
- LightGBM (primary): supports sample weights, robust, fast
- EBM (interpretability): shape functions for physical sanity + policy narratives

Section 6.3: Output non-negativity
- Gamma/Tweedie objective for non-negative predictions
- Post-processing clamp at 0 with reporting

CALIBRATION METHODOLOGY (leakage-free):
----------------------------------------
The isotonic calibration is fitted INSIDE each outer fold of nested CV:
1. For each outer fold, split data into train/test
2. Fit LightGBM on training data
3. Fit isotonic calibrator on TRAINING predictions vs training targets
4. Apply calibrator when predicting on TEST data
This ensures calibration does not leak test information.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.base import BaseEstimator, RegressorMixin, clone
import warnings

from src.utils.helpers import logger, Timer, clip_predictions


class LightGBMHeatingModel(BaseEstimator, RegressorMixin):
    """
    LightGBM model for heating demand prediction.
    
    Features:
    - Gamma objective for non-negative predictions
    - Sample weight support
    - Optional monotonic constraints
    - Hyperparameter configuration
    """
    
    # Default hyperparameters
    # Using Tweedie with power=1.5 (between Poisson=1 and Gamma=2)
    # Better for right-skewed data with many moderate values
    DEFAULT_PARAMS = {
        'objective': 'tweedie',  # Tweedie: handles zero-inflated + heavy tails
        'tweedie_variance_power': 1.5,  # 1=Poisson, 2=Gamma, 1.5=compound
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
    
    # Monotonic constraints for physical plausibility
    # 1 = increasing, -1 = decreasing, 0 = no constraint
    MONOTONIC_CONSTRAINTS_MAP = {
        'HDD65': 1,           # More HDD -> More energy
        'HDD30YR_PUB': 1,     # More HDD -> More energy
        'TOTSQFT_EN': 1,      # Larger area -> More energy
        'TOTHSQFT': 1,        # Larger heated area -> More energy
        # Note: ADQINSUL requires knowing its encoding direction
    }
    
    def __init__(self, 
                 params: Optional[Dict] = None,
                 use_monotonic_constraints: bool = False,
                 feature_names: Optional[List[str]] = None,
                 apply_bias_correction: bool = True,
                 # Calibration controls
                 calibration_strategy: str = 'oof',
                 calibration_folds: int = 3,
                 calibration_random_state: int = 42,
                 calibration_by_hdd_bin: bool = True,
                 calibration_min_bin_samples: int = 200,
                 calibration_fit_intercept: bool = False,
                 calibration_scale_clip: Tuple[float, float] = (0.2, 5.0),
                 calibration_tail_quantile: float = 0.90,
                 calibration_tail_weight_multiplier: float = 2.0,
                 isotonic_enabled: bool = True,
                 isotonic_min_samples: int = 800,
                 isotonic_min_improvement: float = 0.0,
                 # Training reweighting (debiasing)
                 training_reweighting_enabled: bool = False,
                 training_tail_quantile: float = 0.95,
                 training_tail_multiplier: float = 1.5,
                 training_cold_hdd_quantile: float = 0.85,
                 training_cold_multiplier: float = 1.25,
                 training_two_pass_focal: bool = False,
                 training_focal_gamma: float = 2.0,
                 early_stopping_rounds: int = 50):
        """
        Initialize LightGBM heating model.
        
        Parameters
        ----------
        params : dict, optional
            LightGBM parameters (merged with defaults)
        use_monotonic_constraints : bool
            Whether to apply monotonic constraints
        feature_names : list, optional
            Feature names for constraint mapping
        apply_bias_correction : bool
            Whether to apply post-hoc bias correction (reduces systematic under/over-prediction)
        """
        self.params = params or {}
        self.use_monotonic_constraints = use_monotonic_constraints
        self.feature_names = feature_names
        self.apply_bias_correction = apply_bias_correction

        # Calibration settings
        self.calibration_strategy = calibration_strategy
        self.calibration_folds = calibration_folds
        self.calibration_random_state = calibration_random_state
        self.calibration_by_hdd_bin = calibration_by_hdd_bin
        self.calibration_min_bin_samples = calibration_min_bin_samples
        self.calibration_fit_intercept = calibration_fit_intercept
        self.calibration_scale_clip = calibration_scale_clip
        self.calibration_tail_quantile = calibration_tail_quantile
        self.calibration_tail_weight_multiplier = calibration_tail_weight_multiplier
        self.isotonic_enabled = isotonic_enabled
        self.isotonic_min_samples = isotonic_min_samples
        self.isotonic_min_improvement = isotonic_min_improvement

        # Training reweighting (optional debiasing / tail emphasis)
        self.training_reweighting_enabled = training_reweighting_enabled
        self.training_tail_quantile = training_tail_quantile
        self.training_tail_multiplier = training_tail_multiplier
        self.training_cold_hdd_quantile = training_cold_hdd_quantile
        self.training_cold_multiplier = training_cold_multiplier
        self.training_two_pass_focal = training_two_pass_focal
        self.training_focal_gamma = training_focal_gamma
        self.early_stopping_rounds = int(early_stopping_rounds)
        self.training_weight_summary_ = {}
        self.focal_reweighting_summary_ = {}

        # Learned calibration models:
        # - global calibration always present
        # - optionally per-HDD-bin calibration (Tech×HDD-bin emerges naturally when using split models)
        self.calibration_models_ = None

        self.model_ = None
        self.feature_importances_ = None
        self.n_clipped_ = 0
        self.clip_rate_ = 0.0
        self.bias_correction_ = 0.0  # Additive correction
        self.scale_correction_ = 1.0  # Multiplicative correction
        self.isotonic_calibrator_ = None  # For tail calibration
        
    def _get_params(self) -> Dict:
        """Get merged parameters."""
        merged = self.DEFAULT_PARAMS.copy()
        merged.update(self.params)
        return merged
    
    def _build_monotonic_constraints(self, feature_names: List[str]) -> List[int]:
        """Build monotonic constraint list for features."""
        constraints = []
        for feat in feature_names:
            if feat in self.MONOTONIC_CONSTRAINTS_MAP:
                constraints.append(self.MONOTONIC_CONSTRAINTS_MAP[feat])
            else:
                constraints.append(0)  # No constraint
        return constraints
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], 
            y: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
            eval_set: Optional[List[Tuple]] = None) -> 'LightGBMHeatingModel':
        """
        Fit the LightGBM model.
        
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
        
        # Apply monotonic constraints if requested
        if self.use_monotonic_constraints:
            constraints = self._build_monotonic_constraints(feature_names)
            params['monotone_constraints'] = constraints
            logger.info(f"Applied monotonic constraints to {sum(c != 0 for c in constraints)} features")
        
        # Handle gamma objective with zero values
        # Gamma requires y > 0, so we need to handle zeros
        y_adjusted = np.maximum(y, 1.0)  # Minimum of 1 BTU
        
        # Create model
        # Optional training reweighting to emphasize tails / cold climates (debiasing utility)
        sw = np.asarray(sample_weight) if sample_weight is not None else None
        if self.training_reweighting_enabled:
            w_adj = sw.copy() if sw is not None else np.ones(len(y_adjusted), dtype=float)

            # Tail emphasis (right tail)
            tq = float(self.training_tail_quantile)
            if 0.0 < tq < 1.0:
                tail_thr = float(np.quantile(y_adjusted, tq))
                tail_mask = y_adjusted >= tail_thr
                w_adj[tail_mask] *= float(self.training_tail_multiplier)
            else:
                tail_thr = None
                tail_mask = np.zeros(len(y_adjusted), dtype=bool)

            # Cold-climate emphasis (high HDD)
            hdd_vals = self._get_hdd_from_X(X, feature_names)
            if hdd_vals is not None:
                cq = float(self.training_cold_hdd_quantile)
                if 0.0 < cq < 1.0:
                    cold_thr = float(np.quantile(hdd_vals, cq))
                    cold_mask = np.asarray(hdd_vals) >= cold_thr
                    w_adj[cold_mask] *= float(self.training_cold_multiplier)
                else:
                    cold_thr = None
                    cold_mask = np.zeros(len(y_adjusted), dtype=bool)
            else:
                cold_thr = None
                cold_mask = np.zeros(len(y_adjusted), dtype=bool)

            self.training_weight_summary_ = {
                "tail_quantile": tq,
                "tail_threshold": tail_thr,
                "tail_multiplier": float(self.training_tail_multiplier),
                "tail_fraction": float(np.mean(tail_mask)) if tail_thr is not None else 0.0,
                "cold_hdd_quantile": float(self.training_cold_hdd_quantile),
                "cold_hdd_threshold": cold_thr,
                "cold_multiplier": float(self.training_cold_multiplier),
                "cold_fraction": float(np.mean(cold_mask)) if cold_thr is not None else 0.0
            }

            sw = w_adj

        def _train_lgbm(local_params: Dict, local_fit_params: Dict):
            """Train LightGBM, with automatic GPU->CPU fallback if GPU is unavailable."""
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

        # Fit with or without sample weights
        fit_params = {}
        if sw is not None:
            fit_params['sample_weight'] = sw

        if eval_set is not None:
            fit_params['eval_set'] = eval_set
            fit_params['callbacks'] = [lgb.early_stopping(self.early_stopping_rounds, verbose=False)]

        self.model_ = _train_lgbm(params, fit_params)

        # Two-pass focal-like reweighting (optional): upweight hard-to-fit points based on first-pass residuals.
        # This is a pragmatic alternative to custom focal losses for gradient boosting.
        if self.training_two_pass_focal:
            try:
                preds_pass1 = self.model_.predict(X_array)
                rel_err = np.abs(y_adjusted - preds_pass1) / np.maximum(y_adjusted, 1.0)
                gamma = float(self.training_focal_gamma)

                w_base = sw if sw is not None else np.ones(len(y_adjusted), dtype=float)
                w2 = w_base * np.power(1.0 + rel_err, gamma)

                # Normalize to preserve total weight scale
                denom = np.sum(w2)
                if denom > 0:
                    w2 *= (np.sum(w_base) / denom)

                self.focal_reweighting_summary_ = {
                    "gamma": gamma,
                    "rel_err_mean": float(np.mean(rel_err)),
                    "rel_err_p90": float(np.quantile(rel_err, 0.90)),
                    "weight_multiplier_mean": float(np.mean(w2 / np.maximum(w_base, 1e-12))),
                    "weight_multiplier_p90": float(np.quantile(w2 / np.maximum(w_base, 1e-12), 0.90))
                }

                # Refit model with reweighted samples
                fit_params_pass2 = dict(fit_params)
                fit_params_pass2['sample_weight'] = w2
                self.model_ = _train_lgbm(params, fit_params_pass2)

            except Exception as e:
                logger.warning(f"Two-pass focal reweighting failed; continuing with first-pass model: {e}")

        
        # Store feature importances
        self.feature_importances_ = dict(zip(
            feature_names, 
            self.model_.feature_importances_
        ))
        
        # Compute calibration on training data (leakage-aware)
        if self.apply_bias_correction:
            # Prefer out-of-fold (OOF) predictions for calibrator fitting to reduce overfitting
            hdd_values = self._get_hdd_from_X(X, feature_names)
            y_pred_for_calib = None

            if (self.calibration_strategy == 'oof' and self.calibration_folds is not None 
                    and int(self.calibration_folds) > 1):
                try:
                    from sklearn.model_selection import KFold

                    y_pred_for_calib = np.zeros_like(y_adjusted, dtype=float)
                    kf = KFold(n_splits=int(self.calibration_folds),
                               shuffle=True,
                               random_state=int(self.calibration_random_state))

                    for tr_idx, va_idx in kf.split(X_array):
                        # Use conservative CPU params for OOF calibration to avoid GPU / split edge-case crashes on Windows
                        params_cal = dict(params)
                        params_cal.pop('device', None)
                        params_cal.pop('device_type', None)
                        params_cal.pop('gpu_platform_id', None)
                        params_cal.pop('gpu_device_id', None)
                        params_cal['device_type'] = 'cpu'
                        params_cal['force_col_wise'] = True
                        params_cal['verbosity'] = -1
                        # Keep calibration fast/stable: cap trees; early stopping handles the rest
                        if 'n_estimators' in params_cal:
                            params_cal['n_estimators'] = int(min(int(params_cal['n_estimators']), 250))
                        if 'min_child_samples' in params_cal:
                            params_cal['min_child_samples'] = int(min(int(params_cal['min_child_samples']), 50))
                        fold_model = lgb.LGBMRegressor(**params_cal)

                        fit_params_fold = {}
                        if sample_weight is not None:
                            fit_params_fold['sample_weight'] = np.asarray(sample_weight)[tr_idx]

                        # Early stopping on validation fold to avoid unnecessary trees
                        fit_params_fold['eval_set'] = [(X_array[va_idx], y_adjusted[va_idx])]
                        fit_params_fold['callbacks'] = [lgb.early_stopping(self.early_stopping_rounds, verbose=False)]

                        fold_model.fit(X_array[tr_idx], y_adjusted[tr_idx], **fit_params_fold)
                        y_pred_for_calib[va_idx] = fold_model.predict(X_array[va_idx])
                except Exception as e:
                    logger.warning(f"OOF calibration failed; falling back to in-sample calibration: {e}")
                    y_pred_for_calib = self.model_.predict(X_array)
            else:
                y_pred_for_calib = self.model_.predict(X_array)

            # Build calibration weights (optionally tail-aware)
            w_cal = self._make_calibration_weights(y, sample_weight)

            # Fit global calibrator + optional per-HDD-bin calibrators
            self.calibration_models_ = {}

            # Global
            bias_g, scale_g = self._fit_linear_calibrator(y_pred_for_calib, y, w_cal)
            y_lin_g = y_pred_for_calib * scale_g + bias_g
            iso_g = self._fit_isotonic_if_better(y_lin_g, y, w_cal)

            self.calibration_models_['__global__'] = {
                'bias': bias_g,
                'scale': scale_g,
                'isotonic': iso_g
            }

            # Per HDD bin (Tech×HDD-bin happens naturally when using split models)
            if self.calibration_by_hdd_bin and hdd_values is not None:
                bins = self._hdd_to_bins(hdd_values)
                for b in self._HDD_BIN_LABELS:
                    mask = (bins == b)
                    if int(mask.sum()) < int(self.calibration_min_bin_samples):
                        continue

                    bias_b, scale_b = self._fit_linear_calibrator(
                        y_pred_for_calib[mask], y[mask], w_cal[mask]
                    )
                    y_lin_b = y_pred_for_calib[mask] * scale_b + bias_b
                    iso_b = self._fit_isotonic_if_better(y_lin_b, y[mask], w_cal[mask])

                    self.calibration_models_[b] = {
                        'bias': bias_b,
                        'scale': scale_b,
                        'isotonic': iso_b
                    }

            # Backward-compatible attributes (global)
            self.bias_correction_ = float(bias_g)
            self.scale_correction_ = float(scale_g)
            self.isotonic_calibrator_ = iso_g

            logger.debug(
                f"Calibration fitted: global bias={self.bias_correction_:.2f}, "
                f"scale={self.scale_correction_:.3f}; "
                f"per-bin={max(0, len(self.calibration_models_) - 1)}"
            )
        
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray], 
                apply_correction: bool = True) -> np.ndarray:
        """
        Predict heating demand.
        
        Parameters
        ----------
        X : DataFrame or array
            Features
        apply_correction : bool
            Whether to apply bias/scale correction
            
        Returns
        -------
        array
            Predictions (non-negative)
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        hdd_values = None
        if isinstance(X, pd.DataFrame):
            hdd_values = X['HDD65'].values if 'HDD65' in X.columns else None
            X_array = X.values
        else:
            X_array = X
            # Try to extract HDD from array using provided feature names
            try:
                feature_names = self.feature_names or []
                if feature_names and 'HDD65' in feature_names:
                    hdd_values = np.asarray(X_array)[:, feature_names.index('HDD65')]
            except Exception:
                hdd_values = None

        predictions = self.model_.predict(X_array)

        # Apply calibration (global + optional per-HDD-bin)
        if apply_correction and self.apply_bias_correction:
            calib = self.calibration_models_ or {}

            def _apply_cal(pred: np.ndarray, cdict: Dict[str, Any]) -> np.ndarray:
                out = pred * float(cdict.get('scale', 1.0)) + float(cdict.get('bias', 0.0))
                iso = cdict.get('isotonic', None)
                if iso is not None:
                    out = iso.predict(out)
                return out

            if (self.calibration_by_hdd_bin and hdd_values is not None and '__global__' in calib):
                bins = self._hdd_to_bins(hdd_values)
                corrected = np.empty_like(predictions, dtype=float)
                global_c = calib['__global__']
                for b in np.unique(bins):
                    mask = (bins == b)
                    c = calib.get(b, global_c)
                    corrected[mask] = _apply_cal(predictions[mask], c)
                predictions = corrected
            elif '__global__' in calib:
                predictions = _apply_cal(predictions, calib['__global__'])
            else:
                # Fallback to legacy single calibrator
                predictions = predictions * self.scale_correction_ + self.bias_correction_
                if self.isotonic_calibrator_ is not None:
                    predictions = self.isotonic_calibrator_.predict(predictions)
# Clip negative predictions (shouldn't happen with gamma but safety check)
        predictions, self.clip_rate_ = clip_predictions(predictions, min_val=0)
        
        if self.clip_rate_ > 0:
            logger.warning(f"Clipped {self.clip_rate_*100:.2f}% of predictions to non-negative")
        
        return predictions
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance.
        
        Parameters
        ----------
        importance_type : str
            Type of importance ('gain', 'split')
            
        Returns
        -------
        DataFrame
            Feature importance table
        """
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

    # --- Calibration helpers ---
    _HDD_BINS = np.array([0.0, 2000.0, 4000.0, 6000.0, 8000.0, np.inf], dtype=float)
    _HDD_BIN_LABELS = ['very_mild', 'mild', 'moderate', 'cold', 'very_cold']

    def _hdd_to_bins(self, hdd: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Map HDD values to categorical bin labels."""
        if hdd is None:
            return None
        hdd = np.asarray(hdd, dtype=float)
        out = np.full(hdd.shape, 'unknown', dtype=object)
        mask = np.isfinite(hdd)
        if mask.sum() == 0:
            return out
        # digitize returns indices 1..len(bins)-1
        idx = np.digitize(hdd[mask], self._HDD_BINS, right=False) - 1
        idx = np.clip(idx, 0, len(self._HDD_BIN_LABELS) - 1)
        out[mask] = np.array(self._HDD_BIN_LABELS, dtype=object)[idx]
        return out

    def _get_hdd_from_X(self, X: Union[pd.DataFrame, np.ndarray], feature_names: List[str]) -> Optional[np.ndarray]:
        """Extract HDD65 from X if available."""
        if isinstance(X, pd.DataFrame):
            return X['HDD65'].values if 'HDD65' in X.columns else None
        if feature_names is not None and 'HDD65' in feature_names:
            try:
                j = feature_names.index('HDD65')
                return np.asarray(X)[:, j]
            except Exception:
                return None
        return None

    def _make_calibration_weights(self,
                                 y_true: np.ndarray,
                                 sample_weight: Optional[np.ndarray]) -> np.ndarray:
        """Build weights for calibration (optionally tail-aware)."""
        if sample_weight is None:
            w = np.ones_like(y_true, dtype=float)
        else:
            w = np.asarray(sample_weight, dtype=float).copy()

        # Tail upweighting (policy-relevant high-use homes)
        try:
            q = float(np.nanquantile(y_true, self.calibration_tail_quantile))
            tail_mask = y_true >= q
            if np.any(tail_mask) and self.calibration_tail_weight_multiplier is not None:
                w[tail_mask] *= float(self.calibration_tail_weight_multiplier)
        except Exception:
            pass

        # Avoid zero weights
        w = np.maximum(w, 1e-12)
        return w

    def _fit_linear_calibrator(self,
                               y_pred: np.ndarray,
                               y_true: np.ndarray,
                               w: np.ndarray) -> Tuple[float, float]:
        """Fit a weighted linear calibrator y ≈ bias + scale*y_pred."""
        y_pred = np.asarray(y_pred, dtype=float)
        y_true = np.asarray(y_true, dtype=float)
        w = np.asarray(w, dtype=float)

        # Through-origin calibration (scale-only) is more stable for heavy tails and avoids large intercepts.
        if not self.calibration_fit_intercept:
            den = np.sum(w * (y_pred ** 2))
            if den <= 1e-12:
                scale = 1.0
            else:
                scale = float(np.sum(w * y_pred * y_true) / den)
            bias = 0.0
        else:
            wsum = float(np.sum(w))
            mx = float(np.sum(w * y_pred) / wsum)
            my = float(np.sum(w * y_true) / wsum)
            x0 = y_pred - mx
            y0 = y_true - my
            den = float(np.sum(w * x0 * x0))
            if den <= 1e-12:
                scale = 1.0
            else:
                scale = float(np.sum(w * x0 * y0) / den)
            bias = float(my - scale * mx)

        lo, hi = self.calibration_scale_clip
        scale = float(np.clip(scale, lo, hi))
        return bias, scale

    @staticmethod
    def _weighted_rmse(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        w = np.asarray(w, dtype=float)
        return float(np.sqrt(np.sum(w * (y_true - y_pred) ** 2) / np.sum(w)))

    def _fit_isotonic_if_better(self,
                                y_pred_linear: np.ndarray,
                                y_true: np.ndarray,
                                w: np.ndarray):
        """Fit isotonic regression only if it improves weighted RMSE."""
        if not self.isotonic_enabled:
            return None
        if y_pred_linear.shape[0] < self.isotonic_min_samples:
            return None

        try:
            from sklearn.isotonic import IsotonicRegression
            base_rmse = self._weighted_rmse(y_true, y_pred_linear, w)

            iso = IsotonicRegression(y_min=0, y_max=None, out_of_bounds='clip')
            iso.fit(y_pred_linear, y_true, sample_weight=w)

            y_iso = iso.predict(y_pred_linear)
            iso_rmse = self._weighted_rmse(y_true, y_iso, w)

            if iso_rmse <= (base_rmse - float(self.isotonic_min_improvement)):
                return iso
            return None
        except Exception as e:
            logger.warning(f"Isotonic calibration failed: {e}")
            return None



class EBMHeatingModel(BaseEstimator, RegressorMixin):
    """
    Explainable Boosting Machine for heating demand prediction.
    
    Features:
    - Interpretable shape functions
    - Physical plausibility verification via shapes
    - Sample weight support
    """
    
    DEFAULT_PARAMS = {
        'max_bins': 256,
        'interactions': 10,
        'outer_bags': 8,
        'inner_bags': 0,
        'learning_rate': 0.01,
        'validation_size': 0.15,
        'early_stopping_rounds': 50,
        'max_rounds': 10000,
        'random_state': 42
    }
    
    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize EBM heating model.
        
        Parameters
        ----------
        params : dict, optional
            EBM parameters (merged with defaults)
        """
        self.params = params or {}
        self.model_ = None
        self.feature_names_ = None
        
    def _get_params(self) -> Dict:
        """Get merged parameters."""
        merged = self.DEFAULT_PARAMS.copy()
        merged.update(self.params)
        return merged
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            y: np.ndarray,
            sample_weight: Optional[np.ndarray] = None) -> 'EBMHeatingModel':
        """
        Fit the EBM model.
        
        Parameters
        ----------
        X : DataFrame or array
            Features
        y : array
            Target values
        sample_weight : array, optional
            Sample weights
            
        Returns
        -------
        self
        """
        try:
            from interpret.glassbox import ExplainableBoostingRegressor
        except ImportError:
            raise ImportError("interpret is required. Install with: pip install interpret")
        
        params = self._get_params()
        
        # Get feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_array = X.values
        else:
            self.feature_names_ = [f'f{i}' for i in range(X.shape[1])]
            X_array = X
        
        # Create and fit model
        self.model_ = ExplainableBoostingRegressor(
            feature_names=self.feature_names_,
            **params
        )
        
        # Handle sample weights
        if sample_weight is not None:
            self.model_.fit(X_array, y, sample_weight=sample_weight)
        else:
            self.model_.fit(X_array, y)
        
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict heating demand.
        
        Parameters
        ----------
        X : DataFrame or array
            Features
            
        Returns
        -------
        array
            Predictions
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        predictions = self.model_.predict(X)
        
        # Clip negative predictions
        predictions, clip_rate = clip_predictions(predictions, min_val=0)
        
        if clip_rate > 0:
            logger.warning(f"Clipped {clip_rate*100:.2f}% of EBM predictions to non-negative")
        
        return predictions
    
    def get_shape_functions(self) -> Dict[str, Any]:
        """
        Get shape functions for interpretation.
        
        Returns
        -------
        dict
            Dictionary with feature shapes
        """
        if self.model_ is None:
            raise ValueError("Model not fitted.")
        
        shapes = {}
        
        # Get global explanations
        global_exp = self.model_.explain_global()
        
        for i, name in enumerate(global_exp.data()['names']):
            if i < len(self.model_.term_features_):
                term_idx = i
                
                shapes[name] = {
                    'bins': self.model_.bins_[term_idx] if term_idx < len(self.model_.bins_) else None,
                    'scores': self.model_.term_scores_[term_idx] if term_idx < len(self.model_.term_scores_) else None,
                }
        
        return shapes
    
    def verify_hdd_direction(self) -> Dict[str, Any]:
        """
        Verify that HDD shape function has correct direction.
        
        For heating, HDD should have positive effect on energy.
        
        Returns
        -------
        dict
            Verification results
        """
        if self.model_ is None:
            raise ValueError("Model not fitted.")
        
        result = {
            'hdd_found': False,
            'correct_direction': None,
            'correlation': None
        }
        
        # Find HDD feature
        for i, name in enumerate(self.feature_names_):
            if 'HDD' in name.upper():
                result['hdd_found'] = True
                
                if i < len(self.model_.term_scores_):
                    scores = self.model_.term_scores_[i]
                    
                    # Check if scores generally increase
                    if len(scores) > 1:
                        # Compute correlation with index (should be positive)
                        indices = np.arange(len(scores))
                        valid = ~np.isnan(scores)
                        if valid.sum() > 1:
                            corr = np.corrcoef(indices[valid], scores[valid])[0, 1]
                            result['correlation'] = corr
                            result['correct_direction'] = corr > 0
                
                break
        
        return result


class HeatingDemandModels:
    """
    Container for all heating demand models.
    
    Manages:
    - Technology-specific models
    - Monolithic comparison model
    - Model selection and evaluation
    """
    
    def __init__(self, 
                 model_type: str = 'lightgbm',
                 use_monotonic_constraints: bool = False,
                 params: Optional[Dict] = None):
        """
        Initialize heating demand models.
        
        Parameters
        ----------
        model_type : str
            Model type: 'lightgbm' or 'ebm'
        use_monotonic_constraints : bool
            Whether to use monotonic constraints (LightGBM only)
        params : dict, optional
            Model parameters
        """
        self.model_type = model_type
        self.use_monotonic_constraints = use_monotonic_constraints
        self.params = params or {}
        
        self.tech_models_ = {}
        self.monolithic_model_ = None
        self.is_fitted_ = False
        
    def _create_model(self) -> Union[LightGBMHeatingModel, EBMHeatingModel]:
        """Create a model instance."""
        if self.model_type == 'lightgbm':
            return LightGBMHeatingModel(
                params=self.params,
                use_monotonic_constraints=self.use_monotonic_constraints
            )
        elif self.model_type == 'ebm':
            return EBMHeatingModel(params=self.params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit_split(self, X: pd.DataFrame, y: pd.Series,
                  tech_group: pd.Series,
                  sample_weight: Optional[pd.Series] = None) -> 'HeatingDemandModels':
        """
        Fit separate models for each technology group.
        
        Parameters
        ----------
        X : DataFrame
            Features
        y : Series
            Target values
        tech_group : Series
            Technology group labels
        sample_weight : Series, optional
            Sample weights
            
        Returns
        -------
        self
        """
        with Timer("Fitting split models"):
            for group_name in tech_group.unique():
                if group_name in ['no_heating', 'unknown']:
                    continue
                
                mask = tech_group == group_name
                n_samples = mask.sum()
                
                if n_samples < 50:
                    logger.warning(f"Skipping {group_name}: only {n_samples} samples")
                    continue
                
                X_group = X.loc[mask].copy()
                y_group = y.loc[mask].values
                weights_group = sample_weight.loc[mask].values if sample_weight is not None else None
                
                model = self._create_model()
                model.fit(X_group, y_group, sample_weight=weights_group)
                
                self.tech_models_[group_name] = model
                logger.info(f"Fitted {self.model_type} for {group_name} on {n_samples} samples")
        
        self.is_fitted_ = True
        return self
    
    def fit_monolithic(self, X: pd.DataFrame, y: pd.Series,
                       sample_weight: Optional[pd.Series] = None) -> 'HeatingDemandModels':
        """
        Fit a single monolithic model (for comparison).
        
        Parameters
        ----------
        X : DataFrame
            Features
        y : Series
            Target values
        sample_weight : Series, optional
            Sample weights
            
        Returns
        -------
        self
        """
        with Timer("Fitting monolithic model"):
            self.monolithic_model_ = self._create_model()
            
            weights = sample_weight.values if sample_weight is not None else None
            self.monolithic_model_.fit(X, y.values, sample_weight=weights)
            
            logger.info(f"Fitted monolithic {self.model_type} on {len(X)} samples")
        
        return self
    
    def predict_split(self, X: pd.DataFrame, 
                      tech_group: pd.Series,
                      apply_correction: bool = True) -> np.ndarray:
        """
        Predict using technology-specific models.
        
        Parameters
        ----------
        X : DataFrame
            Features
        tech_group : Series
            Technology group labels
        apply_correction : bool
            Whether to apply calibration correction
            
        Returns
        -------
        array
            Predictions
        """
        if not self.tech_models_:
            raise ValueError("Split models not fitted. Call fit_split() first.")
        
        predictions = np.zeros(len(X))
        
        for group_name, model in self.tech_models_.items():
            mask = tech_group == group_name
            if mask.sum() == 0:
                continue
            
            X_group = X.loc[mask].copy()
            # Pass apply_correction if model supports it
            try:
                predictions[mask.values] = model.predict(X_group, apply_correction=apply_correction)
            except TypeError:
                predictions[mask.values] = model.predict(X_group)
        
        # Handle groups without models
        unhandled_mask = ~np.isin(tech_group.values, list(self.tech_models_.keys()))
        if unhandled_mask.sum() > 0:
            # Use average of other predictions or a default
            if predictions[~unhandled_mask].sum() > 0:
                default_pred = np.mean(predictions[~unhandled_mask])
            else:
                default_pred = 0
            predictions[unhandled_mask] = default_pred
            logger.warning(f"Used default prediction for {unhandled_mask.sum()} samples without tech group model")
        
        return predictions
    
    def predict_monolithic(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using monolithic model.
        
        Parameters
        ----------
        X : DataFrame
            Features
            
        Returns
        -------
        array
            Predictions
        """
        if self.monolithic_model_ is None:
            raise ValueError("Monolithic model not fitted. Call fit_monolithic() first.")
        
        return self.monolithic_model_.predict(X)
    
    def get_feature_importance(self, model_name: str = 'monolithic') -> pd.DataFrame:
        """
        Get feature importance from specified model.
        
        Parameters
        ----------
        model_name : str
            'monolithic' or a tech group name
            
        Returns
        -------
        DataFrame
            Feature importance table
        """
        if model_name == 'monolithic':
            if self.monolithic_model_ is None:
                raise ValueError("Monolithic model not fitted.")
            return self.monolithic_model_.get_feature_importance()
        else:
            if model_name not in self.tech_models_:
                raise ValueError(f"No model for tech group: {model_name}")
            return self.tech_models_[model_name].get_feature_importance()