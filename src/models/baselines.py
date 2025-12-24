"""
Physics-Based Baseline Models for Heating Demand

Implements Section 6.1: Required baselines for physics credibility.
Each technology group has its own baseline reflecting thermodynamic behavior.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge
from scipy.optimize import minimize
import warnings

from src.utils.helpers import logger, Timer


class CombustionBaseline(BaseEstimator, RegressorMixin):
    """
    Physics baseline for combustion heating systems.
    
    Model: E = a + b*HDD + c*Area + d*HDD*Area
    
    Justification:
    - Energy proportional to HDD (degree-day method)
    - Energy proportional to heated area
    - Interaction captures heat loss rate per unit area
    """
    
    def __init__(self, include_interaction: bool = True, regularization: float = 0.1):
        self.include_interaction = include_interaction
        self.regularization = regularization
        self.coef_ = None
        self.intercept_ = None
        self.model_ = None
        
    def _build_features(self, hdd: np.ndarray, area: np.ndarray) -> np.ndarray:
        """Build feature matrix."""
        features = [hdd, area]
        if self.include_interaction:
            features.append(hdd * area)
        return np.column_stack(features)
    
    def fit(self, X: pd.DataFrame, y: np.ndarray, 
            sample_weight: Optional[np.ndarray] = None) -> 'CombustionBaseline':
        """
        Fit the combustion baseline model.
        
        Parameters
        ----------
        X : DataFrame
            Must contain 'HDD65' and 'TOTSQFT_EN' columns
        y : array
            Target values (energy)
        sample_weight : array, optional
            Sample weights
            
        Returns
        -------
        self
        """
        hdd = X['HDD65'].values
        area = X['TOTSQFT_EN'].values
        
        features = self._build_features(hdd, area)
        
        self.model_ = Ridge(alpha=self.regularization, fit_intercept=True)
        self.model_.fit(features, y, sample_weight=sample_weight)
        
        self.coef_ = self.model_.coef_
        self.intercept_ = self.model_.intercept_
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict energy consumption."""
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        hdd = X['HDD65'].values
        area = X['TOTSQFT_EN'].values
        features = self._build_features(hdd, area)
        
        predictions = self.model_.predict(features)
        # Clip to non-negative
        return np.maximum(predictions, 0)
    
    def get_coefficients(self) -> Dict[str, float]:
        """Get interpretable coefficient values."""
        coefs = {
            'intercept': self.intercept_,
            'hdd_coef': self.coef_[0],
            'area_coef': self.coef_[1],
        }
        if self.include_interaction:
            coefs['hdd_area_interaction'] = self.coef_[2]
        return coefs


class HeatPumpBaseline(BaseEstimator, RegressorMixin):
    """
    Physics baseline for heat pump systems.
    
    Model: Piecewise or smooth nonlinear vs HDD/temperature proxy.
    
    Heat pump COP varies with outdoor temperature:
    - Higher efficiency at mild temperatures
    - Decreasing efficiency as temperature drops
    - May include auxiliary resistance heating at very low temps
    
    Energy = Area * HDD * (1 / COP(T))
    
    Simplified model uses piecewise linear or polynomial in HDD.
    """
    
    def __init__(self, model_type: str = 'piecewise', 
                 hdd_breakpoint: float = 4000,
                 regularization: float = 0.1):
        self.model_type = model_type
        self.hdd_breakpoint = hdd_breakpoint
        self.regularization = regularization
        self.coef_ = None
        self.model_low_ = None
        self.model_high_ = None
        self.model_single_ = None
        
    def _build_features_piecewise(self, hdd: np.ndarray, area: np.ndarray,
                                   regime: str = 'low') -> np.ndarray:
        """Build features for piecewise model."""
        return np.column_stack([hdd, area, hdd * area])
    
    def _build_features_polynomial(self, hdd: np.ndarray, area: np.ndarray) -> np.ndarray:
        """Build polynomial features."""
        return np.column_stack([
            hdd, hdd**2, area, hdd * area
        ])
    
    def fit(self, X: pd.DataFrame, y: np.ndarray,
            sample_weight: Optional[np.ndarray] = None) -> 'HeatPumpBaseline':
        """
        Fit the heat pump baseline model.
        
        Parameters
        ----------
        X : DataFrame
            Must contain 'HDD65' and 'TOTSQFT_EN' columns
        y : array
            Target values (energy)
        sample_weight : array, optional
            Sample weights
            
        Returns
        -------
        self
        """
        hdd = X['HDD65'].values
        area = X['TOTSQFT_EN'].values
        
        if self.model_type == 'piecewise':
            # Split into low and high HDD regimes
            low_mask = hdd <= self.hdd_breakpoint
            high_mask = ~low_mask
            
            if low_mask.sum() > 10:
                features_low = self._build_features_piecewise(hdd[low_mask], area[low_mask], 'low')
                weights_low = sample_weight[low_mask] if sample_weight is not None else None
                self.model_low_ = Ridge(alpha=self.regularization)
                self.model_low_.fit(features_low, y[low_mask], sample_weight=weights_low)
            
            if high_mask.sum() > 10:
                features_high = self._build_features_piecewise(hdd[high_mask], area[high_mask], 'high')
                weights_high = sample_weight[high_mask] if sample_weight is not None else None
                self.model_high_ = Ridge(alpha=self.regularization)
                self.model_high_.fit(features_high, y[high_mask], sample_weight=weights_high)
            
            # Fallback single model if either regime has too few samples
            if self.model_low_ is None or self.model_high_ is None:
                features = self._build_features_polynomial(hdd, area)
                self.model_single_ = Ridge(alpha=self.regularization)
                self.model_single_.fit(features, y, sample_weight=sample_weight)
                
        else:  # polynomial
            features = self._build_features_polynomial(hdd, area)
            self.model_single_ = Ridge(alpha=self.regularization)
            self.model_single_.fit(features, y, sample_weight=sample_weight)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict energy consumption."""
        hdd = X['HDD65'].values
        area = X['TOTSQFT_EN'].values
        
        predictions = np.zeros(len(hdd))
        
        if self.model_type == 'piecewise' and self.model_low_ is not None and self.model_high_ is not None:
            low_mask = hdd <= self.hdd_breakpoint
            high_mask = ~low_mask
            
            if low_mask.sum() > 0:
                features_low = self._build_features_piecewise(hdd[low_mask], area[low_mask], 'low')
                predictions[low_mask] = self.model_low_.predict(features_low)
            
            if high_mask.sum() > 0:
                features_high = self._build_features_piecewise(hdd[high_mask], area[high_mask], 'high')
                predictions[high_mask] = self.model_high_.predict(features_high)
        else:
            features = self._build_features_polynomial(hdd, area)
            predictions = self.model_single_.predict(features)
        
        return np.maximum(predictions, 0)


class ResistanceBaseline(BaseEstimator, RegressorMixin):
    """
    Physics baseline for electric resistance heating.
    
    Model: E = a + b*HDD + c*Area + d*HDD*Area
    
    Similar structure to combustion but with different efficiency characteristics.
    Electric resistance has ~100% efficiency (COP=1), so energy use directly
    proportional to heat load.
    """
    
    def __init__(self, regularization: float = 0.1):
        self.regularization = regularization
        self.model_ = None
        self.coef_ = None
        self.intercept_ = None
        
    def _build_features(self, hdd: np.ndarray, area: np.ndarray) -> np.ndarray:
        """Build feature matrix."""
        return np.column_stack([hdd, area, hdd * area])
    
    def fit(self, X: pd.DataFrame, y: np.ndarray,
            sample_weight: Optional[np.ndarray] = None) -> 'ResistanceBaseline':
        """
        Fit the resistance baseline model.
        """
        hdd = X['HDD65'].values
        area = X['TOTSQFT_EN'].values
        
        features = self._build_features(hdd, area)
        
        self.model_ = Ridge(alpha=self.regularization, fit_intercept=True)
        self.model_.fit(features, y, sample_weight=sample_weight)
        
        self.coef_ = self.model_.coef_
        self.intercept_ = self.model_.intercept_
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict energy consumption."""
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        hdd = X['HDD65'].values
        area = X['TOTSQFT_EN'].values
        features = self._build_features(hdd, area)
        
        predictions = self.model_.predict(features)
        return np.maximum(predictions, 0)
    
    def get_coefficients(self) -> Dict[str, float]:
        """Get interpretable coefficient values."""
        return {
            'intercept': self.intercept_,
            'hdd_coef': self.coef_[0],
            'area_coef': self.coef_[1],
            'hdd_area_interaction': self.coef_[2]
        }


class PhysicsBaselines:
    """
    Container for all physics baselines by technology group.
    
    Provides unified interface for fitting and predicting across tech groups.
    """
    
    def __init__(self):
        self.baselines = {
            'combustion': CombustionBaseline(),
            'electric_heat_pump': HeatPumpBaseline(),
            'electric_resistance': ResistanceBaseline()
        }
        self.is_fitted_ = False
        self.tech_group_metrics_ = {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            tech_group: pd.Series,
            sample_weight: Optional[pd.Series] = None) -> 'PhysicsBaselines':
        """
        Fit all baselines on their respective technology groups.
        
        Parameters
        ----------
        X : DataFrame
            Features (must include HDD65, TOTSQFT_EN)
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
        with Timer("Fitting physics baselines"):
            for group_name, baseline in self.baselines.items():
                mask = tech_group == group_name
                n_samples = mask.sum()
                
                if n_samples < 20:
                    logger.warning(f"Skipping {group_name} baseline: only {n_samples} samples")
                    continue
                
                X_group = X.loc[mask].copy()
                y_group = y.loc[mask].values
                weights_group = sample_weight.loc[mask].values if sample_weight is not None else None
                
                baseline.fit(X_group, y_group, sample_weight=weights_group)
                logger.info(f"Fitted {group_name} baseline on {n_samples} samples")
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X: pd.DataFrame, tech_group: pd.Series) -> np.ndarray:
        """
        Predict using appropriate baseline for each technology group.
        
        Parameters
        ----------
        X : DataFrame
            Features
        tech_group : Series
            Technology group labels
            
        Returns
        -------
        array
            Predictions
        """
        if not self.is_fitted_:
            raise ValueError("Baselines not fitted. Call fit() first.")
        
        predictions = np.zeros(len(X))
        
        for group_name, baseline in self.baselines.items():
            mask = tech_group == group_name
            if mask.sum() == 0:
                continue
            
            # Check if baseline is fitted (handles different baseline types)
            is_fitted = False
            if hasattr(baseline, 'model_') and baseline.model_ is not None:
                is_fitted = True
            elif hasattr(baseline, 'model_single_') and baseline.model_single_ is not None:
                is_fitted = True
            elif hasattr(baseline, 'model_low_') and baseline.model_low_ is not None:
                is_fitted = True
            
            if not is_fitted:
                logger.warning(f"No fitted model for {group_name}, using zeros")
                continue
            
            X_group = X.loc[mask].copy()
            predictions[mask.values] = baseline.predict(X_group)
        
        return predictions
    
    def get_baseline_predictions(self, X: pd.DataFrame, 
                                  tech_group: pd.Series) -> pd.DataFrame:
        """
        Get baseline predictions as DataFrame with metadata.
        
        Returns
        -------
        DataFrame
            Predictions with tech_group column
        """
        predictions = self.predict(X, tech_group)
        
        return pd.DataFrame({
            'baseline_pred': predictions,
            'tech_group': tech_group
        })
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series,
                 tech_group: pd.Series, 
                 sample_weight: pd.Series) -> pd.DataFrame:
        """
        Evaluate baselines on each technology group.
        
        Parameters
        ----------
        X : DataFrame
            Features
        y : Series
            True values
        tech_group : Series
            Technology group labels
        sample_weight : Series
            Sample weights
            
        Returns
        -------
        DataFrame
            Evaluation metrics by tech group
        """
        from src.evaluation.metrics import WeightedMetrics
        
        metrics = WeightedMetrics()
        predictions = self.predict(X, tech_group)
        
        results = []
        
        for group_name in self.baselines.keys():
            mask = tech_group == group_name
            if mask.sum() == 0:
                continue
            
            y_true = y.loc[mask].values
            y_pred = predictions[mask.values]
            weights = sample_weight.loc[mask].values
            
            group_metrics = metrics.compute_all_metrics(y_true, y_pred, weights)
            group_metrics['tech_group'] = group_name
            results.append(group_metrics)
        
        # Overall metrics
        overall_metrics = metrics.compute_all_metrics(
            y.values, predictions, sample_weight.values
        )
        overall_metrics['tech_group'] = 'overall'
        results.append(overall_metrics)
        
        return pd.DataFrame(results)
    
    def get_coefficients_summary(self) -> pd.DataFrame:
        """Get summary of baseline coefficients for interpretation."""
        coefs = []
        
        for group_name, baseline in self.baselines.items():
            if hasattr(baseline, 'get_coefficients') and baseline.model_ is not None:
                group_coefs = baseline.get_coefficients()
                group_coefs['tech_group'] = group_name
                coefs.append(group_coefs)
        
        if coefs:
            return pd.DataFrame(coefs)
        return pd.DataFrame()


class MonolithicBaseline(BaseEstimator, RegressorMixin):
    """
    Monolithic baseline without technology splits.
    
    Used for comparison with split models to test H1.
    """
    
    def __init__(self, regularization: float = 0.1):
        self.regularization = regularization
        self.model_ = None
        
    def _build_features(self, X: pd.DataFrame) -> np.ndarray:
        """Build feature matrix."""
        hdd = X['HDD65'].values
        area = X['TOTSQFT_EN'].values
        
        return np.column_stack([
            hdd, area, hdd * area, hdd**2
        ])
    
    def fit(self, X: pd.DataFrame, y: np.ndarray,
            sample_weight: Optional[np.ndarray] = None) -> 'MonolithicBaseline':
        """Fit monolithic baseline."""
        features = self._build_features(X)
        
        self.model_ = Ridge(alpha=self.regularization, fit_intercept=True)
        self.model_.fit(features, y, sample_weight=sample_weight)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict."""
        if self.model_ is None:
            raise ValueError("Model not fitted.")
        
        features = self._build_features(X)
        predictions = self.model_.predict(features)
        return np.maximum(predictions, 0)
