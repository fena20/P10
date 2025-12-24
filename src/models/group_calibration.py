"""
Group-Conditional Calibration for Equity Mitigation

Implements Priority 4.1: Group-conditional calibration (CV-safe)

Instead of one global isotonic calibration, fit calibration within groups:
- renter vs owner
- cold-climate vs others
- technology × climate bins

All calibration is fitted on training folds only to avoid leakage.
Uses minimum group size threshold (e.g., min_n=200) with fallback to global calibration.

Reports subgroup bias and normalized errors before vs after calibration.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.isotonic import IsotonicRegression
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils.helpers import logger, compute_weighted_quantile
from src.evaluation.metrics import WeightedMetrics


class GroupConditionalCalibrator(BaseEstimator, TransformerMixin):
    """
    Group-conditional isotonic calibration for equity mitigation.
    
    Fits separate calibration curves for different subgroups to reduce
    disparate prediction errors across demographic/housing groups.
    
    Parameters
    ----------
    grouping_column : str
        Column name to use for grouping (e.g., 'KOWNRENT', 'climate_zone')
    min_group_size : int
        Minimum samples in a group for separate calibration (default 200)
    y_min : float
        Minimum calibrated value (default 0)
    fallback_to_global : bool
        Whether to use global calibration for small groups (default True)
    """
    
    def __init__(self,
                 grouping_column: str,
                 min_group_size: int = 200,
                 y_min: float = 0.0,
                 fallback_to_global: bool = True):
        self.grouping_column = grouping_column
        self.min_group_size = min_group_size
        self.y_min = y_min
        self.fallback_to_global = fallback_to_global
        
        self.group_calibrators_: Dict[Any, IsotonicRegression] = {}
        self.global_calibrator_: Optional[IsotonicRegression] = None
        self.group_sizes_: Dict[Any, int] = {}
        self.fitted_ = False
        
    def fit(self, X: pd.DataFrame, y: np.ndarray,
            sample_weight: Optional[np.ndarray] = None) -> 'GroupConditionalCalibrator':
        """
        Fit group-conditional calibrators.
        
        Parameters
        ----------
        X : DataFrame
            Must contain the grouping column
        y : array
            True target values
        sample_weight : array, optional
            Sample weights
            
        Returns
        -------
        self
        """
        if self.grouping_column not in X.columns:
            raise ValueError(f"Grouping column '{self.grouping_column}' not found in X")
        
        groups = X[self.grouping_column].values
        
        # Get predictions from X (expects 'y_pred' column or first numeric column)
        if 'y_pred' in X.columns:
            y_pred = X['y_pred'].values
        else:
            raise ValueError("X must contain 'y_pred' column with predictions")
        
        # Fit global calibrator
        self.global_calibrator_ = IsotonicRegression(
            y_min=self.y_min, y_max=None, out_of_bounds='clip'
        )
        self.global_calibrator_.fit(y_pred, y, sample_weight=sample_weight)
        
        # Fit group-specific calibrators
        for group in np.unique(groups):
            if pd.isna(group):
                continue
            
            mask = groups == group
            n_in_group = mask.sum()
            self.group_sizes_[group] = n_in_group
            
            if n_in_group < self.min_group_size:
                logger.debug(f"Group '{group}' has {n_in_group} samples < {self.min_group_size}; "
                           "using global calibration")
                continue
            
            # Fit group-specific calibrator
            group_calibrator = IsotonicRegression(
                y_min=self.y_min, y_max=None, out_of_bounds='clip'
            )
            
            group_weights = sample_weight[mask] if sample_weight is not None else None
            group_calibrator.fit(y_pred[mask], y[mask], sample_weight=group_weights)
            
            self.group_calibrators_[group] = group_calibrator
            logger.info(f"Fitted calibrator for group '{group}' (n={n_in_group})")
        
        self.fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame,
                  y_pred: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply group-conditional calibration.
        
        Parameters
        ----------
        X : DataFrame
            Must contain the grouping column
        y_pred : array, optional
            Predictions to calibrate (or from X['y_pred'])
            
        Returns
        -------
        array
            Calibrated predictions
        """
        if not self.fitted_:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        
        if y_pred is None:
            if 'y_pred' in X.columns:
                y_pred = X['y_pred'].values
            else:
                raise ValueError("Either y_pred must be provided or X must contain 'y_pred'")
        
        groups = X[self.grouping_column].values
        calibrated = np.zeros_like(y_pred, dtype=float)
        
        for i, (pred, group) in enumerate(zip(y_pred, groups)):
            if group in self.group_calibrators_:
                # Use group-specific calibrator
                calibrated[i] = self.group_calibrators_[group].predict([pred])[0]
            elif self.fallback_to_global and self.global_calibrator_ is not None:
                # Fallback to global
                calibrated[i] = self.global_calibrator_.predict([pred])[0]
            else:
                # No calibration
                calibrated[i] = pred
        
        return calibrated
    
    def get_calibration_report(self) -> pd.DataFrame:
        """
        Get report on calibration by group.
        
        Returns
        -------
        DataFrame
            Calibration report
        """
        rows = []
        
        for group, n in self.group_sizes_.items():
            has_specific = group in self.group_calibrators_
            rows.append({
                'group': group,
                'n_samples': n,
                'calibration_type': 'group-specific' if has_specific else 'global',
                'min_threshold': self.min_group_size
            })
        
        return pd.DataFrame(rows)


class MultiGroupCalibrator(BaseEstimator, TransformerMixin):
    """
    Multi-group calibrator that fits calibration across multiple grouping dimensions.
    
    Supports calibration by:
    - Tenure (renter vs owner)
    - Climate zone (HDD bins)
    - Technology group
    - Combined (e.g., tech × climate)
    
    Parameters
    ----------
    grouping_columns : list
        Columns to use for grouping (e.g., ['KOWNRENT', 'HDD_bin'])
    combine_groups : bool
        Whether to combine columns into single grouping (default True)
    min_group_size : int
        Minimum samples for separate calibration (default 200)
    """
    
    def __init__(self,
                 grouping_columns: List[str],
                 combine_groups: bool = True,
                 min_group_size: int = 200):
        self.grouping_columns = grouping_columns
        self.combine_groups = combine_groups
        self.min_group_size = min_group_size
        
        self.group_calibrators_: Dict[str, IsotonicRegression] = {}
        self.global_calibrator_: Optional[IsotonicRegression] = None
        self.group_sizes_: Dict[str, int] = {}
        self.fitted_ = False
        
    def _create_group_key(self, row: pd.Series) -> str:
        """Create group key from row values."""
        values = [str(row.get(col, 'unknown')) for col in self.grouping_columns]
        return '_'.join(values)
    
    def fit(self, y_pred: np.ndarray, y_true: np.ndarray,
            metadata: pd.DataFrame,
            sample_weight: Optional[np.ndarray] = None) -> 'MultiGroupCalibrator':
        """
        Fit multi-group calibrators.
        
        Parameters
        ----------
        y_pred : array
            Predictions to calibrate
        y_true : array
            True target values
        metadata : DataFrame
            Must contain the grouping columns
        sample_weight : array, optional
            Sample weights
            
        Returns
        -------
        self
        """
        # Validate columns
        missing = [c for c in self.grouping_columns if c not in metadata.columns]
        if missing:
            raise ValueError(f"Missing grouping columns: {missing}")
        
        # Fit global calibrator
        self.global_calibrator_ = IsotonicRegression(
            y_min=0, y_max=None, out_of_bounds='clip'
        )
        self.global_calibrator_.fit(y_pred, y_true, sample_weight=sample_weight)
        
        # Create group keys
        if self.combine_groups:
            group_keys = metadata[self.grouping_columns].astype(str).agg('_'.join, axis=1).values
        else:
            # Use first column only
            group_keys = metadata[self.grouping_columns[0]].astype(str).values
        
        # Fit group-specific calibrators
        for group in np.unique(group_keys):
            if pd.isna(group) or group == 'nan':
                continue
            
            mask = group_keys == group
            n_in_group = mask.sum()
            self.group_sizes_[group] = n_in_group
            
            if n_in_group < self.min_group_size:
                continue
            
            # Fit calibrator
            calibrator = IsotonicRegression(
                y_min=0, y_max=None, out_of_bounds='clip'
            )
            
            group_weights = sample_weight[mask] if sample_weight is not None else None
            calibrator.fit(y_pred[mask], y_true[mask], sample_weight=group_weights)
            
            self.group_calibrators_[group] = calibrator
            logger.info(f"Fitted calibrator for group '{group}' (n={n_in_group})")
        
        self.fitted_ = True
        return self
    
    def transform(self, y_pred: np.ndarray, metadata: pd.DataFrame) -> np.ndarray:
        """
        Apply group-conditional calibration.
        
        Parameters
        ----------
        y_pred : array
            Predictions to calibrate
        metadata : DataFrame
            Must contain grouping columns
            
        Returns
        -------
        array
            Calibrated predictions
        """
        if not self.fitted_:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        
        # Create group keys
        if self.combine_groups:
            group_keys = metadata[self.grouping_columns].astype(str).agg('_'.join, axis=1).values
        else:
            group_keys = metadata[self.grouping_columns[0]].astype(str).values
        
        calibrated = np.zeros_like(y_pred, dtype=float)
        
        for i, (pred, group) in enumerate(zip(y_pred, group_keys)):
            if group in self.group_calibrators_:
                calibrated[i] = self.group_calibrators_[group].predict([pred])[0]
            elif self.global_calibrator_ is not None:
                calibrated[i] = self.global_calibrator_.predict([pred])[0]
            else:
                calibrated[i] = pred
        
        return calibrated


def compute_calibration_equity_report(y_true: np.ndarray,
                                       y_pred_before: np.ndarray,
                                       y_pred_after: np.ndarray,
                                       weights: np.ndarray,
                                       group_column: str,
                                       metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Compute equity report comparing before vs after group calibration.
    
    Parameters
    ----------
    y_true : array
        True values
    y_pred_before : array
        Predictions before calibration
    y_pred_after : array
        Predictions after calibration
    weights : array
        Sample weights
    group_column : str
        Column to stratify by
    metadata : DataFrame
        Contains group_column
        
    Returns
    -------
    DataFrame
        Equity comparison report
    """
    wm = WeightedMetrics()
    
    groups = metadata[group_column].values
    rows = []
    
    for group in np.unique(groups):
        if pd.isna(group):
            continue
        
        mask = groups == group
        if mask.sum() < 10:
            continue
        
        # Before calibration
        metrics_before = wm.compute_all_metrics(
            y_true[mask], y_pred_before[mask], weights[mask]
        )
        
        # After calibration
        metrics_after = wm.compute_all_metrics(
            y_true[mask], y_pred_after[mask], weights[mask]
        )
        
        # Improvement
        bias_improvement = abs(metrics_before['weighted_bias']) - abs(metrics_after['weighted_bias'])
        mae_improvement = metrics_before['weighted_mae'] - metrics_after['weighted_mae']
        
        rows.append({
            'group': group,
            'n_samples': int(mask.sum()),
            'weight_sum': float(weights[mask].sum()),
            'bias_before': metrics_before['weighted_bias'],
            'bias_after': metrics_after['weighted_bias'],
            'bias_reduction': bias_improvement,
            'mae_before': metrics_before['weighted_mae'],
            'mae_after': metrics_after['weighted_mae'],
            'mae_reduction': mae_improvement,
            'nmae_before': metrics_before['weighted_nmae'],
            'nmae_after': metrics_after['weighted_nmae']
        })
    
    df = pd.DataFrame(rows)
    df.attrs['note'] = (
        f"Equity report by {group_column}. "
        "Bias reduction and MAE reduction are improvements (positive = better). "
        "nMAE = MAE/mean(Y) × 100 (scale-free)."
    )
    
    return df


class EquityAwareCalibrationPipeline:
    """
    CV-safe calibration pipeline for equity mitigation.
    
    This pipeline:
    1. Fits group-conditional calibration on training fold
    2. Applies to test fold
    3. Reports equity metrics before/after
    
    Supports multiple grouping strategies:
    - 'tenure': KOWNRENT (renter vs owner)
    - 'climate': HDD_bin (climate zones)
    - 'tech': tech_group (technology)
    - 'tech_climate': tech_group × HDD_bin
    """
    
    GROUPING_PRESETS = {
        'tenure': ['KOWNRENT'],
        'climate': ['HDD_bin'],
        'tech': ['tech_group'],
        'tech_climate': ['tech_group', 'HDD_bin'],
        'income': ['income_bin'],
        'housing': ['TYPEHUQ']
    }
    
    def __init__(self,
                 grouping_strategy: str = 'climate',
                 custom_columns: Optional[List[str]] = None,
                 min_group_size: int = 200,
                 combine_groups: bool = True):
        """
        Initialize equity-aware calibration pipeline.
        
        Parameters
        ----------
        grouping_strategy : str
            Preset: 'tenure', 'climate', 'tech', 'tech_climate', 'income', 'housing'
        custom_columns : list, optional
            Custom grouping columns (overrides preset)
        min_group_size : int
            Minimum samples for group-specific calibration
        combine_groups : bool
            Whether to combine multiple columns into single grouping
        """
        self.grouping_strategy = grouping_strategy
        self.custom_columns = custom_columns
        self.min_group_size = min_group_size
        self.combine_groups = combine_groups
        
        # Determine grouping columns
        if custom_columns is not None:
            self.grouping_columns = custom_columns
        elif grouping_strategy in self.GROUPING_PRESETS:
            self.grouping_columns = self.GROUPING_PRESETS[grouping_strategy]
        else:
            raise ValueError(f"Unknown grouping_strategy: {grouping_strategy}. "
                           f"Options: {list(self.GROUPING_PRESETS.keys())}")
        
        self.calibrator_: Optional[MultiGroupCalibrator] = None
        self.equity_reports_: Dict[str, pd.DataFrame] = {}
        
    def fit_transform_cv_fold(self,
                               y_pred_train: np.ndarray,
                               y_true_train: np.ndarray,
                               metadata_train: pd.DataFrame,
                               weights_train: np.ndarray,
                               y_pred_test: np.ndarray,
                               metadata_test: pd.DataFrame) -> np.ndarray:
        """
        Fit calibration on train fold and transform test fold (CV-safe).
        
        Parameters
        ----------
        y_pred_train : array
            Training predictions
        y_true_train : array
            Training true values
        metadata_train : DataFrame
            Training metadata (contains grouping columns)
        weights_train : array
            Training weights
        y_pred_test : array
            Test predictions to calibrate
        metadata_test : DataFrame
            Test metadata
            
        Returns
        -------
        array
            Calibrated test predictions
        """
        # Validate columns exist
        available = [c for c in self.grouping_columns if c in metadata_train.columns]
        if not available:
            logger.warning(f"No grouping columns available in metadata; using global calibration")
            # Fallback to global calibration
            calibrator = IsotonicRegression(y_min=0, y_max=None, out_of_bounds='clip')
            calibrator.fit(y_pred_train, y_true_train, sample_weight=weights_train)
            return calibrator.predict(y_pred_test)
        
        # Fit calibrator on training data
        self.calibrator_ = MultiGroupCalibrator(
            grouping_columns=available,
            combine_groups=self.combine_groups,
            min_group_size=self.min_group_size
        )
        
        self.calibrator_.fit(
            y_pred=y_pred_train,
            y_true=y_true_train,
            metadata=metadata_train,
            sample_weight=weights_train
        )
        
        # Transform test predictions
        return self.calibrator_.transform(y_pred_test, metadata_test)
    
    def compute_equity_improvement(self,
                                    y_true: np.ndarray,
                                    y_pred_before: np.ndarray,
                                    y_pred_after: np.ndarray,
                                    weights: np.ndarray,
                                    metadata: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Compute equity improvement reports across different subgroups.
        
        Parameters
        ----------
        y_true : array
            True values
        y_pred_before : array
            Predictions before calibration
        y_pred_after : array
            Predictions after calibration
        weights : array
            Sample weights
        metadata : DataFrame
            Contains subgroup columns
            
        Returns
        -------
        dict
            Equity reports by subgroup type
        """
        reports = {}
        
        # Standard subgroup analyses
        subgroup_columns = {
            'tenure': 'KOWNRENT',
            'climate': 'HDD_bin',
            'housing_type': 'TYPEHUQ',
            'income': 'MONEYPY',
            'tech_group': 'tech_group'
        }
        
        for name, col in subgroup_columns.items():
            if col not in metadata.columns:
                continue
            
            report = compute_calibration_equity_report(
                y_true=y_true,
                y_pred_before=y_pred_before,
                y_pred_after=y_pred_after,
                weights=weights,
                group_column=col,
                metadata=metadata
            )
            
            reports[name] = report
        
        self.equity_reports_ = reports
        return reports
    
    def get_summary(self) -> pd.DataFrame:
        """
        Get summary of calibration effectiveness.
        
        Returns
        -------
        DataFrame
            Summary statistics
        """
        if not self.equity_reports_:
            raise ValueError("No equity reports computed. Call compute_equity_improvement() first.")
        
        rows = []
        for name, report in self.equity_reports_.items():
            # Aggregate metrics
            total_bias_reduction = report['bias_reduction'].sum()
            avg_mae_reduction = report['mae_reduction'].mean()
            
            # Count groups with improved bias
            n_improved = (report['bias_reduction'] > 0).sum()
            n_groups = len(report)
            
            rows.append({
                'grouping': name,
                'n_groups': n_groups,
                'n_improved': n_improved,
                'pct_improved': n_improved / n_groups * 100 if n_groups > 0 else 0,
                'total_bias_reduction': total_bias_reduction,
                'avg_mae_reduction': avg_mae_reduction
            })
        
        return pd.DataFrame(rows)


def add_climate_bin(metadata: pd.DataFrame,
                    hdd_column: str = 'HDD65',
                    bin_column: str = 'HDD_bin') -> pd.DataFrame:
    """
    Add climate bin column to metadata if not present.
    
    Parameters
    ----------
    metadata : DataFrame
        Metadata with HDD column
    hdd_column : str
        Name of HDD column
    bin_column : str
        Name for new bin column
        
    Returns
    -------
    DataFrame
        Metadata with climate bin column
    """
    if bin_column in metadata.columns:
        return metadata
    
    if hdd_column not in metadata.columns:
        logger.warning(f"HDD column '{hdd_column}' not found; cannot create climate bins")
        return metadata
    
    metadata = metadata.copy()
    
    bins = [-np.inf, 2000, 4000, 6000, 8000, np.inf]
    labels = ['very_mild', 'mild', 'moderate', 'cold', 'very_cold']
    
    metadata[bin_column] = pd.cut(
        metadata[hdd_column],
        bins=bins,
        labels=labels
    )
    
    return metadata
