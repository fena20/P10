"""
Utility functions and helpers for the heating demand modeling framework.
"""

import os
import yaml
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/default_config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def ensure_dir(path: str) -> None:
    """Ensure directory exists, create if not."""
    Path(path).mkdir(parents=True, exist_ok=True)


def set_random_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, 
                fill_value: float = np.nan) -> np.ndarray:
    """Safely divide arrays, handling division by zero."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.true_divide(numerator, denominator)
        result[~np.isfinite(result)] = fill_value
    return result


def compute_weighted_quantile(values: np.ndarray, weights: np.ndarray, 
                               quantile: float) -> float:
    """
    Compute weighted quantile.
    
    Parameters
    ----------
    values : array-like
        Values to compute quantile for
    weights : array-like
        Weights for each value
    quantile : float
        Quantile to compute (0-1)
        
    Returns
    -------
    float
        Weighted quantile value
    """
    values = np.asarray(values)
    weights = np.asarray(weights)
    
    # Handle missing values
    valid_mask = ~(np.isnan(values) | np.isnan(weights))
    values = values[valid_mask]
    weights = weights[valid_mask]
    
    if len(values) == 0:
        return np.nan
    
    # Sort by values
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]
    
    # Compute cumulative weights
    cumulative_weights = np.cumsum(sorted_weights)
    total_weight = cumulative_weights[-1]
    
    # Find quantile position
    threshold = quantile * total_weight
    idx = np.searchsorted(cumulative_weights, threshold)
    
    if idx >= len(sorted_values):
        return sorted_values[-1]
    
    return sorted_values[idx]


def compute_weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted mean."""
    valid_mask = ~(np.isnan(values) | np.isnan(weights))
    return np.average(values[valid_mask], weights=weights[valid_mask])


def compute_weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted standard deviation."""
    valid_mask = ~(np.isnan(values) | np.isnan(weights))
    v = values[valid_mask]
    w = weights[valid_mask]
    
    weighted_mean = np.average(v, weights=w)
    weighted_var = np.average((v - weighted_mean) ** 2, weights=w)
    return np.sqrt(weighted_var)


def create_stratification_key(df: pd.DataFrame, 
                               columns: List[str]) -> pd.Series:
    """
    Create a stratification key from multiple columns.
    
    Parameters
    ----------
    df : DataFrame
        Input dataframe
    columns : list
        Columns to combine for stratification
        
    Returns
    -------
    Series
        Combined stratification key
    """
    return df[columns].astype(str).agg('_'.join, axis=1)


def bin_continuous_variable(series: pd.Series, 
                            bins: List[float], 
                            labels: Optional[List[str]] = None) -> pd.Series:
    """
    Bin a continuous variable into categories.
    
    Parameters
    ----------
    series : Series
        Continuous variable to bin
    bins : list
        Bin edges
    labels : list, optional
        Labels for bins
        
    Returns
    -------
    Series
        Binned categorical variable
    """
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True)


def log1p_transform(x: np.ndarray) -> np.ndarray:
    """Apply log1p transformation."""
    return np.log1p(np.maximum(x, 0))


def inverse_log1p_transform(x: np.ndarray) -> np.ndarray:
    """Inverse log1p transformation."""
    return np.expm1(x)


def clip_predictions(predictions: np.ndarray, 
                     min_val: float = 0, 
                     max_val: Optional[float] = None) -> Tuple[np.ndarray, float]:
    """
    Clip predictions to valid range and report clipping rate.
    
    Parameters
    ----------
    predictions : array
        Predicted values
    min_val : float
        Minimum valid value
    max_val : float, optional
        Maximum valid value
        
    Returns
    -------
    clipped_predictions : array
        Clipped predictions
    clip_rate : float
        Fraction of predictions that were clipped
    """
    n_total = len(predictions)
    
    clipped = predictions.copy()
    n_clipped = 0
    
    if min_val is not None:
        below_mask = clipped < min_val
        n_clipped += below_mask.sum()
        clipped[below_mask] = min_val
        
    if max_val is not None:
        above_mask = clipped > max_val
        n_clipped += above_mask.sum()
        clipped[above_mask] = max_val
    
    clip_rate = n_clipped / n_total if n_total > 0 else 0
    
    return clipped, clip_rate


def format_metric(value: float, se: Optional[float] = None, 
                  precision: int = 3) -> str:
    """Format metric value with optional standard error."""
    if se is not None:
        return f"{value:.{precision}f} ± {se:.{precision}f}"
    return f"{value:.{precision}f}"


def print_section_header(title: str, char: str = "=", width: int = 80) -> None:
    """Print a formatted section header."""
    print("\n" + char * width)
    print(title.center(width))
    print(char * width + "\n")


def validate_no_leakage(feature_cols: List[str], 
                        blacklist: List[str]) -> List[str]:
    """
    Validate that no leakage features are in the feature list.
    
    Parameters
    ----------
    feature_cols : list
        List of feature columns
    blacklist : list
        List of blacklisted (leakage) columns
        
    Returns
    -------
    list
        List of any blacklisted columns found
    """
    return [col for col in feature_cols if col in blacklist]


class Timer:
    """Simple timer context manager for tracking execution time."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.elapsed = None
        
    def __enter__(self):
        import time
        self.start_time = time.time()
        return self
        
    def __exit__(self, *args):
        import time
        self.elapsed = time.time() - self.start_time
        logger.info(f"{self.name} completed in {self.elapsed:.2f} seconds")


def get_recs_equipment_labels() -> Dict[int, str]:
    """Get human-readable labels for RECS equipment codes."""
    return {
        -2: "No heating",
        2: "Steam/hot water system",
        3: "Central warm-air furnace",
        4: "Heat pump",
        5: "Built-in electric units",
        7: "Built-in room heater",
        8: "Wood-burning stove",
        10: "Portable electric heaters",
        13: "Other equipment",
        99: "Other/unknown"
    }


def get_recs_fuel_labels() -> Dict[int, str]:
    """Get human-readable labels for RECS fuel codes."""
    return {
        -2: "No heating fuel",
        1: "Natural gas",
        2: "Propane/LPG",
        3: "Fuel oil/kerosene",
        5: "Electricity",
        7: "Wood",
        99: "Other"
    }


def get_recs_division_labels() -> Dict[int, str]:
    """Get human-readable labels for Census divisions."""
    return {
        1: "New England",
        2: "Middle Atlantic",
        3: "East North Central",
        4: "West North Central",
        5: "South Atlantic",
        6: "East South Central",
        7: "West South Central",
        8: "Mountain North",
        9: "Mountain South",
        10: "Pacific"
    }


__all__ = [
    'load_config',
    'ensure_dir',
    'set_random_seed',
    'safe_divide',
    'compute_weighted_quantile',
    'compute_weighted_mean',
    'compute_weighted_std',
    'create_stratification_key',
    'bin_continuous_variable',
    'log1p_transform',
    'inverse_log1p_transform',
    'clip_predictions',
    'format_metric',
    'print_section_header',
    'validate_no_leakage',
    'Timer',
    'get_recs_equipment_labels',
    'get_recs_fuel_labels',
    'get_recs_division_labels',
    'logger'
]
