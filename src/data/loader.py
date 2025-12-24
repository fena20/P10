"""
RECS 2020 Data Loader

Handles loading and initial validation of RECS 2020 microdata.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from src.utils.helpers import logger, Timer


class RECSDataLoader:
    """
    Data loader for RECS 2020 microdata.
    
    Responsibilities:
    - Load raw CSV data
    - Validate required columns
    - Extract weight columns
    - Basic data type handling
    """
    
    # Required columns that must be present
    REQUIRED_COLUMNS = [
        'DOEID',           # Unique household identifier
        'NWEIGHT',         # Final survey weight
        'TOTALBTUSPH',     # Total BTU for space heating (target)
        'TOTSQFT_EN',      # Total square footage
        'HDD65',           # Heating degree days
        'DIVISION',        # Census division
        'EQUIPM',          # Main heating equipment
        'FUELHEAT',        # Main heating fuel
    ]
    
    # RECS 2020 has 60 replicate weights
    N_REPLICATE_WEIGHTS = 60
    
    def __init__(self, data_path: str = "data/raw/recs2020_public_v7.csv"):
        """
        Initialize the data loader.
        
        Parameters
        ----------
        data_path : str
            Path to the RECS 2020 CSV file
        """
        self.data_path = Path(data_path)
        self.data = None
        self.replicate_weight_cols = [f"NWEIGHT{i}" for i in range(1, self.N_REPLICATE_WEIGHTS + 1)]
        
    def load(self, validate: bool = True) -> pd.DataFrame:
        """
        Load the RECS 2020 data.
        
        Parameters
        ----------
        validate : bool
            Whether to validate required columns
            
        Returns
        -------
        DataFrame
            Loaded RECS 2020 data
        """
        with Timer("Loading RECS 2020 data"):
            if not self.data_path.exists():
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
            self.data = pd.read_csv(self.data_path)
            logger.info(f"Loaded {len(self.data):,} records with {len(self.data.columns)} columns")
            
            if validate:
                self._validate_columns()
                self._validate_weights()
                self._log_basic_stats()
        
        return self.data
    
    def _validate_columns(self) -> None:
        """Validate that required columns are present."""
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        logger.info("All required columns present")
    
    def _validate_weights(self) -> None:
        """Validate survey weights."""
        # Check final weight
        if self.data['NWEIGHT'].isna().any():
            n_missing = self.data['NWEIGHT'].isna().sum()
            logger.warning(f"Found {n_missing} records with missing NWEIGHT")
        
        # Check replicate weights
        missing_replicate_cols = [col for col in self.replicate_weight_cols 
                                   if col not in self.data.columns]
        if missing_replicate_cols:
            logger.warning(f"Missing {len(missing_replicate_cols)} replicate weight columns")
        else:
            logger.info(f"All {self.N_REPLICATE_WEIGHTS} replicate weight columns present")
    
    def _log_basic_stats(self) -> None:
        """Log basic statistics about the loaded data."""
        # Target statistics
        target_col = 'TOTALBTUSPH'
        logger.info(f"Target ({target_col}) statistics:")
        logger.info(f"  Mean: {self.data[target_col].mean():,.0f} BTU")
        logger.info(f"  Median: {self.data[target_col].median():,.0f} BTU")
        logger.info(f"  Std: {self.data[target_col].std():,.0f} BTU")
        logger.info(f"  Min: {self.data[target_col].min():,.0f} BTU")
        logger.info(f"  Max: {self.data[target_col].max():,.0f} BTU")
        logger.info(f"  Zero values: {(self.data[target_col] == 0).sum():,}")
        
        # Weight statistics
        total_weight = self.data['NWEIGHT'].sum()
        logger.info(f"Total survey weight: {total_weight:,.0f} (representing ~{total_weight/1e6:.1f}M households)")
    
    def get_target(self, target_col: str = 'TOTALBTUSPH') -> pd.Series:
        """Get the target variable."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")
        return self.data[target_col]
    
    def get_weights(self, include_replicates: bool = False) -> pd.DataFrame:
        """
        Get survey weights.
        
        Parameters
        ----------
        include_replicates : bool
            Whether to include replicate weights
            
        Returns
        -------
        DataFrame
            Weight columns
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")
            
        weight_cols = ['NWEIGHT']
        if include_replicates:
            weight_cols.extend(self.replicate_weight_cols)
        
        return self.data[weight_cols]
    
    def get_replicate_weight_columns(self) -> List[str]:
        """Get list of replicate weight column names."""
        return self.replicate_weight_cols
    
    def compute_weighted_statistics(self, column: str) -> Dict[str, float]:
        """
        Compute weighted statistics for a column.
        
        Parameters
        ----------
        column : str
            Column to compute statistics for
            
        Returns
        -------
        dict
            Dictionary of weighted statistics
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        values = self.data[column].values
        weights = self.data['NWEIGHT'].values
        
        # Handle missing values
        valid_mask = ~(np.isnan(values) | np.isnan(weights))
        values = values[valid_mask]
        weights = weights[valid_mask]
        
        if len(values) == 0:
            return {'weighted_mean': np.nan, 'weighted_std': np.nan}
        
        weighted_mean = np.average(values, weights=weights)
        weighted_var = np.average((values - weighted_mean) ** 2, weights=weights)
        weighted_std = np.sqrt(weighted_var)
        
        # Weighted quantiles
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_weights = weights[sorted_indices]
        cumulative_weights = np.cumsum(sorted_weights)
        total_weight = cumulative_weights[-1]
        
        def get_weighted_quantile(q):
            idx = np.searchsorted(cumulative_weights, q * total_weight)
            return sorted_values[min(idx, len(sorted_values) - 1)]
        
        return {
            'weighted_mean': weighted_mean,
            'weighted_std': weighted_std,
            'weighted_median': get_weighted_quantile(0.5),
            'weighted_p10': get_weighted_quantile(0.1),
            'weighted_p25': get_weighted_quantile(0.25),
            'weighted_p75': get_weighted_quantile(0.75),
            'weighted_p90': get_weighted_quantile(0.9),
            'unweighted_mean': values.mean(),
            'unweighted_std': values.std(),
            'n_valid': len(values),
            'total_weight': total_weight
        }
    
    def create_verification_table(self) -> pd.DataFrame:
        """
        Create verification table to compare with EIA published statistics.
        
        Returns
        -------
        DataFrame
            Verification statistics by division
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        verification_stats = []
        
        # By Census Division
        for division in sorted(self.data['DIVISION'].unique()):
            div_data = self.data[self.data['DIVISION'] == division]
            
            stats = {
                'DIVISION': division,
                'n_households': len(div_data),
                'total_weight': div_data['NWEIGHT'].sum(),
            }
            
            # Weighted mean energy
            values = div_data['TOTALBTUSPH'].values
            weights = div_data['NWEIGHT'].values
            valid_mask = ~np.isnan(values)
            stats['weighted_mean_btu'] = np.average(
                values[valid_mask], weights=weights[valid_mask]
            )
            
            # Weighted mean HDD
            hdd_values = div_data['HDD65'].values
            stats['weighted_mean_hdd'] = np.average(
                hdd_values[valid_mask], weights=weights[valid_mask]
            )
            
            # Weighted mean sqft
            sqft_values = div_data['TOTSQFT_EN'].values
            stats['weighted_mean_sqft'] = np.average(
                sqft_values[valid_mask], weights=weights[valid_mask]
            )
            
            verification_stats.append(stats)
        
        return pd.DataFrame(verification_stats)


def load_recs_data(data_path: str = "data/raw/recs2020_public_v7.csv") -> pd.DataFrame:
    """
    Convenience function to load RECS 2020 data.
    
    Parameters
    ----------
    data_path : str
        Path to the RECS 2020 CSV file
        
    Returns
    -------
    DataFrame
        Loaded RECS 2020 data
    """
    loader = RECSDataLoader(data_path)
    return loader.load()
