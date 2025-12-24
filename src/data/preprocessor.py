"""
RECS 2020 Data Preprocessor

Handles technology grouping, data cleaning, and preprocessing.
Implements Section 3: Technology Grouping (critical, leakage-safe, reproducible)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

from src.utils.helpers import (
    logger, Timer, bin_continuous_variable, 
    get_recs_equipment_labels, get_recs_fuel_labels
)


class TechGroup(Enum):
    """Technology group enumeration."""
    COMBUSTION = "combustion"
    ELECTRIC_HEAT_PUMP = "electric_heat_pump"
    ELECTRIC_RESISTANCE = "electric_resistance"
    HYBRID_AMBIGUOUS = "hybrid_ambiguous"
    NO_HEATING = "no_heating"


class RECSPreprocessor:
    """
    Preprocessor for RECS 2020 data.
    
    Implements technology grouping rules as specified in Section 3:
    - Combustion: natural gas / propane / fuel oil / other combustion primary heating
    - Electric—Heat Pump: primary heating equipment = heat pump
    - Electric—Resistance: primary heating equipment = electric resistance/baseboard
    - Hybrid/Ambiguous: cases with conflicting indicators or EQUIPM 13, 99
    
    Key principle: All preprocessing that could leak information must be fit
    inside the CV inner loop only.
    """
    
    # Technology grouping rules (Section 3.1)
    # EQUIPM codes
    COMBUSTION_EQUIPM = [2, 3, 7, 8]  # Steam/radiator, furnace, room heater, wood stove
    HEAT_PUMP_EQUIPM = [4]  # Heat pump
    RESISTANCE_EQUIPM = [5, 10]  # Built-in electric, portable heaters
    AMBIGUOUS_EQUIPM = [13, 99]  # Other equipment
    NO_HEATING_EQUIPM = [-2]  # No heating
    
    # FUELHEAT codes
    COMBUSTION_FUELS = [1, 2, 3, 7]  # Natural gas, propane, fuel oil, wood
    ELECTRIC_FUEL = [5]  # Electricity
    
    # HDD bins for stratification (Section 3.2)
    HDD_BINS = [0, 2000, 4000, 6000, 8000, float('inf')]
    HDD_BIN_LABELS = ['very_mild', 'mild', 'moderate', 'cold', 'very_cold']
    
    def __init__(self, 
                 assignment_rule: str = 'primary_only',
                 handle_hybrid: str = 'keep_separate'):
        """
        Initialize preprocessor.
        
        Parameters
        ----------
        assignment_rule : str
            Technology assignment rule:
            - 'primary_only': Use only primary equipment/fuel
            - 'primary_secondary': Consider secondary heating
        handle_hybrid : str
            How to handle hybrid/ambiguous cases:
            - 'keep_separate': Keep as separate group (Option A)
            - 'exclude': Exclude from analysis (Option B)
            - 'assign_primary': Assign based on primary fuel
        """
        self.assignment_rule = assignment_rule
        self.handle_hybrid = handle_hybrid
        self.tech_group_stats_ = None
        
    def assign_technology_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign technology groups to each household.
        
        This is the critical technology grouping step (Section 3.1).
        Groups are mutually exclusive.
        
        Parameters
        ----------
        df : DataFrame
            Input dataframe with EQUIPM and FUELHEAT columns
            
        Returns
        -------
        DataFrame
            DataFrame with 'tech_group' column added
        """
        with Timer("Assigning technology groups"):
            df = df.copy()
            
            # Initialize with unknown
            df['tech_group'] = 'unknown'
            df['tech_group_code'] = -99
            
            # Get equipment and fuel
            equipm = df['EQUIPM'].values
            fuelheat = df['FUELHEAT'].values
            
            # Check for auxiliary equipment (for hybrid detection)
            has_aux_equipment = df['EQUIPAUX'].notna() & (df['EQUIPAUX'] > 0)
            
            # Rule 1: No heating
            no_heat_mask = np.isin(equipm, self.NO_HEATING_EQUIPM)
            df.loc[no_heat_mask, 'tech_group'] = TechGroup.NO_HEATING.value
            df.loc[no_heat_mask, 'tech_group_code'] = 0
            
            # Rule 2: Electric Heat Pump
            # Must be heat pump equipment AND electric fuel
            heat_pump_mask = (
                np.isin(equipm, self.HEAT_PUMP_EQUIPM) & 
                np.isin(fuelheat, self.ELECTRIC_FUEL)
            )
            df.loc[heat_pump_mask, 'tech_group'] = TechGroup.ELECTRIC_HEAT_PUMP.value
            df.loc[heat_pump_mask, 'tech_group_code'] = 2
            
            # Rule 3: Electric Resistance
            # Electric resistance equipment with electric fuel
            resistance_mask = (
                np.isin(equipm, self.RESISTANCE_EQUIPM) & 
                np.isin(fuelheat, self.ELECTRIC_FUEL)
            )
            df.loc[resistance_mask, 'tech_group'] = TechGroup.ELECTRIC_RESISTANCE.value
            df.loc[resistance_mask, 'tech_group_code'] = 3
            
            # Rule 4: Combustion
            # Combustion equipment with combustion fuel
            combustion_mask = (
                np.isin(equipm, self.COMBUSTION_EQUIPM) & 
                np.isin(fuelheat, self.COMBUSTION_FUELS)
            )
            df.loc[combustion_mask, 'tech_group'] = TechGroup.COMBUSTION.value
            df.loc[combustion_mask, 'tech_group_code'] = 1
            
            # Rule 5: Hybrid/Ambiguous
            # Cases with ambiguous equipment codes OR
            # Mismatched equipment/fuel combinations
            ambiguous_equipm_mask = np.isin(equipm, self.AMBIGUOUS_EQUIPM)
            
            # Mismatched: combustion equipment with electric fuel or vice versa
            mismatch_mask = (
                (np.isin(equipm, self.COMBUSTION_EQUIPM) & np.isin(fuelheat, self.ELECTRIC_FUEL)) |
                (np.isin(equipm, self.RESISTANCE_EQUIPM) & np.isin(fuelheat, self.COMBUSTION_FUELS))
            )
            
            # Remaining unknown cases
            remaining_unknown = df['tech_group'] == 'unknown'
            
            hybrid_mask = (
                ambiguous_equipm_mask | 
                mismatch_mask | 
                (remaining_unknown & ~no_heat_mask)
            )
            
            df.loc[hybrid_mask, 'tech_group'] = TechGroup.HYBRID_AMBIGUOUS.value
            df.loc[hybrid_mask, 'tech_group_code'] = 4
            
            # Log statistics
            self._log_tech_group_stats(df)
            
            return df
    
    def _log_tech_group_stats(self, df: pd.DataFrame) -> None:
        """Log technology group statistics."""
        stats = df.groupby('tech_group').agg({
            'DOEID': 'count',
            'NWEIGHT': 'sum'
        }).rename(columns={'DOEID': 'n_households', 'NWEIGHT': 'total_weight'})
        
        stats['pct_households'] = stats['n_households'] / len(df) * 100
        stats['pct_weighted'] = stats['total_weight'] / df['NWEIGHT'].sum() * 100
        
        self.tech_group_stats_ = stats
        
        logger.info("Technology group distribution:")
        for group, row in stats.iterrows():
            logger.info(f"  {group}: {row['n_households']:,} households "
                       f"({row['pct_households']:.1f}%), "
                       f"{row['total_weight']:,.0f} weighted ({row['pct_weighted']:.1f}%)")
    
    def create_stratification_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create columns for CV stratification (Section 3.2).
        
        Stratify by: Census Division × technology group × HDD bins
        
        Parameters
        ----------
        df : DataFrame
            Input dataframe
            
        Returns
        -------
        DataFrame
            DataFrame with stratification columns added
        """
        df = df.copy()
        
        # HDD bins
        df['HDD_bin'] = pd.cut(
            df['HDD65'],
            bins=self.HDD_BINS,
            labels=self.HDD_BIN_LABELS,
            include_lowest=True
        )
        
        # Combined stratification key
        df['strat_key'] = (
            df['DIVISION'].astype(str) + '_' +
            df['tech_group'].astype(str) + '_' +
            df['HDD_bin'].astype(str)
        )
        
        return df
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features (non-leakage).
        
        Parameters
        ----------
        df : DataFrame
            Input dataframe
            
        Returns
        -------
        DataFrame
            DataFrame with derived features
        """
        df = df.copy()
        
        # Energy intensity (kBTU/sqft)
        df['E_heat_per_area'] = df['TOTALBTUSPH'] / df['TOTSQFT_EN']
        
        # HDD per sqft (climate-adjusted size)
        df['HDD_per_sqft'] = df['HDD65'] / df['TOTSQFT_EN']
        
        # Heating degree day categories
        df['HDD_category'] = pd.cut(
            df['HDD65'],
            bins=[0, 2000, 4000, 6000, float('inf')],
            labels=['mild', 'moderate', 'cold', 'very_cold']
        )
        
        # Building age categories
        df['building_age_category'] = pd.cut(
            df['YEARMADERANGE'],
            bins=[0, 3, 6, 9, float('inf')],
            labels=['pre_1970', '1970_1990', '1990_2010', 'post_2010']
        )
        
        # Household size category
        df['household_size_category'] = pd.cut(
            df['NHSLDMEM'],
            bins=[0, 1, 2, 4, float('inf')],
            labels=['single', 'couple', 'small_family', 'large_family']
        )
        
        return df
    
    def filter_for_analysis(self, 
                            df: pd.DataFrame,
                            exclude_no_heating: bool = True,
                            exclude_hybrid: bool = False,
                            min_hdd: Optional[float] = None,
                            max_hdd: Optional[float] = None) -> pd.DataFrame:
        """
        Filter data for analysis.
        
        Parameters
        ----------
        df : DataFrame
            Input dataframe with tech_group assigned
        exclude_no_heating : bool
            Whether to exclude households with no heating
        exclude_hybrid : bool
            Whether to exclude hybrid/ambiguous cases
        min_hdd : float, optional
            Minimum HDD threshold
        max_hdd : float, optional
            Maximum HDD threshold
            
        Returns
        -------
        DataFrame
            Filtered dataframe
        """
        df_filtered = df.copy()
        initial_n = len(df_filtered)
        
        # Exclude no heating
        if exclude_no_heating:
            mask = df_filtered['tech_group'] != TechGroup.NO_HEATING.value
            df_filtered = df_filtered[mask]
            logger.info(f"Excluded {initial_n - len(df_filtered)} no-heating records")
        
        # Exclude hybrid
        if exclude_hybrid:
            n_before = len(df_filtered)
            mask = df_filtered['tech_group'] != TechGroup.HYBRID_AMBIGUOUS.value
            df_filtered = df_filtered[mask]
            logger.info(f"Excluded {n_before - len(df_filtered)} hybrid/ambiguous records")
        
        # HDD filtering
        if min_hdd is not None:
            n_before = len(df_filtered)
            df_filtered = df_filtered[df_filtered['HDD65'] >= min_hdd]
            logger.info(f"Excluded {n_before - len(df_filtered)} records with HDD < {min_hdd}")
        
        if max_hdd is not None:
            n_before = len(df_filtered)
            df_filtered = df_filtered[df_filtered['HDD65'] <= max_hdd]
            logger.info(f"Excluded {n_before - len(df_filtered)} records with HDD > {max_hdd}")
        
        # Remove records with zero or missing target
        n_before = len(df_filtered)
        df_filtered = df_filtered[
            (df_filtered['TOTALBTUSPH'] > 0) & 
            (df_filtered['TOTALBTUSPH'].notna())
        ]
        logger.info(f"Excluded {n_before - len(df_filtered)} records with zero/missing energy")
        
        logger.info(f"Final analysis dataset: {len(df_filtered):,} records "
                   f"({len(df_filtered)/initial_n*100:.1f}% of original)")
        
        return df_filtered.reset_index(drop=True)
    
    def get_tech_group_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get detailed summary statistics by technology group.
        
        Parameters
        ----------
        df : DataFrame
            Preprocessed dataframe
            
        Returns
        -------
        DataFrame
            Summary statistics
        """
        summary = []
        
        for group in df['tech_group'].unique():
            group_df = df[df['tech_group'] == group]
            weights = group_df['NWEIGHT'].values
            
            stats = {
                'tech_group': group,
                'n_households': len(group_df),
                'total_weight': weights.sum(),
                'pct_weighted': weights.sum() / df['NWEIGHT'].sum() * 100,
            }
            
            # Weighted mean energy
            energy = group_df['TOTALBTUSPH'].values
            valid = ~np.isnan(energy)
            if valid.sum() > 0:
                stats['weighted_mean_energy'] = np.average(energy[valid], weights=weights[valid])
                stats['weighted_std_energy'] = np.sqrt(
                    np.average((energy[valid] - stats['weighted_mean_energy'])**2, 
                              weights=weights[valid])
                )
            
            # Weighted mean HDD
            hdd = group_df['HDD65'].values
            stats['weighted_mean_hdd'] = np.average(hdd[valid], weights=weights[valid])
            
            # Weighted mean sqft
            sqft = group_df['TOTSQFT_EN'].values
            stats['weighted_mean_sqft'] = np.average(sqft[valid], weights=weights[valid])
            
            # Energy intensity
            intensity = group_df['E_heat_per_area'].values
            valid_intensity = ~np.isnan(intensity)
            if valid_intensity.sum() > 0:
                stats['weighted_mean_intensity'] = np.average(
                    intensity[valid_intensity], weights=weights[valid_intensity]
                )
            
            summary.append(stats)
        
        return pd.DataFrame(summary)
    
    def identify_outliers(self, 
                          df: pd.DataFrame,
                          method: str = 'iqr',
                          columns: List[str] = None,
                          threshold: float = 3.0) -> pd.DataFrame:
        """
        Identify outliers for documentation (not automatic removal).
        
        Per Section 7: Keep policy-relevant "super-emitters"; 
        remove only impossible records with documented rules.
        
        Parameters
        ----------
        df : DataFrame
            Input dataframe
        method : str
            Outlier detection method ('iqr' or 'zscore')
        columns : list
            Columns to check for outliers
        threshold : float
            IQR multiplier or z-score threshold
            
        Returns
        -------
        DataFrame
            DataFrame with outlier flags
        """
        if columns is None:
            columns = ['TOTALBTUSPH', 'E_heat_per_area', 'HDD65']
        
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
                
            values = df[col].values
            weights = df['NWEIGHT'].values
            valid = ~np.isnan(values)
            
            if method == 'iqr':
                # Weighted quartiles
                sorted_idx = np.argsort(values[valid])
                sorted_vals = values[valid][sorted_idx]
                sorted_weights = weights[valid][sorted_idx]
                cum_weights = np.cumsum(sorted_weights)
                total_weight = cum_weights[-1]
                
                q25_idx = np.searchsorted(cum_weights, 0.25 * total_weight)
                q75_idx = np.searchsorted(cum_weights, 0.75 * total_weight)
                q25 = sorted_vals[q25_idx]
                q75 = sorted_vals[q75_idx]
                iqr = q75 - q25
                
                lower = q25 - threshold * iqr
                upper = q75 + threshold * iqr
                
                df[f'{col}_outlier'] = (values < lower) | (values > upper)
                
            elif method == 'zscore':
                weighted_mean = np.average(values[valid], weights=weights[valid])
                weighted_std = np.sqrt(
                    np.average((values[valid] - weighted_mean)**2, weights=weights[valid])
                )
                z_scores = np.abs((values - weighted_mean) / weighted_std)
                df[f'{col}_outlier'] = z_scores > threshold
        
        # Log outlier counts
        for col in columns:
            outlier_col = f'{col}_outlier'
            if outlier_col in df.columns:
                n_outliers = df[outlier_col].sum()
                logger.info(f"Identified {n_outliers} outliers in {col} "
                           f"({n_outliers/len(df)*100:.2f}%)")
        
        return df
    
    def identify_impossible_records(self, df: pd.DataFrame) -> pd.Series:
        """
        Identify physically impossible records for removal.
        
        These are records that violate basic physical constraints:
        - Negative energy values
        - Energy values exceeding reasonable physical limits
        - Zero square footage
        
        Parameters
        ----------
        df : DataFrame
            Input dataframe
            
        Returns
        -------
        Series
            Boolean mask of impossible records
        """
        impossible = pd.Series(False, index=df.index)
        
        # Negative energy (should not happen but check)
        impossible |= df['TOTALBTUSPH'] < 0
        
        # Impossibly high energy (> 500 million BTU/year for residential)
        # This would be > 10x typical commercial building
        impossible |= df['TOTALBTUSPH'] > 500000000
        
        # Zero or negative square footage
        impossible |= df['TOTSQFT_EN'] <= 0
        
        # Impossibly high intensity (> 500 kBTU/sqft)
        # Highest efficiency buildings are ~30-50, very inefficient ~150-200
        if 'E_heat_per_area' in df.columns:
            impossible |= df['E_heat_per_area'] > 500
        
        n_impossible = impossible.sum()
        logger.info(f"Identified {n_impossible} physically impossible records")
        
        return impossible


def preprocess_recs_data(df: pd.DataFrame,
                         assignment_rule: str = 'primary_only',
                         exclude_no_heating: bool = True,
                         exclude_hybrid: bool = False,
                         min_hdd: Optional[float] = None) -> Tuple[pd.DataFrame, RECSPreprocessor]:
    """
    Convenience function to preprocess RECS data.
    
    Parameters
    ----------
    df : DataFrame
        Raw RECS data
    assignment_rule : str
        Technology assignment rule
    exclude_no_heating : bool
        Whether to exclude no-heating records
    exclude_hybrid : bool
        Whether to exclude hybrid/ambiguous records
    min_hdd : float, optional
        Minimum HDD threshold
        
    Returns
    -------
    df_processed : DataFrame
        Preprocessed data
    preprocessor : RECSPreprocessor
        Fitted preprocessor
    """
    preprocessor = RECSPreprocessor(assignment_rule=assignment_rule)
    
    # Assign technology groups
    df = preprocessor.assign_technology_groups(df)
    
    # Create stratification columns
    df = preprocessor.create_stratification_columns(df)
    
    # Create derived features
    df = preprocessor.create_derived_features(df)
    
    # Filter for analysis
    df = preprocessor.filter_for_analysis(
        df,
        exclude_no_heating=exclude_no_heating,
        exclude_hybrid=exclude_hybrid,
        min_hdd=min_hdd
    )
    
    return df, preprocessor
