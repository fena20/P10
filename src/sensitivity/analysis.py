"""
Sensitivity Analysis Module

Implements Section 14: Sensitivity analyses (minimum required)
- Targets: E vs E/Area vs excess-demand score
- Climate restriction: exclude HDD < 1000
- COVID controls: direct vs proxies vs none
- Technology assignment: primary-only vs hybrid-handling vs exclusion
- Constraints: with vs without monotonic constraints
- Uncertainty: metric-only jackknife vs refit-sensitivity subset
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from copy import deepcopy

from src.utils.helpers import logger, Timer


@dataclass
class SensitivityResult:
    """Result from a single sensitivity analysis."""
    name: str
    variation: str
    metrics: Dict[str, float]
    n_samples: int
    config: Dict[str, Any]


class SensitivityAnalysis:
    """
    Sensitivity analysis framework for heating demand modeling.
    
    Runs variations of the analysis to assess robustness.
    """
    
    def __init__(self, base_config: Dict[str, Any] = None):
        """
        Initialize sensitivity analysis.
        
        Parameters
        ----------
        base_config : dict
            Base configuration for the analysis
        """
        self.base_config = base_config or {}
        self.results_ = []
        
    def run_target_sensitivity(self,
                                df: pd.DataFrame,
                                model_factory: Callable,
                                feature_builder: 'FeatureBuilder',
                                cv_runner: Callable) -> Dict[str, SensitivityResult]:
        """
        Compare different target definitions.
        
        Parameters
        ----------
        df : DataFrame
            Preprocessed data
        model_factory : callable
            Function to create models
        feature_builder : FeatureBuilder
            Feature builder
        cv_runner : callable
            Function to run CV
            
        Returns
        -------
        dict
            Results for each target
        """
        results = {}
        
        targets = {
            'E_absolute': 'TOTALBTUSPH',
            'E_per_area': 'E_heat_per_area'
        }
        
        for target_name, target_col in targets.items():
            if target_col not in df.columns:
                logger.warning(f"Target {target_col} not found, skipping")
                continue
            
            logger.info(f"Running sensitivity: target = {target_name}")
            
            # Run CV with this target
            cv_result = cv_runner(df, target_col=target_col)
            
            results[target_name] = SensitivityResult(
                name='target',
                variation=target_name,
                metrics=cv_result.outer_metrics if hasattr(cv_result, 'outer_metrics') else cv_result,
                n_samples=len(df),
                config={'target': target_col}
            )
        
        return results
    
    def run_climate_restriction_sensitivity(self,
                                             df: pd.DataFrame,
                                             cv_runner: Callable,
                                             min_hdd_values: List[float] = [0, 1000, 2000]) -> Dict[str, SensitivityResult]:
        """
        Compare results with different HDD restrictions.
        
        Parameters
        ----------
        df : DataFrame
            Preprocessed data
        cv_runner : callable
            Function to run CV
        min_hdd_values : list
            Minimum HDD thresholds to test
            
        Returns
        -------
        dict
            Results for each threshold
        """
        results = {}
        
        for min_hdd in min_hdd_values:
            df_filtered = df[df['HDD65'] >= min_hdd].copy()
            
            logger.info(f"Running sensitivity: min_hdd = {min_hdd} "
                       f"({len(df_filtered)} samples)")
            
            if len(df_filtered) < 100:
                logger.warning(f"Too few samples for min_hdd={min_hdd}, skipping")
                continue
            
            cv_result = cv_runner(df_filtered)
            
            results[f'min_hdd_{min_hdd}'] = SensitivityResult(
                name='climate_restriction',
                variation=f'min_hdd_{min_hdd}',
                metrics=cv_result.outer_metrics if hasattr(cv_result, 'outer_metrics') else cv_result,
                n_samples=len(df_filtered),
                config={'min_hdd': min_hdd}
            )
        
        return results
    
    def run_covid_controls_sensitivity(self,
                                        df: pd.DataFrame,
                                        cv_runner: Callable,
                                        modes: List[str] = ['direct', 'proxy', 'none']) -> Dict[str, SensitivityResult]:
        """
        Compare results with different COVID control strategies.
        
        Parameters
        ----------
        df : DataFrame
            Preprocessed data
        cv_runner : callable
            Function to run CV with covid_mode parameter
        modes : list
            COVID control modes to test
            
        Returns
        -------
        dict
            Results for each mode
        """
        results = {}
        
        for mode in modes:
            logger.info(f"Running sensitivity: covid_mode = {mode}")
            
            cv_result = cv_runner(df, covid_mode=mode)
            
            results[f'covid_{mode}'] = SensitivityResult(
                name='covid_controls',
                variation=mode,
                metrics=cv_result.outer_metrics if hasattr(cv_result, 'outer_metrics') else cv_result,
                n_samples=len(df),
                config={'covid_mode': mode}
            )
        
        return results
    
    def run_tech_assignment_sensitivity(self,
                                         df_raw: pd.DataFrame,
                                         cv_runner: Callable,
                                         rules: List[str] = ['primary_only', 'with_hybrid', 'exclude_hybrid']) -> Dict[str, SensitivityResult]:
        """
        Compare results with different technology assignment rules.
        
        Parameters
        ----------
        df_raw : DataFrame
            Raw (unprocessed) data
        cv_runner : callable
            Function to run CV
        rules : list
            Assignment rules to test
            
        Returns
        -------
        dict
            Results for each rule
        """
        from src.data.preprocessor import preprocess_recs_data
        
        results = {}
        
        for rule in rules:
            exclude_hybrid = rule == 'exclude_hybrid'
            
            logger.info(f"Running sensitivity: tech_assignment = {rule}")
            
            df_proc, _ = preprocess_recs_data(
                df_raw.copy(),
                assignment_rule='primary_only',
                exclude_no_heating=True,
                exclude_hybrid=exclude_hybrid
            )
            
            cv_result = cv_runner(df_proc)
            
            results[f'tech_{rule}'] = SensitivityResult(
                name='tech_assignment',
                variation=rule,
                metrics=cv_result.outer_metrics if hasattr(cv_result, 'outer_metrics') else cv_result,
                n_samples=len(df_proc),
                config={'assignment_rule': rule, 'exclude_hybrid': exclude_hybrid}
            )
        
        return results
    
    def run_monotonic_constraints_sensitivity(self,
                                               df: pd.DataFrame,
                                               cv_runner: Callable,
                                               constraints: List[bool] = [True, False]) -> Dict[str, SensitivityResult]:
        """
        Compare results with and without monotonic constraints.
        
        Parameters
        ----------
        df : DataFrame
            Preprocessed data
        cv_runner : callable
            Function to run CV with use_constraints parameter
        constraints : list
            Constraint settings to test
            
        Returns
        -------
        dict
            Results for each setting
        """
        results = {}
        
        for use_constraints in constraints:
            constraint_str = 'with_constraints' if use_constraints else 'no_constraints'
            logger.info(f"Running sensitivity: {constraint_str}")
            
            cv_result = cv_runner(df, use_constraints=use_constraints)
            
            results[constraint_str] = SensitivityResult(
                name='monotonic_constraints',
                variation=constraint_str,
                metrics=cv_result.outer_metrics if hasattr(cv_result, 'outer_metrics') else cv_result,
                n_samples=len(df),
                config={'use_constraints': use_constraints}
            )
        
        return results
    
    def compile_results(self, all_results: Dict[str, Dict[str, SensitivityResult]]) -> pd.DataFrame:
        """
        Compile all sensitivity results into a summary table.
        
        Parameters
        ----------
        all_results : dict
            Dictionary of result dictionaries from each sensitivity
            
        Returns
        -------
        DataFrame
            Summary table
        """
        rows = []
        
        for analysis_name, analysis_results in all_results.items():
            for variation_name, result in analysis_results.items():
                row = {
                    'analysis': result.name,
                    'variation': result.variation,
                    'n_samples': result.n_samples
                }
                
                # Add metrics
                if isinstance(result.metrics, dict):
                    for metric_name, value in result.metrics.items():
                        row[metric_name] = value
                
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def create_comparison_plot(self,
                                results_df: pd.DataFrame,
                                metric: str = 'weighted_rmse',
                                output_path: Optional[str] = None) -> 'plt.Figure':
        """
        Create comparison plot for sensitivity results.
        
        Parameters
        ----------
        results_df : DataFrame
            Compiled results
        metric : str
            Metric to plot
        output_path : str, optional
            Path to save figure
            
        Returns
        -------
        Figure
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Group by analysis type
        analyses = results_df['analysis'].unique()
        
        x_positions = []
        x_labels = []
        colors = []
        values = []
        
        color_map = {
            'target': 'steelblue',
            'climate_restriction': 'coral',
            'covid_controls': 'forestgreen',
            'tech_assignment': 'purple',
            'monotonic_constraints': 'orange'
        }
        
        pos = 0
        for analysis in analyses:
            analysis_data = results_df[results_df['analysis'] == analysis]
            
            for _, row in analysis_data.iterrows():
                if metric in row:
                    x_positions.append(pos)
                    x_labels.append(f"{row['variation']}")
                    colors.append(color_map.get(analysis, 'gray'))
                    values.append(row[metric])
                    pos += 1
            
            pos += 0.5  # Gap between groups
        
        ax.bar(x_positions, values, color=colors)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'Sensitivity Analysis: {metric.replace("_", " ").title()}')
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved sensitivity plot: {output_path}")
        
        return fig


def run_full_sensitivity_analysis(df_raw: pd.DataFrame,
                                   cv_runner: Callable,
                                   output_dir: str = 'outputs/') -> pd.DataFrame:
    """
    Run full sensitivity analysis.
    
    Parameters
    ----------
    df_raw : DataFrame
        Raw RECS data
    cv_runner : callable
        Function to run CV analysis
    output_dir : str
        Output directory
        
    Returns
    -------
    DataFrame
        Compiled sensitivity results
    """
    from src.data.preprocessor import preprocess_recs_data
    
    sensitivity = SensitivityAnalysis()
    all_results = {}
    
    # Base preprocessed data
    df_proc, _ = preprocess_recs_data(df_raw, exclude_no_heating=True)
    
    # 1. Climate restriction
    with Timer("Climate restriction sensitivity"):
        all_results['climate'] = sensitivity.run_climate_restriction_sensitivity(
            df_proc, cv_runner, min_hdd_values=[0, 1000]
        )
    
    # 2. Technology assignment
    with Timer("Technology assignment sensitivity"):
        all_results['tech'] = sensitivity.run_tech_assignment_sensitivity(
            df_raw, cv_runner, rules=['primary_only', 'exclude_hybrid']
        )
    
    # Compile and save
    results_df = sensitivity.compile_results(all_results)
    results_df.to_csv(f'{output_dir}/tables/sensitivity_summary.csv', index=False, encoding='utf-8-sig')
    
    return results_df
