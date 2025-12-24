"""
Objective Comparison: Tweedie vs Gamma Shootout

Implements Priority 2.1: Two fully identical runs differing only in objective:
- Tweedie: objective="tweedie", tweedie_variance_power=1.5
- Gamma: objective="gamma"

Logs and reports:
- wRMSE / wMAE / wBias / wR²
- Tail metrics (top-10% underprediction)
- Lift@10 / NDCG

This resolves objective inconsistencies flagged in the manuscript.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
from copy import deepcopy

from src.utils.helpers import logger, Timer
from src.evaluation.metrics import WeightedMetrics
from src.models.tail_bias_models import TailBiasMetrics


def run_objective_comparison(X: pd.DataFrame,
                             y: pd.Series,
                             weights: pd.Series,
                             feature_builder,
                             tech_group: Optional[pd.Series] = None,
                             metadata: Optional[pd.DataFrame] = None,
                             n_outer_folds: int = 5,
                             n_inner_folds: int = 3,
                             n_search_iter: int = 20,
                             random_state: int = 42,
                             output_dir: str = 'outputs/',
                             base_params: Optional[Dict] = None,
                             param_grid: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Run Tweedie vs Gamma objective comparison with identical configuration.
    
    This produces a side-by-side comparison to resolve objective inconsistencies.
    
    Parameters
    ----------
    X : DataFrame
        Features
    y : Series
        Target (TOTALBTUSPH)
    weights : Series
        Survey weights (NWEIGHT)
    feature_builder : FeatureBuilder
        Feature preprocessor
    tech_group : Series, optional
        Technology group labels
    metadata : DataFrame, optional
        Additional metadata (HDD65, DIVISION, etc.)
    n_outer_folds : int
        Number of outer CV folds
    n_inner_folds : int
        Number of inner CV folds
    n_search_iter : int
        Number of hyperparameter search iterations
    random_state : int
        Random seed
    output_dir : str
        Output directory
    base_params : dict, optional
        Base LightGBM parameters
    param_grid : dict, optional
        Hyperparameter search space
        
    Returns
    -------
    dict
        Comparison results
    """
    from src.evaluation.nested_cv import NestedCrossValidator
    from src.models.main_models import LightGBMHeatingModel
    
    out_dir = Path(output_dir) / 'objective_comparison'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Default params (shared between both objectives)
    if base_params is None:
        base_params = {
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
            'random_state': random_state,
            'n_jobs': -1
        }
    
    # Default search space
    if param_grid is None:
        param_grid = {
            'n_estimators': [200, 300, 400],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.03, 0.05, 0.1],
            'num_leaves': [31, 63],
            'min_child_samples': [20, 50],
        }
    
    # Objectives to compare
    objectives = {
        'tweedie': {
            'objective': 'tweedie',
            'tweedie_variance_power': 1.5,
            'metric': 'rmse'
        },
        'gamma': {
            'objective': 'gamma',
            'metric': 'rmse'
        }
    }
    
    results = {}
    all_predictions = {}
    fold_metrics = {}
    
    logger.info("="*60)
    logger.info("OBJECTIVE COMPARISON: Tweedie vs Gamma")
    logger.info("="*60)
    
    for obj_name, obj_params in objectives.items():
        logger.info(f"\nRunning {obj_name} objective...")
        
        # Merge params
        model_params = {**base_params, **obj_params}
        
        # Model factory for this objective
        def model_factory(obj_name=obj_name, model_params=model_params, **params):
            merged = {**model_params, **params}
            return LightGBMHeatingModel(params=merged, use_monotonic_constraints=False)
        
        # Run nested CV
        cv = NestedCrossValidator(
            outer_folds=n_outer_folds,
            inner_folds=n_inner_folds,
            n_search_iter=n_search_iter,
            random_state=random_state,
            output_dir=str(out_dir / obj_name)
        )
        
        cv_result = cv.run(
            X, y, weights,
            model_factory=model_factory,
            param_grid=param_grid,
            feature_builder=feature_builder,
            tech_group=tech_group,
            additional_metadata=metadata
        )
        
        # Store results
        results[obj_name] = cv_result
        all_predictions[obj_name] = cv.outer_predictions_
        fold_metrics[obj_name] = cv_result.outer_metrics_by_fold.copy()
        
        logger.info(f"{obj_name} overall metrics: {cv_result.outer_metrics}")
    
    # Build comparison table
    comparison_df = build_comparison_table(
        results, all_predictions, y.values, weights.values, tech_group
    )
    
    # Save results
    comparison_df.to_csv(out_dir / 'objective_comparison.csv', index=False, encoding='utf-8-sig')
    
    # Log final recommendation
    logger.info("\n" + "="*60)
    logger.info("OBJECTIVE COMPARISON SUMMARY")
    logger.info("="*60)
    print(comparison_df.to_string())
    
    # Make recommendation
    tweedie_rmse = results['tweedie'].outer_metrics['weighted_rmse']
    gamma_rmse = results['gamma'].outer_metrics['weighted_rmse']
    
    recommendation = 'tweedie' if tweedie_rmse <= gamma_rmse else 'gamma'
    logger.info(f"\nRecommendation: Use {recommendation} objective")
    logger.info(f"  Tweedie wRMSE: {tweedie_rmse:,.0f}")
    logger.info(f"  Gamma wRMSE: {gamma_rmse:,.0f}")
    
    return {
        'cv_results': results,
        'predictions': all_predictions,
        'fold_metrics': fold_metrics,
        'comparison_df': comparison_df,
        'recommendation': recommendation,
        'output_dir': str(out_dir)
    }


def build_comparison_table(results: Dict[str, Any],
                           predictions: Dict[str, np.ndarray],
                           y_true: np.ndarray,
                           weights: np.ndarray,
                           tech_group: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Build comparison table between objectives.
    
    Parameters
    ----------
    results : dict
        CV results by objective
    predictions : dict
        Predictions by objective
    y_true : array
        True values
    weights : array
        Sample weights
    tech_group : Series, optional
        Technology groups
        
    Returns
    -------
    DataFrame
        Comparison table
    """
    wm = WeightedMetrics()
    tbm = TailBiasMetrics(k_percentile=10)
    
    rows = []
    
    for obj_name in results.keys():
        y_pred = predictions[obj_name]
        cv_result = results[obj_name]
        
        # Overall metrics
        std_metrics = cv_result.outer_metrics
        
        # Tail metrics
        tail_metrics = tbm.compute_all_metrics(y_true, y_pred, weights)
        
        row = {
            'Objective': obj_name.capitalize(),
            'wRMSE': std_metrics.get('weighted_rmse', np.nan),
            'wMAE': std_metrics.get('weighted_mae', np.nan),
            'wR²': std_metrics.get('weighted_r2', np.nan),
            'wBias': std_metrics.get('weighted_bias', np.nan),
            'Top-10% Bias (%)': tail_metrics['top_decile_bias_pct'],
            'Lift@10': tail_metrics['lift_at_k'],
            'NDCG': tail_metrics['ndcg'],
            'Precision@10': tail_metrics['precision_at_k'],
            'Recall@10': tail_metrics['recall_at_k']
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Format numeric columns
    for col in ['wRMSE', 'wMAE', 'wBias']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
    
    for col in ['wR²', 'NDCG', 'Precision@10', 'Recall@10']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    
    for col in ['Lift@10']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    
    for col in ['Top-10% Bias (%)']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "")
    
    df.attrs['note'] = (
        "Objective comparison: Tweedie (power=1.5) vs Gamma. "
        "Identical features, folds, weights. "
        "Top-10% Bias: 0 = perfect, negative = underprediction."
    )
    
    return df


def build_tech_group_comparison(results: Dict[str, Any],
                                predictions: Dict[str, np.ndarray],
                                y_true: np.ndarray,
                                weights: np.ndarray,
                                tech_group: np.ndarray) -> pd.DataFrame:
    """
    Build comparison table stratified by technology group.
    
    Parameters
    ----------
    results : dict
        CV results by objective
    predictions : dict
        Predictions by objective
    y_true : array
        True values
    weights : array
        Sample weights
    tech_group : array
        Technology group labels
        
    Returns
    -------
    DataFrame
        Comparison by tech group
    """
    tbm = TailBiasMetrics(k_percentile=10)
    
    rows = []
    
    for group in np.unique(tech_group):
        mask = tech_group == group
        if mask.sum() < 50:
            continue
        
        for obj_name in results.keys():
            y_pred = predictions[obj_name]
            
            tail_metrics = tbm.compute_all_metrics(
                y_true[mask], y_pred[mask], weights[mask]
            )
            
            row = {
                'tech_group': group,
                'Objective': obj_name.capitalize(),
                'Top-10% Bias (%)': tail_metrics['top_decile_bias_pct'],
                'Lift@10': tail_metrics['lift_at_k'],
                'NDCG': tail_metrics['ndcg'],
                'n': int(mask.sum())
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


class ObjectiveShootout:
    """
    Manages the Tweedie vs Gamma objective comparison.
    
    This class provides a structured interface for:
    1. Running identical experiments with different objectives
    2. Comparing results
    3. Logging and reporting
    4. Making final recommendations
    """
    
    def __init__(self,
                 output_dir: str = 'outputs/',
                 n_outer_folds: int = 5,
                 n_inner_folds: int = 3,
                 n_search_iter: int = 20,
                 random_state: int = 42):
        """
        Initialize objective shootout.
        
        Parameters
        ----------
        output_dir : str
            Output directory
        n_outer_folds : int
            Number of outer CV folds
        n_inner_folds : int
            Number of inner CV folds
        n_search_iter : int
            Search iterations
        random_state : int
            Random seed
        """
        self.output_dir = Path(output_dir)
        self.n_outer_folds = n_outer_folds
        self.n_inner_folds = n_inner_folds
        self.n_search_iter = n_search_iter
        self.random_state = random_state
        
        self.results_ = None
        self.recommendation_ = None
        
    def run(self,
            X: pd.DataFrame,
            y: pd.Series,
            weights: pd.Series,
            feature_builder,
            tech_group: Optional[pd.Series] = None,
            metadata: Optional[pd.DataFrame] = None,
            base_params: Optional[Dict] = None,
            param_grid: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run the objective comparison.
        
        Returns
        -------
        dict
            Comparison results
        """
        self.results_ = run_objective_comparison(
            X=X,
            y=y,
            weights=weights,
            feature_builder=feature_builder,
            tech_group=tech_group,
            metadata=metadata,
            n_outer_folds=self.n_outer_folds,
            n_inner_folds=self.n_inner_folds,
            n_search_iter=self.n_search_iter,
            random_state=self.random_state,
            output_dir=str(self.output_dir),
            base_params=base_params,
            param_grid=param_grid
        )
        
        self.recommendation_ = self.results_['recommendation']
        return self.results_
    
    def get_recommendation(self) -> str:
        """Get the recommended objective."""
        if self.recommendation_ is None:
            raise ValueError("Run shootout first using .run()")
        return self.recommendation_
    
    def get_comparison_table(self) -> pd.DataFrame:
        """Get the comparison table."""
        if self.results_ is None:
            raise ValueError("Run shootout first using .run()")
        return self.results_['comparison_df']
    
    def save_summary(self, path: Optional[str] = None) -> str:
        """
        Save summary to file.
        
        Parameters
        ----------
        path : str, optional
            Output path (default: output_dir/objective_summary.txt)
            
        Returns
        -------
        str
            Path to saved file
        """
        if self.results_ is None:
            raise ValueError("Run shootout first using .run()")
        
        if path is None:
            path = self.output_dir / 'objective_comparison' / 'objective_summary.txt'
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            f.write("OBJECTIVE COMPARISON SUMMARY\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Recommendation: {self.recommendation_}\n\n")
            
            f.write("Comparison Table:\n")
            f.write(self.results_['comparison_df'].to_string() + "\n\n")
            
            f.write("Note: This comparison uses identical features, folds, and weights.\n")
            f.write("The recommended objective should be used consistently in the paper.\n")
        
        return str(path)
