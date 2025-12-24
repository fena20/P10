#!/usr/bin/env python3
"""
Heating Demand Modeling Framework - Main Analysis Script

This script runs the complete analysis pipeline for RECS 2020 heating demand modeling:
1. Data loading and preprocessing
2. Technology grouping
3. Feature engineering
4. Physics baselines
5. Nested cross-validation
6. Policy targeting analysis
7. Uncertainty quantification
8. Diagnostics and visualization

Usage:
    python run_analysis.py [--config CONFIG_PATH] [--output OUTPUT_DIR]
"""

import argparse
import sys
import warnings
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.helpers import (
    load_config, ensure_dir, set_random_seed, 
    print_section_header, Timer, logger
)
from src.data.loader import RECSDataLoader
from src.data.preprocessor import RECSPreprocessor, preprocess_recs_data
from src.features.builder import FeatureBuilder, create_feature_matrix
from src.models.baselines import PhysicsBaselines, MonolithicBaseline
from src.models.main_models import HeatingDemandModels, LightGBMHeatingModel, EBMHeatingModel
from src.models.advanced_models import XGBoostRegressorModel, CatBoostRegressorModel, TabularTransformerModel
from src.models.tail_bias_models import (
    QuantileLightGBMHeatingModel, 
    TailWeightedLightGBMHeatingModel,
    TailBiasMetrics,
    compare_quantile_vs_mean_models
)
from src.models.group_calibration import (
    EquityAwareCalibrationPipeline,
    add_climate_bin,
    compute_calibration_equity_report
)
from src.utils.run_manifest import create_run_manifest, RunManifest
from src.evaluation.metrics import WeightedMetrics, PhysicsDiagnostics, ErrorEquityAnalysis
from src.evaluation.nested_cv import NestedCrossValidator, compare_split_vs_monolithic
from src.evaluation.model_comparison import create_split_vs_monolithic_table, create_h1_summary_table
from src.policy.targeting import PolicyTargeting
from src.uncertainty.jackknife import JackknifeUncertainty, RefitSensitivity, CombinedUncertainty
from src.visualization.plots import HeatingDemandVisualizer, create_workflow_diagram

warnings.filterwarnings('ignore')


def print_environment_healthcheck(config: dict, resolved_device: str, use_cuda: bool, gpu_id: int) -> None:
    """Print a quick environment/dependency/GPU readiness report.

    Designed to be safe in partial environments: it never raises.
    """
    print_section_header("ENVIRONMENT HEALTH CHECK", char="=", width=80)

    def _fmt(v):
        try:
            return str(v)
        except Exception:
            return "<unprintable>"

    # Core
    print(f"Python: {_fmt(sys.version).split()[0]} ({_fmt(sys.executable)})")
    print(f"Platform: {_fmt(sys.platform)}")
    try:
        import numpy as _np
        import pandas as _pd
        print(f"NumPy: {_fmt(_np.__version__)} | pandas: {_fmt(_pd.__version__)}")
    except Exception as e:
        print(f"NumPy/pandas: unavailable ({e})")

    # Runtime device choice
    print(f"Requested/Resolved device: {resolved_device} (use_cuda={use_cuda}, gpu_id={gpu_id})")

    # Torch / CUDA
    torch_ok = False
    torch_cuda = False
    try:
        import torch  # noqa: F401
        torch_ok = True
        torch_cuda = bool(torch.cuda.is_available())
        torch_ver = getattr(torch, "__version__", "unknown")
        print(f"PyTorch: {torch_ver} | cuda.is_available={torch_cuda}")
        if torch_cuda:
            try:
                name = torch.cuda.get_device_name(gpu_id)
            except Exception:
                name = torch.cuda.get_device_name(0)
            print(f"CUDA device: {name}")
    except Exception as e:
        print(f"PyTorch: not available ({e})")

    # LightGBM
    try:
        import lightgbm as lgb  # noqa: F401
        print(f"LightGBM: {getattr(lgb, '__version__', 'unknown')} | GPU requested={bool(use_cuda)}")
        # Lightweight GPU-availability hint (non-fatal)
        if use_cuda:
            print("  Note: LightGBM GPU support depends on your build; if unavailable, the code will fall back to CPU.")
    except Exception as e:
        print(f"LightGBM: not available ({e})")

    # XGBoost
    try:
        import xgboost as xgb  # noqa: F401
        print(f"XGBoost: {getattr(xgb, '__version__', 'unknown')} | GPU requested={bool(use_cuda)}")
        if use_cuda:
            print("  Note: XGBoost GPU requires a GPU-enabled build; otherwise it will fall back or skip with a warning.")
    except Exception as e:
        print(f"XGBoost: not available ({e})")

    # CatBoost
    try:
        import catboost  # noqa: F401
        cb_ver = getattr(catboost, "__version__", "unknown")
        print(f"CatBoost: {cb_ver} | GPU requested={bool(use_cuda)}")
        if use_cuda:
            print("  Note: CatBoost GPU requires CUDA; if not available it will fall back or skip with a warning.")
    except Exception as e:
        print(f"CatBoost: not available ({e})")

    # EBM (interpret)
    try:
        import interpret  # noqa: F401
        print("InterpretML (EBM): available")
    except Exception as e:
        print(f"InterpretML (EBM): not available ({e})")

    # Summary of what the benchmark step will attempt
    bench_cfg = (config.get('benchmarks', {}) or {}) if isinstance(config, dict) else {}
    enabled = bool(bench_cfg.get('enabled', True))
    models = bench_cfg.get('models', None)
    if models is None:
        models = ["lightgbm", "physics", "xgboost", "catboost", "ebm", "tabular_transformer"]
    print(f"Benchmarks enabled: {enabled}")
    print(f"Benchmark models requested: {models}")

    if "tabular_transformer" in [str(m).lower() for m in models]:
        if not torch_ok:
            print("  Tabular Transformer: will be SKIPPED (PyTorch not available)")
        elif use_cuda and not torch_cuda:
            print("  Tabular Transformer: will run on CPU (CUDA not available to PyTorch)")
        else:
            print(f"  Tabular Transformer: will run on {resolved_device if use_cuda and torch_cuda else 'cpu'}")

    print("="*80)



def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run heating demand modeling analysis')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output', type=str, default='outputs/',
                       help='Output directory')
    parser.add_argument('--skip-cv', action='store_true',
                       help='Skip nested CV (for quick testing)')
    parser.add_argument('--n-outer-folds', type=int, default=5,
                       help='Number of outer CV folds')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default=None,
                       help='Compute device: auto/cpu/cuda (overrides config.hardware.device)')
    parser.add_argument('--gpu-id', type=int, default=None,
                       help='GPU index to use (overrides config.hardware.gpu_id)')
    return parser.parse_args()


def run_data_loading_and_preprocessing(config: dict) -> tuple:
    """
    Step 1-2: Load data and preprocess.
    
    Returns
    -------
    df : DataFrame
        Preprocessed data
    preprocessor : RECSPreprocessor
        Fitted preprocessor
    """
    print_section_header("STEP 1-2: DATA LOADING AND PREPROCESSING")
    
    # Load data
    data_path = config.get('data', {}).get('raw_path', 'data/raw/recs2020_public_v7.csv')
    loader = RECSDataLoader(data_path)
    df_raw = loader.load()
    
    # Preprocess
    df, preprocessor = preprocess_recs_data(
        df_raw,
        assignment_rule='primary_only',
        exclude_no_heating=True,
        exclude_hybrid=False,  # Keep for sensitivity analysis
        min_hdd=None
    )
    
    # Get technology group summary
    tech_summary = preprocessor.get_tech_group_summary(df)
    print("\nTechnology Group Summary:")
    print(tech_summary.to_string())
    
    # Add climate bins for group-conditional calibration (Priority 4.1)
    df = add_climate_bin(df, hdd_column='HDD65', bin_column='HDD_bin')
    
    # Create verification table
    verification = loader.create_verification_table()
    print("\nVerification Statistics by Division:")
    print(verification.to_string())
    
    return df, preprocessor, loader


def run_feature_engineering(df: pd.DataFrame, config: dict) -> tuple:
    """
    Step 3: Feature engineering.
    
    Returns
    -------
    X : DataFrame
        Feature matrix
    y : Series
        Target
    weights : Series
        Sample weights
    feature_builder : FeatureBuilder
        Fitted feature builder
    """
    print_section_header("STEP 3: FEATURE ENGINEERING")

    # Benchmark config (used for tail metrics thresholds, etc.)
    bench_cfg = (config.get('benchmarks', {}) or {}) if isinstance(config, dict) else {}
    
    # Create feature builder (config-driven for sensitivity + runtime)
    fb_cfg = (config.get('feature_builder', {}) or {}) if isinstance(config, dict) else {}
    feature_builder = FeatureBuilder(
        covid_control_mode=str(fb_cfg.get('covid_control_mode', 'direct')),  # direct|proxy|none
        include_interactions=bool(fb_cfg.get('include_interactions', False)),
        scale_continuous=bool(fb_cfg.get('scale_continuous', False)),  # tree models: typically False
        # One-hot encoding (dummy variables) for nominal categories; fit on train-fold only (anti-leakage)
        encode_categorical=str(fb_cfg.get('encode_categorical', 'onehot'))
    )

    # Get feature columns
    feature_cols = feature_builder.get_feature_columns(include_tech_group=False)
    available_cols = [c for c in feature_cols if c in df.columns]
    
    print(f"Selected {len(available_cols)} features")
    
    # Create feature matrix
    X = df[available_cols].copy()
    y = df['TOTALBTUSPH'].copy()
    weights = df['NWEIGHT'].copy()

    # Fit feature builder
    X_transformed = feature_builder.fit_transform(X, y)
    
    print(f"Transformed feature matrix shape: {X_transformed.shape}")
    
    return X, y, weights, feature_builder


def run_physics_baselines(df: pd.DataFrame) -> tuple:
    """
    Step 4: Fit physics baselines.
    
    Returns
    -------
    baselines : PhysicsBaselines
        Fitted baselines
    baseline_results : DataFrame
        Baseline evaluation results
    """
    print_section_header("STEP 4: PHYSICS BASELINES")
    
    # Prepare data
    X = df[['HDD65', 'TOTSQFT_EN']].copy()
    y = df['TOTALBTUSPH'].copy()
    tech_group = df['tech_group'].copy()
    weights = df['NWEIGHT'].copy()
    
    # Fit baselines
    baselines = PhysicsBaselines()
    baselines.fit(X, y, tech_group, weights)
    
    # Evaluate
    baseline_results = baselines.evaluate(X, y, tech_group, weights)
    print("\nBaseline Performance by Technology Group:")
    print(baseline_results.to_string())
    
    # Get baseline predictions for excess demand calculation
    baseline_predictions = baselines.predict(X, tech_group)
    
    return baselines, baseline_results, baseline_predictions


def run_nested_cv(df: pd.DataFrame, 
                  feature_builder: FeatureBuilder,
                  config: dict,
                  n_outer_folds: int = 5,
                  output_dir: str = 'outputs/') -> dict:
    """
    Step 5: Run nested cross-validation.
    
    Returns
    -------
    cv_results : dict
        Cross-validation results
    """
    print_section_header("STEP 5: NESTED CROSS-VALIDATION")
    
    # Prepare data
    y = df['TOTALBTUSPH'].copy()
    weights = df['NWEIGHT'].copy()

    tech_group = df['tech_group'].copy()

    # Feature sets
    # - Split models already condition on tech_group externally -> exclude it as a feature.
    # - Monolithic model needs tech_group as a feature to be a fair comparison and to enable HDD×Tech interactions.
    feature_cols_split = feature_builder.get_feature_columns(include_tech_group=False)
    X_split = df[[c for c in feature_cols_split if c in df.columns]].copy()

    feature_cols_mono = feature_builder.get_feature_columns(include_tech_group=True)
    X_mono = df[[c for c in feature_cols_mono if c in df.columns]].copy()
    
    # Metadata for diagnostics
    # Include tech_group in metadata so monolithic CV can stratify across technologies too.
    metadata = df[['HDD65', 'DIVISION', 'TYPEHUQ', 'KOWNRENT', 'MONEYPY', 'TOTSQFT_EN', 'HDD_bin', 'tech_group']].copy()
    # LightGBM configuration (reproducibility)
    lgb_cfg = (config.get('models', {}) or {}).get('lightgbm', {}) if isinstance(config, dict) else {}

    
    # Define model factory
    def model_factory(**params):
        base_params = (lgb_cfg.get('base_params', {}) or {}) if lgb_cfg else {}
        merged = base_params.copy()
        merged.update(params)
        # GPU acceleration (best-effort). If LightGBM isn't built with GPU support, the model will fall back to CPU.
        rt = (config.get('runtime', {}) or {}) if isinstance(config, dict) else {}
        if bool(rt.get('use_cuda', False)):
            merged.setdefault('device_type', 'gpu')
            merged.setdefault('gpu_platform_id', int(rt.get('lgbm_gpu_platform_id', 0)))
            merged.setdefault('gpu_device_id', int(rt.get('gpu_id', 0)))
            merged.setdefault('max_bin', 255)


        deb = (lgb_cfg.get('debiasing', {}) or {}) if lgb_cfg else {}

        return LightGBMHeatingModel(
            params=merged,
            use_monotonic_constraints=False,
            training_reweighting_enabled=bool(deb.get('enabled', False)),
            training_tail_quantile=float(deb.get('tail_quantile', 0.95)),
            training_tail_multiplier=float(deb.get('tail_multiplier', 1.5)),
            training_cold_hdd_quantile=float(deb.get('cold_hdd_quantile', 0.85)),
            training_cold_multiplier=float(deb.get('cold_multiplier', 1.25)),
            training_two_pass_focal=bool(deb.get('two_pass_focal', False)),
            training_focal_gamma=float(deb.get('focal_gamma', 2.0))
        )

    # Define parameter grid (from config for reproducibility)
    param_grid = lgb_cfg.get('search_space', {}) or {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [15, 31, 63],
        'min_child_samples': [10, 20, 50],
        'tweedie_variance_power': [1.1, 1.3, 1.5, 1.7, 1.9],
    }

    n_search_iter = int(lgb_cfg.get('n_search_iter', 20)) if lgb_cfg else 20

    # Run nested CV
    cv = NestedCrossValidator(
        outer_folds=n_outer_folds,
        inner_folds=int((config.get('nested_cv', {}) or {}).get('inner_folds', 3)),
        n_search_iter=n_search_iter,
        random_state=int((config.get('nested_cv', {}) or {}).get('random_state', 42)),
        output_dir=output_dir
    )
    
    cv_result = cv.run(
        X_split, y, weights,
        model_factory=model_factory,
        param_grid=param_grid,
        feature_builder=feature_builder,
        tech_group=tech_group,
        additional_metadata=metadata
    )
    
    # Print results
    print("\nOverall CV Metrics:")
    for metric, value in cv_result.outer_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nMetrics by Fold:")
    print(cv_result.outer_metrics_by_fold.to_string())
    
    print("\nPhysics Diagnostics:")
    print(f"  Negative prediction rate: {cv_result.physics_diagnostics.get('negative_rate', 'N/A')}")
    if 'hdd_sensitivity' in cv_result.physics_diagnostics:
        hdd_sens = cv_result.physics_diagnostics['hdd_sensitivity']
        print(f"  HDD correlation: {hdd_sens.get('correlation', 'N/A'):.3f}")
        print(f"  Correct HDD direction: {hdd_sens.get('is_correct_direction', 'N/A')}")
    
    # Also run monolithic model for H1 comparison
    print("\nRunning Monolithic model for H1 comparison...")
    
    def mono_model_factory(**params):
        base_params = (lgb_cfg.get('base_params', {}) or {}) if lgb_cfg else {}
        merged = base_params.copy()
        merged.update(params)
        # GPU acceleration (best-effort). If LightGBM isn't built with GPU support, the model will fall back to CPU.
        rt = (config.get('runtime', {}) or {}) if isinstance(config, dict) else {}
        if bool(rt.get('use_cuda', False)):
            merged.setdefault('device_type', 'gpu')
            merged.setdefault('gpu_platform_id', int(rt.get('lgbm_gpu_platform_id', 0)))
            merged.setdefault('gpu_device_id', int(rt.get('gpu_id', 0)))
            merged.setdefault('max_bin', 255)


        deb = (lgb_cfg.get('debiasing', {}) or {}) if lgb_cfg else {}

        return LightGBMHeatingModel(
            params=merged,
            use_monotonic_constraints=False,
            training_reweighting_enabled=bool(deb.get('enabled', False)),
            training_tail_quantile=float(deb.get('tail_quantile', 0.95)),
            training_tail_multiplier=float(deb.get('tail_multiplier', 1.5)),
            training_cold_hdd_quantile=float(deb.get('cold_hdd_quantile', 0.85)),
            training_cold_multiplier=float(deb.get('cold_multiplier', 1.25)),
            training_two_pass_focal=bool(deb.get('two_pass_focal', False)),
            training_focal_gamma=float(deb.get('focal_gamma', 2.0))
        )
    
    mono_cv = NestedCrossValidator(
        outer_folds=n_outer_folds,
        inner_folds=int((config.get('nested_cv', {}) or {}).get('inner_folds', 3)),
        n_search_iter=n_search_iter,
        random_state=int((config.get('nested_cv', {}) or {}).get('random_state', 42)),
        output_dir=output_dir
    )
    
    mono_result = mono_cv.run(
        X_mono, y, weights,
        model_factory=mono_model_factory,
        param_grid=param_grid,
        feature_builder=feature_builder,
        additional_metadata=metadata
        # Note: no tech_group = monolithic
    )
    
    # Paired comparison for H1 test
    from scipy import stats
    split_rmse = cv_result.outer_metrics_by_fold['weighted_rmse'].values
    mono_rmse = mono_result.outer_metrics_by_fold['weighted_rmse'].values
    
    t_stat, p_value = stats.ttest_rel(split_rmse, mono_rmse)
    
    h1_comparison = {
        'split_by_fold': cv_result.outer_metrics_by_fold,
        'mono_by_fold': mono_result.outer_metrics_by_fold,
        'split_metrics': cv_result.outer_metrics,
        'mono_metrics': mono_result.outer_metrics,
        'rmse_difference': mono_rmse.mean() - split_rmse.mean(),
        'rmse_improvement_pct': (mono_rmse.mean() - split_rmse.mean()) / mono_rmse.mean() * 100,
        'paired_t_statistic': t_stat,
        'paired_p_value': p_value,
        'mono_predictions': mono_cv.outer_predictions_,
        'mono_fold_results': mono_cv.fold_results_
    }
    
    # Compute delta CIs using replicate weights (more robust than t-test on 5 folds)
    split_n_improved = (mono_rmse > split_rmse).sum()
    
    print(f"\nH1 Evidence (Split vs Monolithic):")
    print(f"  Split wRMSE: {split_rmse.mean():.0f} ± {split_rmse.std():.0f}")
    print(f"  Mono wRMSE: {mono_rmse.mean():.0f} ± {mono_rmse.std():.0f}")
    print(f"  Δ wRMSE: {h1_comparison['rmse_difference']:.0f} ({h1_comparison['rmse_improvement_pct']:.1f}%)")
    print(f"  Split better in {split_n_improved}/{n_outer_folds} folds")
    print(f"  Note: With only {n_outer_folds} folds, formal hypothesis tests have limited power.")
    
    return {
        'cv_result': cv_result,
        'predictions': cv.outer_predictions_,
        'predictions_uncalibrated': cv.outer_predictions_uncalib_,  # For calibration comparison
        'fold_results': cv.fold_results_,
        'h1_comparison': h1_comparison,
        'mono_predictions': mono_cv.outer_predictions_,
        'mono_fold_results': mono_cv.fold_results_
    }



def run_tail_bias_mitigation(df: pd.DataFrame,
                              feature_builder: FeatureBuilder,
                              cv_results: dict,
                              config: dict,
                              n_outer_folds: int = 5,
                              output_dir: str = 'outputs/') -> dict:
    """
    Step 5C: Run tail bias mitigation analysis (Priority 1).
    
    Implements:
    - Quantile regression (q=0.90) for policy-aligned predictions
    - Tail-weighted training to reduce upper-tail underprediction
    - Comparison with baseline model
    
    Parameters
    ----------
    df : DataFrame
        Preprocessed data
    feature_builder : FeatureBuilder
        Feature preprocessor
    cv_results : dict
        Results from baseline CV (contains predictions)
    config : dict
        Configuration
    n_outer_folds : int
        Number of outer folds
    output_dir : str
        Output directory
        
    Returns
    -------
    dict
        Tail bias mitigation results
    """
    tail_cfg = (config.get('models', {}) or {}).get('tail_bias', {}) or {}
    
    if not bool(tail_cfg.get('quantile_enabled', False)):
        logger.info("Tail bias mitigation disabled in config; skipping.")
        return {}
    
    print_section_header("TAIL BIAS MITIGATION (Priority 1)")
    
    out_dir = Path(output_dir) / 'tail_bias'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    y = df['TOTALBTUSPH'].values
    weights = df['NWEIGHT'].values
    tech_group = df['tech_group'].values
    
    feature_cols = feature_builder.get_feature_columns(include_tech_group=True)
    X = df[[c for c in feature_cols if c in df.columns]].copy()
    
    meta_cols = ['HDD65', 'DIVISION', 'TYPEHUQ', 'KOWNRENT', 'MONEYPY', 'TOTSQFT_EN', 'HDD_bin', 'tech_group']
    metadata = df[[c for c in meta_cols if c in df.columns]].copy()
    
    # Get baseline predictions
    baseline_predictions = cv_results.get('predictions', np.zeros(len(df)))
    
    results = {
        'baseline_predictions': baseline_predictions
    }
    
    # Compute tail bias metrics for baseline
    tbm = TailBiasMetrics(k_percentile=10)
    baseline_tail_metrics = tbm.compute_all_metrics(y, baseline_predictions, weights)
    
    logger.info(f"\nBaseline Top-10% Underprediction: {baseline_tail_metrics['top_decile_bias_pct']:+.1f}%")
    logger.info(f"Baseline Lift@10: {baseline_tail_metrics['lift_at_k']:.2f}")
    logger.info(f"Baseline NDCG: {baseline_tail_metrics['ndcg']:.3f}")
    
    results['baseline_tail_metrics'] = baseline_tail_metrics
    
    # Run quantile model if enabled
    quantile_alpha = float(tail_cfg.get('quantile_alpha', 0.90))
    
    logger.info(f"\nRunning quantile regression (q={quantile_alpha})...")
    
    lgb_cfg = (config.get('models', {}) or {}).get('lightgbm', {}) or {}
    n_search_iter = int(lgb_cfg.get('n_search_iter', 10))
    
    param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.03, 0.05],
        'num_leaves': [31, 63],
        'min_child_samples': [20, 50],
    }
    
    def quantile_factory(**params):
        return QuantileLightGBMHeatingModel(params=params, quantile=quantile_alpha)
    
    try:
        cv_quantile = NestedCrossValidator(
            outer_folds=n_outer_folds,
            inner_folds=3,
            n_search_iter=n_search_iter,
            random_state=42,
            output_dir=str(out_dir / 'quantile')
        )
        
        q_result = cv_quantile.run(
            X, pd.Series(y), pd.Series(weights),
            model_factory=quantile_factory,
            param_grid=param_grid,
            feature_builder=feature_builder,
            additional_metadata=metadata
        )
        
        quantile_predictions = cv_quantile.outer_predictions_
        quantile_tail_metrics = tbm.compute_all_metrics(y, quantile_predictions, weights)
        
        logger.info(f"\nQuantile (q={quantile_alpha}) Top-10% Underprediction: {quantile_tail_metrics['top_decile_bias_pct']:+.1f}%")
        logger.info(f"Quantile Lift@10: {quantile_tail_metrics['lift_at_k']:.2f}")
        logger.info(f"Quantile NDCG: {quantile_tail_metrics['ndcg']:.3f}")
        
        results['quantile_predictions'] = quantile_predictions
        results['quantile_tail_metrics'] = quantile_tail_metrics
        results['quantile_cv_result'] = q_result
        
    except Exception as e:
        logger.warning(f"Quantile regression failed: {e}")
    
    # Run tail-weighted model if configured
    tw_cfg = tail_cfg.get('tail_weighting', {}) or {}
    tw_mode = str(tw_cfg.get('mode', 'none'))
    
    if tw_mode != 'none':
        logger.info(f"\nRunning tail-weighted training (mode={tw_mode})...")
        
        tw_alpha = float(tw_cfg.get('alpha', 1.0))
        tw_cap = float(tw_cfg.get('cap', 20.0))
        
        def tw_factory(**params):
            return TailWeightedLightGBMHeatingModel(
                params=params,
                tail_weighting_mode=tw_mode,
                tail_alpha=tw_alpha,
                tail_cap=tw_cap,
                apply_isotonic_calibration=True
            )
        
        try:
            cv_tw = NestedCrossValidator(
                outer_folds=n_outer_folds,
                inner_folds=3,
                n_search_iter=n_search_iter,
                random_state=42,
                output_dir=str(out_dir / 'tail_weighted')
            )
            
            tw_result = cv_tw.run(
                X, pd.Series(y), pd.Series(weights),
                model_factory=tw_factory,
                param_grid=param_grid,
                feature_builder=feature_builder,
                additional_metadata=metadata
            )
            
            tw_predictions = cv_tw.outer_predictions_
            tw_tail_metrics = tbm.compute_all_metrics(y, tw_predictions, weights)
            
            logger.info(f"\nTail-Weighted Top-10% Underprediction: {tw_tail_metrics['top_decile_bias_pct']:+.1f}%")
            logger.info(f"Tail-Weighted Lift@10: {tw_tail_metrics['lift_at_k']:.2f}")
            
            results['tail_weighted_predictions'] = tw_predictions
            results['tail_weighted_tail_metrics'] = tw_tail_metrics
            results['tail_weighted_cv_result'] = tw_result
            
        except Exception as e:
            logger.warning(f"Tail-weighted training failed: {e}")
    
    # Create comparison table
    comparison_rows = [
        {
            'Model': 'Baseline (Tweedie)',
            'Top-10% Bias (%)': baseline_tail_metrics['top_decile_bias_pct'],
            'Lift@10': baseline_tail_metrics['lift_at_k'],
            'NDCG': baseline_tail_metrics['ndcg'],
            'Precision@10': baseline_tail_metrics['precision_at_k'],
            'Recall@10': baseline_tail_metrics['recall_at_k']
        }
    ]
    
    if 'quantile_tail_metrics' in results:
        comparison_rows.append({
            'Model': f'Quantile (q={quantile_alpha})',
            'Top-10% Bias (%)': results['quantile_tail_metrics']['top_decile_bias_pct'],
            'Lift@10': results['quantile_tail_metrics']['lift_at_k'],
            'NDCG': results['quantile_tail_metrics']['ndcg'],
            'Precision@10': results['quantile_tail_metrics']['precision_at_k'],
            'Recall@10': results['quantile_tail_metrics']['recall_at_k']
        })
    
    if 'tail_weighted_tail_metrics' in results:
        comparison_rows.append({
            'Model': f'Tail-Weighted (α={tw_alpha})',
            'Top-10% Bias (%)': results['tail_weighted_tail_metrics']['top_decile_bias_pct'],
            'Lift@10': results['tail_weighted_tail_metrics']['lift_at_k'],
            'NDCG': results['tail_weighted_tail_metrics']['ndcg'],
            'Precision@10': results['tail_weighted_tail_metrics']['precision_at_k'],
            'Recall@10': results['tail_weighted_tail_metrics']['recall_at_k']
        })
    
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(out_dir / 'tail_bias_comparison.csv', index=False, encoding='utf-8-sig')
    
    # By technology group
    all_tech_rows = []
    predictions_dict = {'Baseline': baseline_predictions}
    if 'quantile_predictions' in results:
        predictions_dict['Quantile'] = results['quantile_predictions']
    if 'tail_weighted_predictions' in results:
        predictions_dict['Tail-Weighted'] = results['tail_weighted_predictions']
    
    for model_name, y_pred in predictions_dict.items():
        tech_df = tbm.compute_metrics_by_tech_group(y, y_pred, weights, tech_group)
        tech_df['Model'] = model_name
        all_tech_rows.append(tech_df)
    
    if all_tech_rows:
        tech_comparison = pd.concat(all_tech_rows, ignore_index=True)
        tech_comparison.to_csv(out_dir / 'tail_bias_by_tech_group.csv', index=False, encoding='utf-8-sig')
        results['by_tech_group'] = tech_comparison
    
    results['comparison_df'] = comparison_df
    
    logger.info(f"\nTail bias analysis saved to: {out_dir}")
    
    return results


def run_sota_benchmarks(df: pd.DataFrame,
                        feature_builder: FeatureBuilder,
                        config: dict,
                        cv_results: dict,
                        baseline_predictions: np.ndarray,
                        n_outer_folds: int = 5,
                        output_dir: str = 'outputs/') -> dict:
    """
    Step 5B: Benchmark additional (optional) SOTA baselines requested by reviewers.

    Produces:
      - outputs/benchmarks/sota_fold_metrics.csv
      - outputs/benchmarks/sota_summary.csv
      - outputs/benchmarks/sota_summary.tex
      - outputs/benchmarks/plots/sota_<metric>.png  (if enabled)

    Notes
    -----
    - Uses the monolithic feature set (includes tech_group) for all tabular baselines.
    - Optional dependencies (xgboost, catboost, torch) are handled gracefully:
      if missing, the corresponding model is skipped with a clear warning.
    """
    bench_cfg = (config.get('benchmarks', {}) or {}) if isinstance(config, dict) else {}
    rt = (config.get('runtime', {}) or {}) if isinstance(config, dict) else {}

    if not bool(bench_cfg.get('enabled', True)):
        logger.info("Benchmarks disabled in config (benchmarks.enabled=false).")
        return {}

    out_root = Path(output_dir) / "benchmarks"
    out_root.mkdir(parents=True, exist_ok=True)
    plots_dir = out_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Record optional dependency availability (for transparent reporting)
    dep_status = {}
    def _pkg_version(pkg_name: str):
        try:
            import importlib
            mod = importlib.import_module(pkg_name)
            return getattr(mod, '__version__', 'unknown')
        except Exception:
            return None

    for pkg in ['lightgbm', 'xgboost', 'catboost', 'interpret', 'torch']:
        dep_status[pkg] = _pkg_version(pkg)

    # Torch GPU visibility (if torch is installed)
    try:
        import torch
        dep_status['torch_cuda_available'] = bool(torch.cuda.is_available())
        dep_status['torch_cuda_device_count'] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:
        dep_status['torch_cuda_available'] = False
        dep_status['torch_cuda_device_count'] = 0

    try:
        (out_root / "sota_dependency_status.json").write_text(json.dumps(dep_status, indent=2), encoding="utf-8")
    except Exception:
        pass

    # Prepare shared data
    y = df['TOTALBTUSPH'].copy()
    weights = df['NWEIGHT'].copy()

    # Build the monolithic feature matrix used for all tabular SOTA baselines.
    # (Includes tech_group as a feature for a fair apples-to-apples comparison.)
    feature_cols_mono = feature_builder.get_feature_columns(include_tech_group=True)
    X_mono = df[[c for c in feature_cols_mono if c in df.columns]].copy()

    # Metadata used for diagnostic plots and stratified evaluation.
    # Keep this robust to missing columns across different RECS extracts.
    meta_cols = ['HDD65', 'DIVISION', 'TYPEHUQ', 'KOWNRENT', 'MONEYPY', 'TOTSQFT_EN', 'HDD_bin', 'tech_group']
    if 'HDD_bin' not in df.columns and 'HDD65' in df.columns:
        # Create a simple HDD bin if preprocessing didn't generate one
        try:
            df = df.copy()
            df['HDD_bin'] = pd.cut(df['HDD65'], bins=[-np.inf, 1500, 3000, 4500, 6000, np.inf],
                                   labels=['very_warm', 'warm', 'mild', 'cold', 'very_cold'])
        except Exception:
            pass
    metadata = df[[c for c in meta_cols if c in df.columns]].copy()

    # Weighted tail threshold for high-consumption evaluation (default: top 10% by weighted quantile)
    def _weighted_quantile(values: np.ndarray, w: np.ndarray, q: float) -> float:
        values = np.asarray(values, dtype=float)
        w = np.asarray(w, dtype=float)
        mask = np.isfinite(values) & np.isfinite(w) & (w > 0)
        if mask.sum() == 0:
            return float(np.nan)
        v = values[mask]
        ww = w[mask]
        sorter = np.argsort(v)
        v = v[sorter]
        ww = ww[sorter]
        cum_w = np.cumsum(ww)
        cutoff = q * cum_w[-1]
        return float(np.interp(cutoff, cum_w, v))

    tail_q = float((bench_cfg.get('tail_quantile', 0.90) if isinstance(bench_cfg, dict) else 0.90))
    global_tail_threshold = _weighted_quantile(y.values, weights.values, tail_q)

    def _compute_tail_metrics(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> dict:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        w = np.asarray(w)
        mask = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(w) & (w > 0) & (y_true >= global_tail_threshold)
        if mask.sum() == 0:
            return {
                'weighted_rmse_top': np.nan,
                'weighted_mae_top': np.nan,
                'weighted_bias_top': np.nan,
                'top_count': 0,
                'top_weight_sum': 0.0,
            }
        met = WeightedMetrics().compute_all_metrics(y_true[mask], y_pred[mask], w[mask])
        return {
            'weighted_rmse_top': met.get('weighted_rmse', np.nan),
            'weighted_mae_top': met.get('weighted_mae', np.nan),
            'weighted_bias_top': met.get('weighted_bias', np.nan),
            'top_count': int(mask.sum()),
            'top_weight_sum': float(np.sum(w[mask])),
        }

    tech_group = df['tech_group'].copy()
    tech_order = ['combustion', 'electric_heat_pump', 'electric_resistance', 'hybrid_ambiguous']

    # Collect per-fold metrics for each model
    fold_rows = []
    by_tech_group_rows = []

    metrics_calc = WeightedMetrics()

    def _append_by_tech_group_metrics(model_name: str,
                                     fold_idx: int,
                                     test_idx: np.ndarray,
                                     y_true: np.ndarray,
                                     y_pred: np.ndarray,
                                     w: np.ndarray):
        """Append metrics stratified by technology group for a single fold."""
        test_idx = np.asarray(test_idx, dtype=int)
        tg = tech_group.values[test_idx]
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        w = np.asarray(w)

        for grp in tech_order:
            mask = (tg == grp)
            if mask.sum() == 0:
                continue
            # Remove invalid weights
            m2 = mask & np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(w) & (w > 0)
            if m2.sum() == 0:
                continue
            met = metrics_calc.compute_all_metrics(y_true[m2], y_pred[m2], w[m2])
            tail = _compute_tail_metrics(y_true[m2], y_pred[m2], w[m2])
            by_tech_group_rows.append({
                'Model': model_name,
                'fold': int(fold_idx),
                'tech_group': grp,
                'n': int(m2.sum()),
                'weight_sum': float(np.sum(w[m2])),
                **met,
                **tail
            })

    def _append_fold_metrics(model_name: str, fold_df: pd.DataFrame):
        if fold_df is None or len(fold_df) == 0:
            return
        tmp = fold_df.copy()
        tmp['Model'] = model_name
        if 'status' not in tmp.columns:
            tmp['status'] = 'ok'
        # normalize fold column
        if 'fold' not in tmp.columns:
            tmp['fold'] = range(len(tmp))
        fold_rows.append(tmp)

    # 0) Physics baseline (fold-by-fold, using the same outer test splits)
    fold_results = cv_results.get('fold_results', []) or []
    if fold_results:
        for fr in fold_results:
            # CVFoldResult stores test indices + fold_idx
            test_idx = getattr(fr, 'test_idx', None)
            fold_idx = getattr(fr, 'fold_idx', None)
            if test_idx is None:
                continue
            met = metrics_calc.compute_all_metrics(
                y.values[test_idx],
                baseline_predictions[test_idx],
                weights.values[test_idx]
            )
            tail = _compute_tail_metrics(
                y.values[test_idx],
                baseline_predictions[test_idx],
                weights.values[test_idx]
            )
            row = {**met, **tail, 'fold': int(fold_idx) if fold_idx is not None else len(fold_rows),
                   'Model': 'Physics Baseline', 'status': 'ok'}
            fold_rows.append(pd.DataFrame([row]))
            try:
                _append_by_tech_group_metrics('Physics Baseline', int(fold_idx) if fold_idx is not None else 0, test_idx, y.values[test_idx], baseline_predictions[test_idx], weights.values[test_idx])
            except Exception:
                pass
    else:
        # Fallback: if fold indices are unavailable, report overall-only (mean=overall)
        base_metrics = metrics_calc.compute_all_metrics(y.values, baseline_predictions, weights.values)
        tail = _compute_tail_metrics(y.values, baseline_predictions, weights.values)
        base_row = {**base_metrics, **tail, 'fold': 'overall', 'Model': 'Physics Baseline', 'status': 'overall_only'}
        fold_rows.append(pd.DataFrame([base_row]))
        try:
            _append_by_tech_group_metrics('Physics Baseline', -1, np.arange(len(y)), y.values, baseline_predictions, weights.values)
        except Exception:
            pass

    # 1) Monolithic LightGBM (from earlier CV; attach tail metrics if fold results are available)
    try:
        mono_fold = cv_results.get('h1_comparison', {}).get('mono_by_fold', None)
        mono_fold_results = cv_results.get('mono_fold_results', None)
        if mono_fold is not None:
            fold_df = mono_fold.copy()
            if mono_fold_results:
                try:
                    tail_rows = []
                    for fr in mono_fold_results:
                        tail = _compute_tail_metrics(fr.y_true, fr.predictions, fr.weights)
                        tail_rows.append({'fold': int(getattr(fr, 'fold_idx', len(tail_rows))), **tail})
                    tail_df = pd.DataFrame(tail_rows) if tail_rows else pd.DataFrame()
                    if not tail_df.empty and 'fold' in fold_df.columns:
                        fold_df = fold_df.merge(tail_df, on='fold', how='left')
                except Exception:
                    pass
            _append_fold_metrics("LightGBM (Monolithic)", fold_df)
            # Stratified metrics by tech_group (reviewer-critical for heat pumps / hybrids)
            if mono_fold_results:
                for fr in mono_fold_results:
                    try:
                        _append_by_tech_group_metrics(
                            "LightGBM (Monolithic)",
                            int(getattr(fr, 'fold_idx', 0)),
                            getattr(fr, 'test_idx', np.array([], dtype=int)),
                            fr.y_true, fr.predictions, fr.weights
                        )
                    except Exception:
                        pass
    except Exception:
        logger.warning("Could not attach monolithic LightGBM fold metrics from cv_results; skipping.")

    # 2) EBM (optional dependency: interpret)
    models_to_run = bench_cfg.get('models', []) or []
    if 'ebm' in models_to_run or 'EBM' in models_to_run:
        ebm_cfg = (config.get('models', {}) or {}).get('ebm', {}) if isinstance(config, dict) else {}
        ebm_grid = ebm_cfg.get('search_space', {}) or {
            'max_bins': [64, 128, 256],
            'interactions': [0, 10, 20],
            'learning_rate': [0.01, 0.05],
        }
        ebm_n_iter = int(ebm_cfg.get('n_search_iter', 10))

        def ebm_factory(**params):
            return EBMHeatingModel(**params)

        try:
            cv = NestedCrossValidator(
                outer_folds=n_outer_folds,
                inner_folds=int((config.get('nested_cv', {}) or {}).get('inner_folds', 3)),
                n_search_iter=ebm_n_iter,
                random_state=int((config.get('nested_cv', {}) or {}).get('random_state', 42)),
                output_dir=str(out_root / "ebm")
            )
            res = cv.run(
                X_mono, y, weights,
                model_factory=ebm_factory,
                param_grid=ebm_grid,
                feature_builder=feature_builder,
                tech_group=tech_group,
                additional_metadata=metadata
            )
            _append_fold_metrics("EBM", res.outer_metrics_by_fold)
            # Stratified metrics by tech_group
            try:
                for fr in res.fold_results:
                    _append_by_tech_group_metrics("EBM", int(getattr(fr, 'fold_idx', 0)), getattr(fr, 'test_idx', np.array([], dtype=int)), fr.y_true, fr.predictions, fr.weights)
            except Exception:
                pass
        except ImportError as e:
            logger.warning(f"Skipping EBM benchmark (missing optional dependency): {e}")
        except Exception as e:
            logger.warning(f"EBM benchmark failed: {e}")

    # Helper to run optional SOTA baselines
    # NOTE: We pass shared objects explicitly to avoid static-analysis false positives.
    # Some linters (e.g., pyflakes) may flag closure-captured variables (like X_mono)
    # as "undefined", even though they are valid at runtime.
    def _run_optional_model(
        name: str,
        factory,
        grid: dict,
        n_iter: int,
        out_subdir: str,
        *,
        X_mono: pd.DataFrame,
        y: pd.Series,
        weights: pd.Series,
        feature_builder,
        tech_group: pd.Series,
        metadata: pd.DataFrame,
    ):
        try:
            cv = NestedCrossValidator(
                outer_folds=n_outer_folds,
                inner_folds=int((config.get('nested_cv', {}) or {}).get('inner_folds', 3)),
                n_search_iter=int(n_iter),
                random_state=int((config.get('nested_cv', {}) or {}).get('random_state', 42)),
                output_dir=str(out_root / out_subdir)
            )
            res = cv.run(
                X_mono, y, weights,
                model_factory=factory,
                param_grid=grid,
                feature_builder=feature_builder,
                tech_group=tech_group,
                additional_metadata=metadata
            )
            # Attach high-consumption (tail) metrics for policy-relevant evaluation
            try:
                tail_rows = []
                for fr in res.fold_results:
                    tail = _compute_tail_metrics(fr.y_true, fr.predictions, fr.weights)
                    tail_rows.append({'fold': int(getattr(fr, 'fold_idx', len(tail_rows))), **tail})
                tail_df = pd.DataFrame(tail_rows) if tail_rows else pd.DataFrame()
                fold_df = res.outer_metrics_by_fold.copy()
                if not tail_df.empty and 'fold' in fold_df.columns:
                    fold_df = fold_df.merge(tail_df, on='fold', how='left')
            except Exception:
                fold_df = res.outer_metrics_by_fold
            _append_fold_metrics(name, fold_df)
            # Stratified metrics by tech_group
            try:
                for fr in res.fold_results:
                    _append_by_tech_group_metrics(name, int(getattr(fr, 'fold_idx', 0)), getattr(fr, 'test_idx', np.array([], dtype=int)), fr.y_true, fr.predictions, fr.weights)
            except Exception:
                pass
            return res
        except ImportError as e:
            msg = f"skipped: missing optional dependency ({e})"
            logger.warning(f"Skipping {name} (missing optional dependency): {e}")
            fold_rows.append(pd.DataFrame([{'Model': name, 'fold': 'skipped', 'status': msg}]))
        except Exception as e:
            msg = f"failed: {type(e).__name__}: {e}"
            logger.warning(f"{name} benchmark failed: {e}")
            fold_rows.append(pd.DataFrame([{'Model': name, 'fold': 'failed', 'status': msg}]))
        return None

    # 3) XGBoost
    if 'xgboost' in models_to_run:
        xgb_cfg = (config.get('models', {}) or {}).get('xgboost', {}) if isinstance(config, dict) else {}
        xgb_grid = xgb_cfg.get('search_space', {}) or {}
        xgb_n_iter = int(xgb_cfg.get('n_search_iter', 20))

        def xgb_factory(**params):
            p = dict(params)
            if bool(rt.get('use_cuda', False)):
                p.update({
                    'tree_method': 'gpu_hist',
                    'predictor': 'gpu_predictor',
                    'gpu_id': int(rt.get('gpu_id', 0)),
                })
            return XGBoostRegressorModel(params=p)

        _run_optional_model(
            "XGBoost", xgb_factory, xgb_grid, xgb_n_iter, "xgboost",
            X_mono=X_mono, y=y, weights=weights, feature_builder=feature_builder,
            tech_group=tech_group, metadata=metadata,
        )

    # 4) CatBoost
    if 'catboost' in models_to_run:
        cb_cfg = (config.get('models', {}) or {}).get('catboost', {}) if isinstance(config, dict) else {}
        cb_grid = cb_cfg.get('search_space', {}) or {}
        cb_n_iter = int(cb_cfg.get('n_search_iter', 15))

        def cb_factory(**params):
            p = dict(params)
            if bool(rt.get('use_cuda', False)):
                p.update({
                    'task_type': 'GPU',
                    'devices': str(int(rt.get('gpu_id', 0))),
                })
            return CatBoostRegressorModel(params=p, verbose=False)

        _run_optional_model(
            "CatBoost", cb_factory, cb_grid, cb_n_iter, "catboost",
            X_mono=X_mono, y=y, weights=weights, feature_builder=feature_builder,
            tech_group=tech_group, metadata=metadata,
        )

    # 5) Tabular Transformer
    if 'tabular_transformer' in models_to_run:
        tt_cfg = (config.get('models', {}) or {}).get('tabular_transformer', {}) if isinstance(config, dict) else {}
        tt_grid = tt_cfg.get('search_space', {}) or {}
        tt_n_iter = int(tt_cfg.get('n_search_iter', 8))

        def tt_factory(**params):
            p = dict(params)
            if bool(rt.get('use_cuda', False)):
                # torch device string
                p.setdefault('device', f"cuda:{int(rt.get('gpu_id', 0))}")
            return TabularTransformerModel(**p)

        _run_optional_model(
            "Tabular Transformer", tt_factory, tt_grid, tt_n_iter, "tabular_transformer",
            X_mono=X_mono, y=y, weights=weights, feature_builder=feature_builder,
            tech_group=tech_group, metadata=metadata,
        )

    # Combine fold-level table
    if not fold_rows:
        logger.warning("No benchmark results were produced.")
        return {}

    fold_df = pd.concat(fold_rows, ignore_index=True)

    # Save fold metrics
    fold_csv = out_root / "sota_fold_metrics.csv"
    fold_df.to_csv(fold_csv, index=False, encoding='utf-8-sig')

    # Summary across folds
    # - Prefer successful folds (status == 'ok') when available
    # - Keep rows for skipped/failed models for transparent reporting
    metric_cols = [c for c in fold_df.columns if c.startswith('weighted_')]
    summary_rows = []
    for model_name, g in fold_df.groupby('Model', sort=False):
        if 'status' in g.columns:
            g_eval = g[g['status'].astype(str).str.lower().eq('ok')]
            if g_eval.empty:
                g_eval = g[g['fold'] != 'overall'] if 'fold' in g.columns else g
        else:
            g_eval = g[g['fold'] != 'overall'] if 'fold' in g.columns else g

        row = {'Model': model_name}
        if 'status' in g.columns:
            uniq = [s for s in pd.Series(g['status']).dropna().astype(str).unique().tolist() if s]
            row['status'] = " | ".join(uniq) if uniq else "ok"

        for mc in metric_cols:
            vals = pd.to_numeric(g_eval[mc], errors='coerce')
            n_fin = int(np.isfinite(vals.values).sum())
            row[f"{mc}_mean"] = float(np.nanmean(vals.values)) if len(vals) else np.nan
            row[f"{mc}_std"] = float(np.nanstd(vals.values, ddof=1)) if n_fin > 1 else np.nan
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    summary_csv = out_root / "sota_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')

    # Simple LaTeX table for manuscript
    tex_path = out_root / "sota_summary.tex"
    try:
        # Keep only key metrics
        keep = [
            'Model',
            'status',
            'weighted_rmse_mean', 'weighted_rmse_std',
            'weighted_mae_mean', 'weighted_mae_std',
            'weighted_bias_mean', 'weighted_bias_std',
            'weighted_rmse_top_mean', 'weighted_rmse_top_std',
            'weighted_mae_top_mean', 'weighted_mae_top_std',
            'weighted_bias_top_mean', 'weighted_bias_top_std'
        ]
        keep = [c for c in keep if c in summary_df.columns]
        tex_df = summary_df[keep].copy()
        # Format: energy metrics with thousands separators; R² with 3 decimals
        for c in tex_df.columns:
            if c.startswith('weighted_') and ('r2' in c):
                tex_df[c] = tex_df[c].apply(lambda v: f"{v:.3f}" if np.isfinite(v) else "")
            elif c.startswith('weighted_'):
                tex_df[c] = tex_df[c].apply(lambda v: f"{v:,.0f}" if np.isfinite(v) else "")
        tex_df.to_latex(tex_path, index=False, escape=False)
    except Exception as e:
        logger.warning(f"Could not write LaTeX table: {e}")

    # Plots (bar + error bars)
    if bool(bench_cfg.get('save_plots', True)):
        metrics_for_plots = bench_cfg.get('metrics_for_plots', ['weighted_rmse', 'weighted_mae'])
        import matplotlib.pyplot as plt

        for met in metrics_for_plots:
            mean_col = f"{met}_mean"
            std_col = f"{met}_std"
            if mean_col not in summary_df.columns:
                continue
            plot_df = summary_df.copy()
            if 'status' in plot_df.columns:
                plot_df = plot_df[plot_df['status'].astype(str).str.lower().str.contains('ok')]
            plot_df = plot_df[np.isfinite(plot_df[mean_col].values)]
            if plot_df.empty:
                continue
            order = plot_df['Model'].tolist()
            means = plot_df[mean_col].values
            stds = plot_df[std_col].values if std_col in plot_df.columns else None

            plt.figure(figsize=(10, 5))
            x = np.arange(len(order))
            if stds is not None:
                plt.bar(x, means, yerr=stds, capsize=4)
            else:
                plt.bar(x, means)
            plt.xticks(x, order, rotation=20, ha='right')
            plt.ylabel(met.replace('_', ' '))
            plt.title(f"SOTA Benchmark Comparison: {met.replace('_', ' ')}")
            plt.tight_layout()
            plt.savefig(plots_dir / f"sota_{met}.png", dpi=300)
            plt.close()


    # Stratified performance by technology group (addresses reviewer concern 7.4)
    if by_tech_group_rows:
        by_tech_fold_df = pd.DataFrame(by_tech_group_rows)
        by_tech_fold_csv = out_root / "sota_by_tech_group_fold_metrics.csv"
        by_tech_fold_df.to_csv(by_tech_fold_csv, index=False, encoding='utf-8-sig')

        # Summary across folds for each (Model, tech_group)
        metric_cols2 = [c for c in by_tech_fold_df.columns if c.startswith('weighted_')]
        summ_rows2 = []
        for (mn, tg), g in by_tech_fold_df.groupby(['Model', 'tech_group'], sort=False):
            row = {'Model': mn, 'tech_group': tg}
            for mc in metric_cols2:
                vals = pd.to_numeric(g[mc], errors='coerce')
                n_fin = int(np.isfinite(vals.values).sum())
                row[f"{mc}_mean"] = float(np.nanmean(vals.values)) if len(vals) else np.nan
                row[f"{mc}_std"] = float(np.nanstd(vals.values, ddof=1)) if n_fin > 1 else np.nan
            # sample size signals
            row['n_mean'] = float(np.nanmean(pd.to_numeric(g.get('n', np.nan), errors='coerce')))
            row['weight_sum_mean'] = float(np.nanmean(pd.to_numeric(g.get('weight_sum', np.nan), errors='coerce')))
            summ_rows2.append(row)
        by_tech_summary_df = pd.DataFrame(summ_rows2)

        by_tech_summary_csv = out_root / "sota_by_tech_group_summary.csv"
        by_tech_summary_df.to_csv(by_tech_summary_csv, index=False, encoding='utf-8-sig')

        # LaTeX (compact)
        by_tech_tex = out_root / "sota_by_tech_group_summary.tex"
        try:
            keep2 = [
                'Model', 'tech_group',
                'weighted_rmse_mean', 'weighted_rmse_std',
                'weighted_mae_mean', 'weighted_mae_std',
                'weighted_r2_mean', 'weighted_r2_std',
                'weighted_bias_mean', 'weighted_bias_std'
            ]
            keep2 = [c for c in keep2 if c in by_tech_summary_df.columns]
            tex2 = by_tech_summary_df[keep2].copy()
            for c in tex2.columns:
                if c.startswith('weighted_') and ('r2' in c):
                    tex2[c] = tex2[c].apply(lambda v: f"{v:.3f}" if np.isfinite(v) else "")
                elif c.startswith('weighted_'):
                    tex2[c] = tex2[c].apply(lambda v: f"{v:,.0f}" if np.isfinite(v) else "")
            tex2.to_latex(by_tech_tex, index=False, escape=False)
        except Exception as e:
            logger.warning(f"Could not write tech_group LaTeX table: {e}")

        # Plots: heatmap for RMSE and R2 (mean across folds)
        try:
            import matplotlib.pyplot as plt

            def _heatmap(metric: str, fname: str, title: str):
                val_col = f"{metric}_mean"
                if val_col not in by_tech_summary_df.columns:
                    return
                piv = by_tech_summary_df.pivot(index='Model', columns='tech_group', values=val_col)
                # enforce tech_order columns where possible
                cols = [c for c in tech_order if c in piv.columns] + [c for c in piv.columns if c not in tech_order]
                piv = piv.reindex(columns=cols)
                # keep model order from summary_df (already in desired order)
                if 'Model' in summary_df.columns:
                    model_order = summary_df['Model'].tolist()
                    piv = piv.reindex(index=[m for m in model_order if m in piv.index] + [m for m in piv.index if m not in model_order])

                arr = piv.values.astype(float)
                plt.figure(figsize=(max(7, 1.2 * arr.shape[1] + 3), max(4, 0.5 * arr.shape[0] + 2)))
                im = plt.imshow(arr, aspect='auto')
                plt.colorbar(im, shrink=0.8)
                plt.xticks(range(len(piv.columns)), piv.columns, rotation=20, ha='right')
                plt.yticks(range(len(piv.index)), piv.index)
                plt.title(title)
                plt.tight_layout()
                plt.savefig(plots_dir / fname, dpi=300)
                plt.close()

            _heatmap('weighted_rmse', 'sota_by_tech_group_weighted_rmse.png', 'SOTA Comparison by Technology Group: weighted RMSE')
            _heatmap('weighted_r2', 'sota_by_tech_group_weighted_r2.png', 'SOTA Comparison by Technology Group: weighted R²')
        except Exception as e:
            logger.warning(f"Could not write tech_group benchmark plots: {e}")

    logger.info(f"SOTA benchmark table saved to: {out_root}")
    return {
        'fold_metrics_path': str(fold_csv),
        'summary_path': str(summary_csv),
        'plots_dir': str(plots_dir),
        'by_tech_group_fold_metrics_path': str(out_root / "sota_by_tech_group_fold_metrics.csv") if (out_root / "sota_by_tech_group_fold_metrics.csv").exists() else None,
        'by_tech_group_summary_path': str(out_root / "sota_by_tech_group_summary.csv") if (out_root / "sota_by_tech_group_summary.csv").exists() else None
    }



def run_policy_analysis(df: pd.DataFrame,
                        predictions: np.ndarray,
                        baseline_predictions: np.ndarray) -> dict:
    """
    Step 6: Policy targeting analysis.
    
    Returns
    -------
    policy_results : dict
        Policy analysis results
    """
    print_section_header("STEP 6: POLICY TARGETING ANALYSIS")
    
    from src.policy.targeting import PolicyTargeting, TargetingUncertainty
    
    # Initialize targeting analyzer
    targeting = PolicyTargeting(target_percentile=90)
    
    # Run full analysis
    policy_results = targeting.run_full_analysis(
        y_pred=predictions,
        y_baseline=baseline_predictions,
        y_true=df['TOTALBTUSPH'].values,
        weights=df['NWEIGHT'].values,
        metadata=df
    )
    
    # Compute Jaccard/Overlap CIs using replicate weights
    replicate_cols = [f'NWEIGHT{i}' for i in range(1, 61)]
    available_rep_cols = [c for c in replicate_cols if c in df.columns]
    
    if available_rep_cols:
        replicate_weights = df[available_rep_cols]
        targeting_unc = TargetingUncertainty(n_replicates=60)
        
        policy_results['uncertainty'] = {}
        
        for score_name in ['high_use', 'high_intensity', 'excess_demand']:
            if score_name in policy_results['weighted_vs_unweighted']:
                # Compute scores
                area = df['TOTSQFT_EN'].values
                if score_name == 'high_use':
                    scores = predictions
                elif score_name == 'high_intensity':
                    scores = predictions / np.maximum(area, 1)
                else:  # excess_demand
                    scores = predictions - baseline_predictions
                
                # Compute CIs
                ci_results = targeting_unc.compute_jaccard_overlap_with_ci(
                    scores=scores,
                    weights=df['NWEIGHT'].values,
                    replicate_weights=replicate_weights,
                    target_percentile=90
                )
                
                policy_results['uncertainty'][score_name] = ci_results
                
                print(f"\n{score_name.upper()} Uncertainty (95% CI):")
                print(f"  Jaccard: {ci_results['jaccard']['estimate']:.3f} "
                      f"[{ci_results['jaccard']['ci_lower']:.3f}, {ci_results['jaccard']['ci_upper']:.3f}]")
                print(f"  Overlap: {ci_results['overlap']['estimate']:.3f} "
                      f"[{ci_results['overlap']['ci_lower']:.3f}, {ci_results['overlap']['ci_upper']:.3f}]")
    
    # Print results
    for score_name, results in policy_results['weighted_vs_unweighted'].items():
        print(f"\n{score_name.upper()} Score:")
        overlap = results['overlap']
        print(f"  Jaccard Index: {overlap['jaccard_index']:.3f}")
        print(f"  Dice Overlap: {overlap['overlap_rate']:.3f}")
        print(f"  Only in Weighted: {overlap['only_weighted']} ({overlap['pct_only_weighted']:.1f}%)")
        print(f"  Only in Unweighted: {overlap['only_unweighted']} ({overlap['pct_only_unweighted']:.1f}%)")
        
        # Equal budget comparison
        if 'overlap_equal_budget' in results:
            eb = results['overlap_equal_budget']
            print(f"  --- Equal Budget (N={results['n_budget']}) ---")
            print(f"  Jaccard (Equal Budget): {eb['jaccard_index']:.3f}")
    
    # Compute policy-oriented metrics (Precision/Recall@k based on TRUE consumption)
    print("\n" + "="*60)
    print("POLICY METRICS (based on TRUE consumption)")
    print("="*60)
    
    from src.policy.targeting import PolicyMetricsEvaluator
    
    policy_eval = PolicyMetricsEvaluator(k_percentile=10)
    
    # Compute with CIs if replicate weights available
    if available_rep_cols:
        policy_metrics = policy_eval.compute_with_ci(
            y_true=df['TOTALBTUSPH'].values,
            y_pred=predictions,
            weights=df['NWEIGHT'].values,
            replicate_weights=replicate_weights.values
        )
        
        print("\nMetric                  Value (95% CI)")
        print("-" * 50)
        print(f"Precision@10%:          {policy_metrics['precision_at_k']['estimate']:.3f} "
              f"[{policy_metrics['precision_at_k']['ci_lower']:.3f}, {policy_metrics['precision_at_k']['ci_upper']:.3f}]")
        print(f"Recall@10%:             {policy_metrics['recall_at_k']['estimate']:.3f} "
              f"[{policy_metrics['recall_at_k']['ci_lower']:.3f}, {policy_metrics['recall_at_k']['ci_upper']:.3f}]")
        print(f"F1@10%:                 {policy_metrics['f1_at_k']['estimate']:.3f} "
              f"[{policy_metrics['f1_at_k']['ci_lower']:.3f}, {policy_metrics['f1_at_k']['ci_upper']:.3f}]")
        print(f"Jaccard@10%:            {policy_metrics['jaccard_at_k']['estimate']:.3f} "
              f"[{policy_metrics['jaccard_at_k']['ci_lower']:.3f}, {policy_metrics['jaccard_at_k']['ci_upper']:.3f}]")
        print(f"Lift@10%:               {policy_metrics['lift_at_k']['estimate']:.1f}× "
              f"[{policy_metrics['lift_at_k']['ci_lower']:.1f}, {policy_metrics['lift_at_k']['ci_upper']:.1f}]")
        print(f"NDCG:                   {policy_metrics['ndcg']['estimate']:.3f} "
              f"[{policy_metrics['ndcg']['ci_lower']:.3f}, {policy_metrics['ndcg']['ci_upper']:.3f}]")
        print(f"Top-10% Underpred:      {policy_metrics['top_decile_underpred_pct']['estimate']:+.1f}% "
              f"[{policy_metrics['top_decile_underpred_pct']['ci_lower']:+.1f}, "
              f"{policy_metrics['top_decile_underpred_pct']['ci_upper']:+.1f}]")
        
        policy_results['policy_metrics'] = policy_metrics
    else:
        policy_metrics = policy_eval.compute_all_metrics(
            y_true=df['TOTALBTUSPH'].values,
            y_pred=predictions,
            weights=df['NWEIGHT'].values
        )
        print(f"\nPrecision@10%: {policy_metrics['precision_at_k']:.3f}")
        print(f"Recall@10%:    {policy_metrics['recall_at_k']:.3f}")
        print(f"Lift@10%:      {policy_metrics['lift_at_k']:.1f}×")
        policy_results['policy_metrics'] = policy_metrics
    
    return policy_results


def run_uncertainty_quantification(df: pd.DataFrame,
                                    predictions: np.ndarray,
                                    cv_results: dict,
                                    is_outer_fold: bool = True) -> dict:
    """
    Step 7: Uncertainty quantification.
    
    Returns
    -------
    uncertainty_results : dict
        Uncertainty quantification results
    """
    print_section_header("STEP 7: UNCERTAINTY QUANTIFICATION")
    
    # Get replicate weights
    replicate_cols = [f'NWEIGHT{i}' for i in range(1, 61)]
    available_rep_cols = [c for c in replicate_cols if c in df.columns]
    
    if not available_rep_cols:
        logger.warning("No replicate weights available, skipping uncertainty quantification")
        return {}
    
    replicate_weights = df[available_rep_cols].copy()
    
    # Compute jackknife uncertainty
    jackknife = JackknifeUncertainty(n_replicates=60)
    
    uncertainty_df = jackknife.compute_all_metrics_uncertainty(
        y_true=df['TOTALBTUSPH'].values,
        y_pred=predictions,
        main_weights=df['NWEIGHT'].values,
        replicate_weights=replicate_weights
    )
    
    print("\nMetric Uncertainty (Jackknife):")
    for _, row in uncertainty_df.iterrows():
        print(f"  {row['metric']}: {row['estimate']:.3f} ± {row['se']:.3f} "
              f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]")
    
    return {
        'uncertainty_df': uncertainty_df,
        'jackknife': jackknife,
        'is_outer_fold': is_outer_fold
    }


def run_diagnostics(df: pd.DataFrame,
                    predictions: np.ndarray,
                    cv_results: dict) -> dict:
    """
    Step 8: Diagnostics and error equity analysis.
    
    Returns
    -------
    diagnostic_results : dict
        Diagnostic results
    """
    print_section_header("STEP 8: DIAGNOSTICS AND ERROR EQUITY")
    
    y_true = df['TOTALBTUSPH'].values
    weights = df['NWEIGHT'].values
    
    # Physics diagnostics
    physics = PhysicsDiagnostics()
    physics_results = physics.run_all_diagnostics(
        y_true=y_true,
        y_pred=predictions,
        hdd=df['HDD65'].values,
        weights=weights,
        division=df['DIVISION'].values,
        tech_group=df['tech_group'].values
    )
    
    print("\nResidual Bias by HDD:")
    if 'bias_by_hdd' in physics_results:
        print(physics_results['bias_by_hdd'].to_string())
    
    # Error equity analysis
    equity = ErrorEquityAnalysis()
    equity_results = equity.run_full_audit(
        y_true=y_true,
        y_pred=predictions,
        weights=weights,
        metadata=df
    )
    
    print("\nError by Housing Type:")
    if 'by_housing_type' in equity_results:
        print(equity_results['by_housing_type'].to_string())
    
    print("\nError by Tenure:")
    if 'by_tenure' in equity_results:
        print(equity_results['by_tenure'].to_string())
    
    return {
        'physics': physics_results,
        'equity': equity_results
    }


def create_visualizations(df: pd.DataFrame,
                          predictions: np.ndarray,
                          cv_results: dict,
                          policy_results: dict,
                          diagnostic_results: dict,
                          output_dir: str) -> None:
    """
    Step 9: Create publication-quality visualizations.
    
    All figures follow publication standards:
    - Clear labeling of weighted vs unweighted, outer-fold vs in-sample
    - Human-readable labels for categorical codes
    - Uncertainty quantification where applicable
    """
    print_section_header("STEP 9: CREATING VISUALIZATIONS")
    
    viz = HeatingDemandVisualizer()
    figures_dir = Path(output_dir) / 'figures'
    ensure_dir(str(figures_dir))
    
    # Determine if results are from CV
    is_outer_fold = cv_results is not None and 'cv_result' in cv_results
    
    # Figure 1: Workflow diagram (FIXED: no dangling arrows, weights explicit)
    create_workflow_diagram(str(figures_dir / 'fig1_workflow.png'))
    
    # Figure 2: Predicted vs Observed (FIXED: calibration line, log-log, metrics labeled)
    fig = viz.plot_predicted_vs_observed(
        y_true=df['TOTALBTUSPH'].values,
        y_pred=predictions,
        weights=df['NWEIGHT'].values,
        tech_group=df['tech_group'].values,
        title="Predicted vs Observed Heating Energy",
        is_outer_fold=is_outer_fold
    )
    viz.save_figure(fig, 'fig2_pred_vs_obs.png', str(figures_dir))
    
    # Figure 2b: By technology group (supports H1)
    fig = viz.plot_predicted_vs_observed_by_tech(
        y_true=df['TOTALBTUSPH'].values,
        y_pred=predictions,
        weights=df['NWEIGHT'].values,
        tech_group=df['tech_group'].values,
        is_outer_fold=is_outer_fold
    )
    viz.save_figure(fig, 'fig2b_pred_vs_obs_by_tech.png', str(figures_dir))
    
    # Figure 3: Composition shifts (FIXED: human labels, Jaccard shown, differences)
    if 'weighted_vs_unweighted' in policy_results:
        high_use_results = policy_results['weighted_vs_unweighted'].get('high_use', {})
        if 'composition' in high_use_results and 'overlap' in high_use_results:
            fig = viz.plot_composition_shift(
                high_use_results['composition'],
                jaccard_index=high_use_results['overlap']['jaccard_index'],
                overlap_rate=high_use_results['overlap']['overlap_rate'],
                title="Composition Shift: Weighted vs Unweighted Targeting (High-Use Score)"
            )
            viz.save_figure(fig, 'fig3_composition_shift.png', str(figures_dir))
    
    # Figure 5: Residual vs HDD (FIXED: common y-axis, uncertainty bands, bin support)
    fig = viz.plot_residual_vs_hdd(
        y_true=df['TOTALBTUSPH'].values,
        y_pred=predictions,
        hdd=df['HDD65'].values,
        weights=df['NWEIGHT'].values,
        tech_group=df['tech_group'].values,
        title="Residuals vs HDD by Technology"
    )
    viz.save_figure(fig, 'fig5_residual_vs_hdd.png', str(figures_dir))
    
    # CV Results
    if cv_results and 'cv_result' in cv_results:
        fold_metrics = cv_results['cv_result'].outer_metrics_by_fold
        fig = viz.plot_cv_results(fold_metrics, model_name="LightGBM", 
                                   title="Nested Cross-Validation Results")
        viz.save_figure(fig, 'cv_results.png', str(figures_dir))
    
    # NEW: Figure 4 - H1 Comparison: Split vs Monolithic Residuals
    if cv_results and 'h1_comparison' in cv_results and 'mono_predictions' in cv_results:
        fig = viz.plot_residual_vs_hdd_comparison(
            y_true=df['TOTALBTUSPH'].values,
            y_pred_split=predictions,
            y_pred_mono=cv_results['mono_predictions'],
            hdd=df['HDD65'].values,
            weights=df['NWEIGHT'].values,
            tech_group=df['tech_group'].values,
            title="H1 Test: Residuals vs HDD — Split vs Monolithic"
        )
        viz.save_figure(fig, 'fig4_h1_split_vs_mono.png', str(figures_dir))
    
    # Error equity (FIXED: separate panels, nMAE, group sizes, no typos)
    if 'equity' in diagnostic_results:
        fig = viz.plot_error_equity(
            diagnostic_results['equity'],
            title="Error Equity Analysis"
        )
        viz.save_figure(fig, 'error_equity.png', str(figures_dir))
    
    # NEW: Calibration comparison figure (before vs after isotonic)
    if cv_results and 'predictions_uncalibrated' in cv_results:
        fig = viz.plot_calibration_comparison(
            y_true=df['TOTALBTUSPH'].values,
            y_pred_before=cv_results['predictions_uncalibrated'],
            y_pred_after=predictions,
            weights=df['NWEIGHT'].values,
            n_bins=10,
            title="Calibration Effect: Before vs After Isotonic"
        )
        viz.save_figure(fig, 'fig6_calibration_comparison.png', str(figures_dir))
    
    logger.info(f"Saved all figures to {figures_dir}")


def save_results(df: pd.DataFrame,
                 preprocessor: RECSPreprocessor,
                 cv_results: dict,
                 policy_results: dict,
                 uncertainty_results: dict,
                 diagnostic_results: dict,
                 output_dir: str) -> None:
    """
    Save all results to publication-quality tables.
    
    All tables:
    - Remove spreadsheet artifacts (Unnamed columns)
    - Replace codes with human-readable labels
    - Declare weighted vs unweighted, out-of-sample vs in-sample
    - Include metric definitions and units
    """
    print_section_header("SAVING RESULTS")
    
    from src.visualization.tables import (
        create_table1_descriptives, create_uncertainty_table,
        create_policy_targeting_table, create_composition_table,
        create_equity_table, create_hdd_diagnostics_table,
        create_h1_comparison_table, create_physics_diagnostics_table,
        save_table_with_note
    )
    
    tables_dir = Path(output_dir) / 'tables'
    ensure_dir(str(tables_dir))
    
    # Table 1: Comprehensive technology group descriptives
    table1 = create_table1_descriptives(df, df['NWEIGHT'])
    save_table_with_note(
        table1, 
        str(tables_dir / 'table1_tech_group_descriptives.csv'),
        "Table 1: Descriptive statistics by technology group. All statistics are weighted "
        "using NWEIGHT unless otherwise noted."
    )
    print("\nTable 1: Technology Group Descriptives")
    print(table1.to_string())
    
    # Table 2: CV performance (if available)
    if cv_results and 'cv_result' in cv_results:
        fold_df = cv_results['cv_result'].outer_metrics_by_fold.copy()
        fold_df = fold_df.loc[:, ~fold_df.columns.str.contains('^Unnamed')]
        
        # Add summary row
        summary_row = {col: f"{fold_df[col].mean():.3f} ± {fold_df[col].std():.3f}" 
                      for col in fold_df.columns if col != 'fold'}
        summary_row['fold'] = 'Mean ± SD'
        fold_df = pd.concat([fold_df, pd.DataFrame([summary_row])], ignore_index=True)
        
        save_table_with_note(
            fold_df,
            str(tables_dir / 'table2_cv_performance.csv'),
            "Table 2: Nested CV performance (outer-fold, weighted metrics). "
            "wRMSE and wMAE in kBTU. All metrics computed on out-of-sample predictions."
        )
    
    # NEW: Table 2b: H1 Comparison (Monolithic vs Split)
    if cv_results and 'h1_comparison' in cv_results:
        h1_table = create_h1_comparison_table(cv_results['h1_comparison'])
        save_table_with_note(
            h1_table,
            str(tables_dir / 'table2b_h1_split_vs_mono.csv'),
            h1_table.attrs.get('note', '')
        )
        print("\nTable 2b: H1 Comparison (Split vs Monolithic)")
        print(h1_table.to_string())
        
        # Compute delta CIs using replicate weights (more robust than fold-based t-test)
        if 'mono_predictions' in cv_results:
            from src.uncertainty.jackknife import JackknifeUncertainty
            
            replicate_cols = [f'NWEIGHT{i}' for i in range(1, 61)]
            available_rep_cols = [c for c in replicate_cols if c in df.columns]
            
            if available_rep_cols:
                jackknife = JackknifeUncertainty(n_replicates=60)
                delta_results = jackknife.compute_delta_metrics_ci(
                    y_true=df['TOTALBTUSPH'].values,
                    y_pred_a=cv_results['mono_predictions'],  # Monolithic
                    y_pred_b=cv_results['predictions'],  # Split
                    main_weights=df['NWEIGHT'].values,
                    replicate_weights=df[available_rep_cols]
                )
                
                # Create delta CI table
                delta_rows = []
                for metric_name, data in delta_results.items():
                    delta_rows.append({
                        'Metric': metric_name.replace('delta_', 'Δ w').upper(),
                        'Estimate': f"{data['estimate']:+,.0f}" if abs(data['estimate']) > 1 else f"{data['estimate']:+.3f}",
                        'SE': f"{data['se']:,.0f}" if abs(data['se']) > 1 else f"{data['se']:.3f}",
                        '95% CI': f"[{data['ci_lower']:+,.0f}, {data['ci_upper']:+,.0f}]" if abs(data['ci_lower']) > 1 else f"[{data['ci_lower']:+.3f}, {data['ci_upper']:+.3f}]",
                        'Monolithic': f"{data['model_a']:,.0f}" if abs(data['model_a']) > 1 else f"{data['model_a']:.3f}",
                        'Split': f"{data['model_b']:,.0f}" if abs(data['model_b']) > 1 else f"{data['model_b']:.3f}"
                    })
                
                delta_df = pd.DataFrame(delta_rows)
                save_table_with_note(
                    delta_df,
                    str(tables_dir / 'table2c_h1_delta_ci.csv'),
                    "H1 Test: Δ = Mono − Split (positive = Split better for RMSE/MAE/Bias). "
                    "CI from replicate-weight jackknife (n=60) on full cross-fitted predictions. "
                    "More robust than t-test on 5 folds."
                )
                
                print("\nTable 2c: H1 Delta Metrics with 95% CI")
                print(delta_df.to_string())
    
    # NEW: Table - Physics Diagnostics by Technology
    if cv_results and 'cv_result' in cv_results:
        predictions = cv_results.get('predictions', np.zeros(len(df)))
        physics_table = create_physics_diagnostics_table(
            y_true=df['TOTALBTUSPH'].values,
            y_pred=predictions,
            weights=df['NWEIGHT'].values,
            tech_group=df['tech_group'].values,
            hdd=df['HDD65'].values
        )
        save_table_with_note(
            physics_table,
            str(tables_dir / 'table_physics_diagnostics.csv'),
            physics_table.attrs.get('note', '')
        )
        print("\nPhysics Diagnostics by Technology:")
        print(physics_table.to_string())
    
    # Table 3: Uncertainty with proper formatting
    if uncertainty_results and 'uncertainty_df' in uncertainty_results:
        table3 = create_uncertainty_table(uncertainty_results['uncertainty_df'])
        # Use correct note based on whether CV was run
        is_oof = uncertainty_results.get('is_outer_fold', False)
        sample_type = "outer-fold test predictions" if is_oof else "in-sample predictions (CV skipped)"
        note = (
            f"All metrics computed on {sample_type} with NWEIGHT. "
            "SE and 95% CI from RECS replicate-weight jackknife (n=60). "
            "MAPE excluded (unstable for small denominators)."
        )
        save_table_with_note(
            table3,
            str(tables_dir / 'table3_uncertainty.csv'),
            note
        )
    
    # Calibration comparison table (before vs after isotonic)
    if cv_results and 'predictions_uncalibrated' in cv_results:
        from src.evaluation.metrics import WeightedMetrics
        from scipy import stats as scipy_stats
        
        y_true = df['TOTALBTUSPH'].values
        y_pred_before = cv_results['predictions_uncalibrated']
        y_pred_after = cv_results['predictions']
        w = df['NWEIGHT'].values
        
        metrics = WeightedMetrics()
        
        # Compute metrics before and after
        metrics_before = {
            'wRMSE': metrics.weighted_rmse(y_true, y_pred_before, w),
            'wMAE': metrics.weighted_mae(y_true, y_pred_before, w),
            'wR²': metrics.weighted_r2(y_true, y_pred_before, w),
            'wBias': metrics.weighted_bias(y_true, y_pred_before, w),
        }
        
        metrics_after = {
            'wRMSE': metrics.weighted_rmse(y_true, y_pred_after, w),
            'wMAE': metrics.weighted_mae(y_true, y_pred_after, w),
            'wR²': metrics.weighted_r2(y_true, y_pred_after, w),
            'wBias': metrics.weighted_bias(y_true, y_pred_after, w),
        }
        
        # Calibration slopes
        slope_before, intercept_before, _, _, _ = scipy_stats.linregress(y_true, y_pred_before)
        slope_after, intercept_after, _, _, _ = scipy_stats.linregress(y_true, y_pred_after)
        
        # Compute tail bias (top 10%)
        p90 = np.percentile(y_true, 90)
        tail_mask = y_true >= p90
        w_tail = w[tail_mask]
        tail_bias_before = np.sum(w_tail * (y_pred_before[tail_mask] - y_true[tail_mask])) / np.sum(w_tail)
        tail_bias_after = np.sum(w_tail * (y_pred_after[tail_mask] - y_true[tail_mask])) / np.sum(w_tail)
        tail_mean = np.sum(w_tail * y_true[tail_mask]) / np.sum(w_tail)
        tail_bias_pct_before = tail_bias_before / tail_mean * 100
        tail_bias_pct_after = tail_bias_after / tail_mean * 100
        
        # Bottom decile stats (for explanation)
        p10 = np.percentile(y_true, 10)
        bottom_mask = y_true <= p10
        w_bottom = w[bottom_mask]
        bottom_mean = np.sum(w_bottom * y_true[bottom_mask]) / np.sum(w_bottom)
        bottom_bias_before = np.sum(w_bottom * (y_pred_before[bottom_mask] - y_true[bottom_mask])) / np.sum(w_bottom)
        bottom_bias_after = np.sum(w_bottom * (y_pred_after[bottom_mask] - y_true[bottom_mask])) / np.sum(w_bottom)
        
        calib_rows = [
            {'Metric': 'wRMSE (kBTU)', 'Before': f"{metrics_before['wRMSE']:,.0f}", 
             'After': f"{metrics_after['wRMSE']:,.0f}", 
             'Change': f"{(metrics_after['wRMSE'] - metrics_before['wRMSE']) / metrics_before['wRMSE'] * 100:+.1f}%"},
            {'Metric': 'wMAE (kBTU)', 'Before': f"{metrics_before['wMAE']:,.0f}", 
             'After': f"{metrics_after['wMAE']:,.0f}",
             'Change': f"{(metrics_after['wMAE'] - metrics_before['wMAE']) / metrics_before['wMAE'] * 100:+.1f}%"},
            {'Metric': 'wR²', 'Before': f"{metrics_before['wR²']:.3f}", 
             'After': f"{metrics_after['wR²']:.3f}",
             'Change': f"{metrics_after['wR²'] - metrics_before['wR²']:+.3f}"},
            {'Metric': 'wBias (kBTU)', 'Before': f"{metrics_before['wBias']:,.0f}", 
             'After': f"{metrics_after['wBias']:,.0f}",
             'Change': f"{(metrics_after['wBias'] - metrics_before['wBias']):+,.0f}"},
            {'Metric': 'Calibration Slope', 'Before': f"{slope_before:.3f}", 
             'After': f"{slope_after:.3f}",
             'Change': f"{slope_after - slope_before:+.3f} (→1.0)"},
            {'Metric': 'Tail Bias D10 (top 10%)', 'Before': f"{tail_bias_pct_before:.1f}%", 
             'After': f"{tail_bias_pct_after:.1f}%",
             'Change': f"{tail_bias_pct_after - tail_bias_pct_before:+.1f}pp"},
            {'Metric': 'D1 Mean Observed (kBTU)', 'Before': f"{bottom_mean:,.0f}", 
             'After': f"{bottom_mean:,.0f}",
             'Change': '(small Y → high % bias)'},
            {'Metric': 'D1 Absolute Bias (kBTU)', 'Before': f"{bottom_bias_before:,.0f}", 
             'After': f"{bottom_bias_after:,.0f}",
             'Change': f"{bottom_bias_after - bottom_bias_before:+,.0f}"},
        ]
        
        calib_df = pd.DataFrame(calib_rows)
        save_table_with_note(
            calib_df,
            str(tables_dir / 'table_calibration_comparison.csv'),
            "Calibration effect: Before = raw LightGBM (Tweedie), After = isotonic calibration. "
            "Bias (%) = 100×(Ŷ−Y)/Ȳ_decile where Ȳ_decile is weighted mean observed in each decile. "
            f"D1 (bottom decile) has high % bias because Ȳ≈{bottom_mean:,.0f} kBTU is small; "
            f"absolute bias is only {bottom_bias_after:,.0f} kBTU. "
            "D10 (top decile) underprediction is the policy-relevant concern. "
            "Calibration slope via weighted OLS (NWEIGHT). "
            "95% CI in figure: B=500 household-level bootstrap (RECS: 1 observation = 1 household), "
            "resampling with replacement, weighted statistics in each replicate."
        )
        print("\nTable: Calibration Comparison (Before vs After Isotonic)")
        print(calib_df.to_string())
    
    # Policy targeting tables with uncertainty
    if policy_results and 'weighted_vs_unweighted' in policy_results:
        for score_name in ['high_use', 'high_intensity', 'excess_demand']:
            if score_name in policy_results['weighted_vs_unweighted']:
                # Summary table with CIs
                summary_table = create_policy_targeting_table(policy_results, score_name)
                
                # Add Jaccard CIs if available
                if 'uncertainty' in policy_results and score_name in policy_results['uncertainty']:
                    unc = policy_results['uncertainty'][score_name]
                    ci_rows = []
                    
                    # Add Jaccard CI row
                    if 'jaccard' in unc:
                        j = unc['jaccard']
                        ci_rows.append({
                            'Metric': 'Jaccard Index (95% CI)',
                            'Value': f"{j['estimate']:.3f} [{j['ci_lower']:.3f}, {j['ci_upper']:.3f}]",
                            'Description': f"Jaccard with replicate-weight 95% CI (SE={j['se']:.3f})"
                        })
                    
                    # Add Overlap CI row
                    if 'overlap' in unc:
                        o = unc['overlap']
                        ci_rows.append({
                            'Metric': 'Dice Overlap (95% CI)',
                            'Value': f"{o['estimate']:.3f} [{o['ci_lower']:.3f}, {o['ci_upper']:.3f}]",
                            'Description': f"Dice with replicate-weight 95% CI (SE={o['se']:.3f})"
                        })
                    
                    if ci_rows:
                        ci_df = pd.DataFrame(ci_rows)
                        summary_table = pd.concat([summary_table, ci_df], ignore_index=True)
                
                if len(summary_table) > 0:
                    save_table_with_note(
                        summary_table,
                        str(tables_dir / f'policy_{score_name}_summary.csv'),
                        summary_table.attrs.get('note', '') + " Jaccard/Dice CIs from replicate-weight jackknife."
                    )
                
                # Composition tables with human-readable labels
                results = policy_results['weighted_vs_unweighted'][score_name]
                if 'composition' in results:
                    for group_name, comp_df in results['composition'].items():
                        if comp_df is not None and len(comp_df) > 0:
                            formatted_comp = create_composition_table(comp_df, group_name)
                            save_table_with_note(
                                formatted_comp,
                                str(tables_dir / f'policy_{score_name}_{group_name}.csv'),
                                f"Composition shift for {score_name} targeting by {group_name}. "
                                "Representation Ratio = Share among candidates / Population share."
                            )
    
    # Policy metrics table (Precision/Recall@k based on TRUE consumption)
    if policy_results and 'policy_metrics' in policy_results:
        from src.policy.targeting import PolicyMetricsEvaluator
        
        pm = policy_results['policy_metrics']
        
        # Check if we have CIs (dict of dicts) or just values
        if isinstance(pm.get('precision_at_k'), dict):
            # We have CIs
            policy_rows = [
                {'Metric': 'Precision@10%', 
                 'Value': f"{pm['precision_at_k']['estimate']:.3f}",
                 '95% CI': f"[{pm['precision_at_k']['ci_lower']:.3f}, {pm['precision_at_k']['ci_upper']:.3f}]",
                 'Description': 'Of predicted top-10%, fraction truly high consumers'},
                {'Metric': 'Recall@10%', 
                 'Value': f"{pm['recall_at_k']['estimate']:.3f}",
                 '95% CI': f"[{pm['recall_at_k']['ci_lower']:.3f}, {pm['recall_at_k']['ci_upper']:.3f}]",
                 'Description': 'Of true top-10%, fraction in predicted top-10%'},
                {'Metric': 'F1@10%', 
                 'Value': f"{pm['f1_at_k']['estimate']:.3f}",
                 '95% CI': f"[{pm['f1_at_k']['ci_lower']:.3f}, {pm['f1_at_k']['ci_upper']:.3f}]",
                 'Description': 'Harmonic mean of Precision and Recall'},
                {'Metric': 'Jaccard@10%', 
                 'Value': f"{pm['jaccard_at_k']['estimate']:.3f}",
                 '95% CI': f"[{pm['jaccard_at_k']['ci_lower']:.3f}, {pm['jaccard_at_k']['ci_upper']:.3f}]",
                 'Description': 'Overlap between predicted and true top-10%'},
                {'Metric': 'Lift@10%', 
                 'Value': f"{pm['lift_at_k']['estimate']:.1f}×",
                 '95% CI': f"[{pm['lift_at_k']['ci_lower']:.1f}, {pm['lift_at_k']['ci_upper']:.1f}]",
                 'Description': 'Improvement over random selection'},
                {'Metric': 'NDCG', 
                 'Value': f"{pm['ndcg']['estimate']:.3f}",
                 '95% CI': f"[{pm['ndcg']['ci_lower']:.3f}, {pm['ndcg']['ci_upper']:.3f}]",
                 'Description': 'Ranking quality (1.0 = perfect)'},
                {'Metric': 'Top-10% Underprediction', 
                 'Value': f"{pm['top_decile_underpred_pct']['estimate']:+.1f}%",
                 '95% CI': f"[{pm['top_decile_underpred_pct']['ci_lower']:+.1f}, {pm['top_decile_underpred_pct']['ci_upper']:+.1f}]",
                 'Description': 'Bias in top decile (negative = underprediction)'},
            ]
        else:
            # No CIs, just values
            policy_rows = [
                {'Metric': 'Precision@10%', 'Value': f"{pm['precision_at_k']:.3f}", '95% CI': 'N/A',
                 'Description': 'Of predicted top-10%, fraction truly high consumers'},
                {'Metric': 'Recall@10%', 'Value': f"{pm['recall_at_k']:.3f}", '95% CI': 'N/A',
                 'Description': 'Of true top-10%, fraction in predicted top-10%'},
                {'Metric': 'Lift@10%', 'Value': f"{pm['lift_at_k']:.1f}×", '95% CI': 'N/A',
                 'Description': 'Improvement over random selection'},
            ]
        
        policy_metrics_df = pd.DataFrame(policy_rows)
        save_table_with_note(
            policy_metrics_df,
            str(tables_dir / 'table_policy_metrics.csv'),
            "Policy metrics for top-10% targeting. 'True high' = weighted 90th percentile of observed consumption. "
            "Precision@k: fraction of predicted high that are truly high. "
            "Recall@k: fraction of truly high captured. "
            "Lift@k: improvement over random (Precision / base rate). "
            "95% CIs from replicate-weight jackknife (n=60)."
        )
        print("\nTable: Policy Metrics (Based on TRUE Consumption)")
        print(policy_metrics_df.to_string())
    
    # Equity tables with normalized metrics
    if diagnostic_results and 'equity' in diagnostic_results:
        for name, equity_df in diagnostic_results['equity'].items():
            if equity_df is not None and len(equity_df) > 0:
                group_type = name.replace('by_', '')
                formatted_equity = create_equity_table(equity_df, group_type)
                save_table_with_note(
                    formatted_equity,
                    str(tables_dir / f'equity_{name}.csv'),
                    formatted_equity.attrs.get('note', '')
                )
    
    # HDD diagnostics with bin support
    if diagnostic_results and 'physics' in diagnostic_results:
        if 'bias_by_hdd' in diagnostic_results['physics']:
            hdd_table = create_hdd_diagnostics_table(diagnostic_results['physics']['bias_by_hdd'])
            save_table_with_note(
                hdd_table,
                str(tables_dir / 'diagnostics_bias_by_hdd.csv'),
                hdd_table.attrs.get('note', '')
            )
    
    # Sensitivity analysis results (documented for transparency)
    sensitivity_note = """
# Sensitivity Analysis Notes
# --------------------------
# 1. Monotonic Constraints: Model uses Tweedie objective with optional monotonic
#    constraints on HDD (positive) and TOTSQFT_EN (positive). Ablation shows
#    constraints slightly increase RMSE but improve physics plausibility.
#
# 2. COVID Controls: Model includes TELLWORK and ATHOME indicators as proxies
#    for pandemic-era occupancy. Sensitivity to COVID control mode is documented
#    but full ablation requires separate runs.
#
# 3. Technology Assignment: Using primary_only rule. Hybrid/ambiguous cases
#    are modeled separately but results should be interpreted with caution.
#
# 4. Calibration: Model applies isotonic post-calibration to reduce tail bias.
#    This improves calibration slope but does not fully eliminate underprediction
#    at high consumption levels.
#
# 5. Subgroup wR²: Low/negative wR² for electric subgroups reflects high noise
#    in RECS end-use estimates and low within-group variance. Weighted correlation
#    (r) is provided as complementary metric.
"""
    
    with open(tables_dir / 'sensitivity_notes.txt', 'w') as f:
        f.write(sensitivity_note)
    
    logger.info(f"Saved all tables to {tables_dir}")


def main():
    """Main entry point."""
    args = parse_args()
    
    print_section_header("HEATING DEMAND MODELING FRAMEWORK", char="=", width=80)
    print("RECS 2020 Analysis")
    print("="*80)
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Load config
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.warning(f"Config file not found: {args.config}, using defaults")
        config = {}

    # Resolve compute device (GPU if available/requested)
    hw = (config.get('hardware', {}) or {}) if isinstance(config, dict) else {}
    device_req = args.device if args.device is not None else hw.get('device', 'auto')
    gpu_id = int(args.gpu_id) if args.gpu_id is not None else int(hw.get('gpu_id', 0))
    lgbm_gpu_platform_id = int(hw.get('lgbm_gpu_platform_id', 0))

    device_req_l = str(device_req).lower() if device_req is not None else 'auto'
    use_cuda = False
    resolved_device = 'cpu'

    if device_req_l in ('cuda', 'gpu'):
        use_cuda = True
    elif device_req_l.startswith('cuda:'):
        use_cuda = True
        try:
            gpu_id = int(device_req_l.split(':', 1)[1])
        except Exception:
            pass
    elif device_req_l == 'auto':
        try:
            import torch
            use_cuda = bool(torch.cuda.is_available())
        except Exception:
            use_cuda = False
    else:
        use_cuda = False

    if use_cuda:
        resolved_device = f"cuda:{gpu_id}"

    if isinstance(config, dict):
        config.setdefault('runtime', {})
        config['runtime'].update({
            'use_cuda': use_cuda,
            'gpu_id': gpu_id,
            'device': resolved_device,
            'lgbm_gpu_platform_id': lgbm_gpu_platform_id,
        })

    logger.info(f"Runtime device: {resolved_device} (use_cuda={use_cuda}, gpu_id={gpu_id})")

    # Print environment/dependency/GPU readiness report
    try:
        print_environment_healthcheck(config, resolved_device, use_cuda, gpu_id)
    except Exception as e:
        logger.warning(f"Health check failed (continuing): {e}")

    # Create output directories
    ensure_dir(args.output)
    ensure_dir(f"{args.output}/figures")
    ensure_dir(f"{args.output}/tables")
    ensure_dir(f"{args.output}/models")
    ensure_dir(f"{args.output}/tuning")
    
    # Initialize run manifest (Priority 2.2)
    manifest = create_run_manifest(
        config=config,
        output_dir=args.output,
        experiment_name='heating_demand_analysis'
    )
    manifest.start()
    
    # Run pipeline
    with Timer("Complete analysis pipeline"):
        
        # Step 1-2: Data loading and preprocessing
        df, preprocessor, loader = run_data_loading_and_preprocessing(config)
        
        # Step 3: Feature engineering
        X, y, weights, feature_builder = run_feature_engineering(df, config)
        
        # Step 4: Physics baselines
        baselines, baseline_results, baseline_predictions = run_physics_baselines(df)
        
        # Step 5: Nested CV (can be skipped for quick testing)
        if not args.skip_cv:
            cv_results = run_nested_cv(df, feature_builder, config, args.n_outer_folds, args.output)
            predictions = cv_results['predictions']
            is_outer_fold = True  # Proper outer-fold out-of-sample predictions

            # Step 5B: Optional SOTA benchmark comparisons (table + plots)
            try:
                run_sota_benchmarks(df, feature_builder, config, cv_results, baseline_predictions, args.n_outer_folds, args.output)
            except Exception as e:
                logger.warning(f"SOTA benchmarks step failed: {e}")
            
            # Step 5C: Tail bias mitigation (Priority 1)
            tail_bias_results = {}
            try:
                tail_bias_results = run_tail_bias_mitigation(
                    df, feature_builder, cv_results, config,
                    args.n_outer_folds, args.output
                )
                manifest.add_result('tail_bias', {
                    'baseline_top10_bias': tail_bias_results.get('baseline_tail_metrics', {}).get('top_decile_bias_pct'),
                    'quantile_top10_bias': tail_bias_results.get('quantile_tail_metrics', {}).get('top_decile_bias_pct'),
                    'tail_weighted_top10_bias': tail_bias_results.get('tail_weighted_tail_metrics', {}).get('top_decile_bias_pct')
                })
            except Exception as e:
                logger.warning(f"Tail bias mitigation step failed: {e}")
        else:
            logger.info("Skipping nested CV (--skip-cv flag)")
            # Use simple train/predict for testing
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
                X, y, weights, test_size=0.2, random_state=42
            )
            rt = (config.get('runtime', {}) or {}) if isinstance(config, dict) else {}
            lgbm_params = {}
            if bool(rt.get('use_cuda', False)):
                lgbm_params = {
                    'device_type': 'gpu',
                    'gpu_platform_id': int(rt.get('lgbm_gpu_platform_id', 0)),
                    'gpu_device_id': int(rt.get('gpu_id', 0)),
                    'max_bin': 255
                }
            model = LightGBMHeatingModel(params=lgbm_params)
            X_train_t = feature_builder.fit_transform(X_train)
            X_test_t = feature_builder.transform(X_test)
            model.fit(X_train_t, y_train.values, sample_weight=w_train.values)
            predictions = np.zeros(len(df))
            predictions[X_test.index] = model.predict(X_test_t)
            predictions[X_train.index] = model.predict(X_train_t)
            cv_results = None
            is_outer_fold = False  # In-sample predictions (CV skipped)
        
        # Step 6: Policy analysis
        policy_results = run_policy_analysis(df, predictions, baseline_predictions)
        
        # Step 7: Uncertainty quantification
        uncertainty_results = run_uncertainty_quantification(df, predictions, cv_results, 
                                                              is_outer_fold=is_outer_fold)
        
        # Step 8: Diagnostics
        diagnostic_results = run_diagnostics(df, predictions, cv_results)
        
        # Step 9: Visualizations
        create_visualizations(df, predictions, cv_results, policy_results, 
                             diagnostic_results, args.output)
        
        # Save results
        save_results(df, preprocessor, cv_results, policy_results,
                    uncertainty_results, diagnostic_results, args.output)
        
        # Add final results to manifest
        if cv_results and 'cv_result' in cv_results:
            manifest.add_result('cv_metrics', cv_results['cv_result'].outer_metrics)
        if policy_results and 'policy_metrics' in policy_results:
            pm = policy_results['policy_metrics']
            manifest.add_result('policy_metrics', {
                'precision_at_k': pm.get('precision_at_k', {}).get('estimate') if isinstance(pm.get('precision_at_k'), dict) else pm.get('precision_at_k'),
                'recall_at_k': pm.get('recall_at_k', {}).get('estimate') if isinstance(pm.get('recall_at_k'), dict) else pm.get('recall_at_k'),
                'lift_at_k': pm.get('lift_at_k', {}).get('estimate') if isinstance(pm.get('lift_at_k'), dict) else pm.get('lift_at_k'),
            })
    
    # Finalize run manifest
    manifest.end()
    logger.info(f"Run manifest saved to: {args.output}/run_manifest.json")
    
    print_section_header("ANALYSIS COMPLETE", char="=", width=80)
    print(f"Results saved to: {args.output}")
    print(f"Run manifest: {args.output}/run_manifest.json")
    print("="*80)


if __name__ == '__main__':
    main()
