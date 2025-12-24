"""
Policy Targeting Analysis

Implements Section 8 Policy Metrics and H2 testing:
- Define candidates using weighted 90th percentile of policy score
- Compare weighted vs unweighted candidate lists
- Jaccard index, overlap rate, composition shifts

POLICY-ORIENTED METRICS (D):
- Precision@k: Of top-k by prediction, fraction truly high consumers
- Recall@k: Of true high consumers, fraction captured in top-k predictions
- Overlap@k: Overlap between predicted top-k and true top-k
- Lift@k: Precision@k / base rate (how much better than random)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

from src.utils.helpers import logger, compute_weighted_quantile, Timer


class PolicyTargeting:
    """
    Policy targeting analysis for retrofit prioritization.
    
    Implements H2 test: Using survey weights changes who gets flagged
    for retrofit targeting, not only average metrics.
    
    Policy scores:
    - High-use score: predicted E_heat
    - High-intensity score: predicted E_heat / Area
    - Excess-demand (inefficiency proxy): residual-like score
    """
    
    def __init__(self, target_percentile: float = 90):
        """
        Initialize policy targeting analysis.
        
        Parameters
        ----------
        target_percentile : float
            Percentile threshold for targeting (default 90 = top 10%)
        """
        self.target_percentile = target_percentile
        self.quantile = target_percentile / 100
        
    def compute_policy_scores(self,
                              y_pred: np.ndarray,
                              y_baseline: np.ndarray,
                              area: np.ndarray) -> pd.DataFrame:
        """
        Compute policy scores for targeting.
        
        Parameters
        ----------
        y_pred : array
            Predicted energy values
        y_baseline : array
            Baseline energy predictions (for excess demand)
        area : array
            Floor area (square feet)
            
        Returns
        -------
        DataFrame
            Policy scores
        """
        scores = pd.DataFrame({
            'high_use': y_pred,
            'high_intensity': y_pred / np.maximum(area, 1),
            'excess_demand': y_pred - y_baseline
        })
        
        return scores
    
    def identify_candidates(self,
                            scores: np.ndarray,
                            weights: np.ndarray,
                            use_weights: bool = True) -> np.ndarray:
        """
        Identify candidates above threshold.
        
        Parameters
        ----------
        scores : array
            Policy scores
        weights : array
            Sample weights
        use_weights : bool
            Whether to use weights for threshold calculation
            
        Returns
        -------
        array
            Boolean mask of candidates
        """
        if use_weights:
            threshold = compute_weighted_quantile(scores, weights, self.quantile)
        else:
            threshold = np.quantile(scores[~np.isnan(scores)], self.quantile)
        
        return scores >= threshold
    
    def compare_weighted_vs_unweighted(self,
                                        scores: pd.DataFrame,
                                        weights: np.ndarray,
                                        metadata: pd.DataFrame) -> Dict[str, Any]:
        """
        Compare weighted vs unweighted targeting.
        
        Parameters
        ----------
        scores : DataFrame
            Policy scores (high_use, high_intensity, excess_demand)
        weights : array
            Sample weights
        metadata : DataFrame
            Metadata for composition analysis (income, housing type, etc.)
            
        Returns
        -------
        dict
            Comparison results
        """
        results = {}
        
        for score_name in scores.columns:
            score_values = scores[score_name].values
            
            # Identify candidates with and without weights
            candidates_weighted = self.identify_candidates(score_values, weights, use_weights=True)
            candidates_unweighted = self.identify_candidates(score_values, weights, use_weights=False)
            
            # Compute overlap metrics
            overlap_metrics = self._compute_overlap_metrics(
                candidates_weighted, candidates_unweighted
            )
            
            # Composition analysis
            composition = self._analyze_composition_shift(
                candidates_weighted, candidates_unweighted,
                weights, metadata
            )
            
            # Equal-budget comparison (fixed number of candidates)
            # This is the "fair" policy comparison
            n_budget = candidates_weighted.sum()  # Use weighted count as budget
            
            # Select top-N by weighted ranking
            weighted_ranks = np.argsort(-score_values * weights / weights.sum())
            candidates_weighted_topN = np.zeros(len(score_values), dtype=bool)
            candidates_weighted_topN[weighted_ranks[:n_budget]] = True
            
            # Select top-N by unweighted ranking  
            unweighted_ranks = np.argsort(-score_values)
            candidates_unweighted_topN = np.zeros(len(score_values), dtype=bool)
            candidates_unweighted_topN[unweighted_ranks[:n_budget]] = True
            
            # Overlap for equal-budget comparison
            overlap_equal_budget = self._compute_overlap_metrics(
                candidates_weighted_topN, candidates_unweighted_topN
            )

            # Fairness / disparate-impact metrics (demographic parity style)
            fairness_audit = self._compute_fairness_audit(
                candidates_weighted=candidates_weighted,
                candidates_unweighted=candidates_unweighted,
                weights=weights,
                metadata=metadata
            )
            
            results[score_name] = {
                'overlap': overlap_metrics,
                'overlap_equal_budget': overlap_equal_budget,  # NEW: equal-budget comparison
                'composition': composition,
                'fairness_audit': fairness_audit,
                'n_weighted_candidates': candidates_weighted.sum(),
                'n_unweighted_candidates': candidates_unweighted.sum(),
                'n_budget': n_budget,
                'weighted_threshold': compute_weighted_quantile(score_values, weights, self.quantile),
                'unweighted_threshold': np.quantile(score_values[~np.isnan(score_values)], self.quantile),
                'target_pct': 100 - self.target_percentile  # 10% for top 10%
            }
        
        return results
    
    def _compute_overlap_metrics(self,
                                  candidates_a: np.ndarray,
                                  candidates_b: np.ndarray) -> Dict[str, float]:
        """
        Compute overlap metrics between two candidate sets.
        
        Parameters
        ----------
        candidates_a : array
            First candidate mask (e.g., weighted)
        candidates_b : array
            Second candidate mask (e.g., unweighted)
            
        Returns
        -------
        dict
            Overlap metrics
        """
        # Counts
        n_a = candidates_a.sum()
        n_b = candidates_b.sum()
        intersection = np.sum(candidates_a & candidates_b)
        union = np.sum(candidates_a | candidates_b)
        
        # Jaccard index: |A ∩ B| / |A ∪ B|
        jaccard = intersection / union if union > 0 else 0
        
        # Dice coefficient (Sørensen–Dice): 2|A ∩ B| / (|A| + |B|)
        # Measures overlap as harmonic mean of precision and recall
        dice = 2 * intersection / (n_a + n_b) if (n_a + n_b) > 0 else 0
        
        # Recall (from perspective of weighted candidates)
        # What fraction of weighted candidates are also in unweighted?
        recall_weighted = intersection / n_a if n_a > 0 else 0
        
        # Precision (from perspective of unweighted candidates)
        # What fraction of unweighted candidates are also in weighted?
        precision_unweighted = intersection / n_b if n_b > 0 else 0
        
        # Containment coefficient: |A ∩ B| / min(|A|, |B|)
        # Equals 1.0 if smaller set is fully contained in larger
        min_size = min(n_a, n_b)
        containment = intersection / min_size if min_size > 0 else 0
        
        # Proportion only in weighted
        only_weighted = np.sum(candidates_a & ~candidates_b)
        # Proportion only in unweighted
        only_unweighted = np.sum(~candidates_a & candidates_b)
        
        return {
            'jaccard_index': jaccard,
            'dice_coefficient': dice,  # F1-like: harmonic mean of precision/recall
            'recall_weighted': recall_weighted,  # Fraction of weighted also in unweighted
            'precision_unweighted': precision_unweighted,  # Fraction of unweighted also in weighted
            'containment': containment,  # Containment coefficient
            'overlap_rate': dice,  # Alias for backward compatibility (now Dice)
            'intersection': intersection,
            'union': union,
            'n_weighted': n_a,
            'n_unweighted': n_b,
            'only_weighted': only_weighted,
            'only_unweighted': only_unweighted,
            'pct_only_weighted': only_weighted / n_a * 100 if n_a > 0 else 0,
            'pct_only_unweighted': only_unweighted / n_b * 100 if n_b > 0 else 0
        }
    
    def _analyze_composition_shift(self,
                                    candidates_weighted: np.ndarray,
                                    candidates_unweighted: np.ndarray,
                                    weights: np.ndarray,
                                    metadata: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Analyze composition shifts by demographic/housing groups.
        
        Parameters
        ----------
        candidates_weighted : array
            Weighted candidate mask
        candidates_unweighted : array
            Unweighted candidate mask
        weights : array
            Sample weights
        metadata : DataFrame
            Metadata columns for grouping
            
        Returns
        -------
        dict
            Composition analysis by group type
        """
        results = {}
        
        # Analyze by available grouping variables
        grouping_vars = {
            'income': 'MONEYPY',
            'housing_type': 'TYPEHUQ',
            'tenure': 'KOWNRENT',
            'division': 'DIVISION',
            'climate': 'HDD_bin'
        }
        
        for group_name, col_name in grouping_vars.items():
            if col_name not in metadata.columns:
                continue
            
            group_col = metadata[col_name].values
            
            composition_df = self._compute_group_composition(
                candidates_weighted, candidates_unweighted,
                weights, group_col, group_name
            )
            
            results[group_name] = composition_df
        
        return results
    
    def _compute_group_composition(self,
                                    candidates_weighted: np.ndarray,
                                    candidates_unweighted: np.ndarray,
                                    weights: np.ndarray,
                                    groups: np.ndarray,
                                    group_name: str) -> pd.DataFrame:
        """
        Compute composition by group with selection rates and representation ratios.
        
        Includes:
        - Candidate share among all candidates
        - Within-group selection rate (% of group selected)
        - Representation ratio (candidate share / population share)
        """
        results = []
        total_weight = weights.sum()
        total_weighted_candidates = weights[candidates_weighted].sum()
        total_unweighted_candidates = candidates_unweighted.sum()
        
        for group_val in np.unique(groups):
            if pd.isna(group_val):
                continue
            
            group_mask = groups == group_val
            group_weight = weights[group_mask].sum()
            n_in_group = group_mask.sum()
            
            # Population share (weighted)
            pop_share = group_weight / total_weight * 100
            
            # Weighted share in weighted candidates
            w_in_weighted_cands = weights[candidates_weighted & group_mask].sum()
            share_weighted = w_in_weighted_cands / total_weighted_candidates * 100 if total_weighted_candidates > 0 else 0
            
            # Share in unweighted candidates
            n_in_unweighted_cands = (candidates_unweighted & group_mask).sum()
            share_unweighted = n_in_unweighted_cands / total_unweighted_candidates * 100 if total_unweighted_candidates > 0 else 0
            
            # Within-group selection rate (weighted)
            # What % of this group gets selected as candidates?
            within_group_selection_rate = w_in_weighted_cands / group_weight * 100 if group_weight > 0 else 0
            
            # Representation ratio (candidate share / population share)
            # >1 = overrepresented, <1 = underrepresented
            representation_ratio = share_weighted / pop_share if pop_share > 0 else 0
            
            results.append({
                f'{group_name}': group_val,
                'n_in_group': n_in_group,
                'population_share': pop_share,
                'share_weighted_candidates': share_weighted,
                'share_unweighted_candidates': share_unweighted,
                'share_difference': share_weighted - share_unweighted,
                'within_group_selection_rate': within_group_selection_rate,
                'representation_ratio': representation_ratio
            })
        
        return pd.DataFrame(results)


    # ---------------------------------------------------------------------
    # Fairness / disparate-impact utilities (requested for equity analysis)
    # ---------------------------------------------------------------------
    def _fairness_metrics_for_groups(self,
                                     candidates: np.ndarray,
                                     weights: np.ndarray,
                                     groups: np.ndarray,
                                     min_group_weight: float = 0.0) -> Dict[str, Any]:
        """Compute simple demographic-parity style metrics for a binary selection mask.

        Uses survey weights for both denominators and numerators so metrics reflect population impact.

        Returns a dict with:
        - group_rates: DataFrame of per-group selection rates
        - demographic_parity_diff_pp: max-min selection rate (percentage points)
        - disparate_impact_ratio: min/max selection rate (unitless)
        """
        rows = []
        groups_series = pd.Series(groups)
        unique_vals = [v for v in groups_series.dropna().unique()]

        for v in unique_vals:
            mask = (groups_series.values == v)
            group_w = float(np.sum(weights[mask]))
            if group_w <= min_group_weight:
                continue
            selected_w = float(np.sum(weights[mask & candidates]))
            sel_rate = 100.0 * selected_w / group_w if group_w > 0 else 0.0
            rows.append({
                "group": v,
                "group_weight": group_w,
                "selected_weight": selected_w,
                "selection_rate_pct": sel_rate
            })

        df_rates = pd.DataFrame(rows).sort_values("selection_rate_pct", ascending=False)

        if len(df_rates) == 0:
            return {
                "group_rates": df_rates,
                "demographic_parity_diff_pp": float("nan"),
                "disparate_impact_ratio": float("nan"),
                "max_group": None,
                "min_group": None
            }

        max_rate = float(df_rates["selection_rate_pct"].max())
        min_rate = float(df_rates["selection_rate_pct"].min())

        max_group = df_rates.loc[df_rates["selection_rate_pct"].idxmax(), "group"]
        min_group = df_rates.loc[df_rates["selection_rate_pct"].idxmin(), "group"]

        di_ratio = (min_rate / max_rate) if max_rate > 0 else float("nan")

        return {
            "group_rates": df_rates,
            "demographic_parity_diff_pp": max_rate - min_rate,
            "disparate_impact_ratio": di_ratio,
            "max_group": max_group,
            "min_group": min_group
        }

    def _compute_fairness_audit(self,
                               candidates_weighted: np.ndarray,
                               candidates_unweighted: np.ndarray,
                               weights: np.ndarray,
                               metadata: pd.DataFrame) -> Dict[str, Any]:
        """Compute fairness metrics (demographic parity / disparate impact) by subgroup."""
        fairness = {}

        grouping_vars = {
            'income': 'MONEYPY',
            'housing_type': 'TYPEHUQ',
            'tenure': 'KOWNRENT',
            'division': 'DIVISION',
            'climate': 'HDD_bin'
        }

        for group_name, col_name in grouping_vars.items():
            if col_name not in metadata.columns:
                continue
            g = metadata[col_name].values

            fairness[group_name] = {
                "weighted_targeting": self._fairness_metrics_for_groups(
                    candidates=candidates_weighted,
                    weights=weights,
                    groups=g
                ),
                "unweighted_targeting": self._fairness_metrics_for_groups(
                    candidates=candidates_unweighted,
                    weights=weights,
                    groups=g
                )
            }

        return fairness
    
    def run_full_analysis(self,
                          y_pred: np.ndarray,
                          y_baseline: np.ndarray,
                          y_true: np.ndarray,
                          weights: np.ndarray,
                          metadata: pd.DataFrame) -> Dict[str, Any]:
        """
        Run full policy targeting analysis.
        
        Parameters
        ----------
        y_pred : array
            Model predictions
        y_baseline : array
            Baseline predictions
        y_true : array
            True values
        weights : array
            Sample weights
        metadata : DataFrame
            Metadata (must include TOTSQFT_EN)
            
        Returns
        -------
        dict
            Complete analysis results
        """
        with Timer("Running policy targeting analysis"):
            
            # Compute policy scores
            area = metadata['TOTSQFT_EN'].values if 'TOTSQFT_EN' in metadata else np.ones(len(y_pred))
            scores = self.compute_policy_scores(y_pred, y_baseline, area)
            
            # Compare weighted vs unweighted
            comparison = self.compare_weighted_vs_unweighted(scores, weights, metadata)
            
            # Add actual energy analysis (who we're actually missing)
            actual_high_use = y_true >= compute_weighted_quantile(y_true, weights, self.quantile)
            
            # Coverage analysis - how well do predicted candidates cover actual high users?
            coverage = {}
            for score_name in scores.columns:
                pred_candidates = self.identify_candidates(scores[score_name].values, weights, True)
                
                # True positive rate (sensitivity)
                tp = np.sum(pred_candidates & actual_high_use)
                coverage[score_name] = {
                    'true_positive_rate': tp / actual_high_use.sum() * 100 if actual_high_use.sum() > 0 else 0,
                    'false_positive_rate': (pred_candidates.sum() - tp) / pred_candidates.sum() * 100 if pred_candidates.sum() > 0 else 0
                }
            
            return {
                'scores': scores,
                'weighted_vs_unweighted': comparison,
                'coverage_analysis': coverage,
                'target_percentile': self.target_percentile
            }


class TargetingUncertainty:
    """
    Uncertainty quantification for policy targeting using replicate weights.
    
    Computes 95% CIs for:
    - Jaccard index
    - Overlap rate
    - Subgroup shares
    - Threshold values
    """
    
    def __init__(self, n_replicates: int = 60, confidence_level: float = 0.95):
        self.n_replicates = n_replicates
        self.confidence_level = confidence_level
        self.z_score = 1.96 if confidence_level == 0.95 else 1.645
        
    def compute_targeting_uncertainty(self,
                                       scores: np.ndarray,
                                       weights: np.ndarray,
                                       replicate_weights: pd.DataFrame,
                                       target_percentile: float = 90) -> Dict[str, Any]:
        """
        Compute uncertainty in targeting metrics using jackknife.
        
        Parameters
        ----------
        scores : array
            Policy scores
        weights : array
            Main weights
        replicate_weights : DataFrame
            Replicate weight columns (NWEIGHT1-NWEIGHT60)
        target_percentile : float
            Target percentile
            
        Returns
        -------
        dict
            Uncertainty estimates
        """
        quantile = target_percentile / 100
        
        # Main estimate
        main_threshold = compute_weighted_quantile(scores, weights, quantile)
        main_candidates = scores >= main_threshold
        main_n_candidates = main_candidates.sum()
        
        # Replicate estimates
        rep_thresholds = []
        rep_n_candidates = []
        rep_jaccard_vs_main = []
        
        for col in replicate_weights.columns:
            rep_w = replicate_weights[col].values
            rep_threshold = compute_weighted_quantile(scores, rep_w, quantile)
            rep_candidates = scores >= rep_threshold
            
            rep_thresholds.append(rep_threshold)
            rep_n_candidates.append(rep_candidates.sum())
            
            # Jaccard vs main
            intersection = np.sum(main_candidates & rep_candidates)
            union = np.sum(main_candidates | rep_candidates)
            jaccard = intersection / union if union > 0 else 0
            rep_jaccard_vs_main.append(jaccard)
        
        rep_thresholds = np.array(rep_thresholds)
        rep_n_candidates = np.array(rep_n_candidates)
        rep_jaccard_vs_main = np.array(rep_jaccard_vs_main)
        
        # Jackknife variance estimation
        # Var = (n-1)/n * sum((theta_i - theta_bar)^2)
        n = len(rep_thresholds)
        
        threshold_var = (n - 1) / n * np.sum((rep_thresholds - rep_thresholds.mean()) ** 2)
        threshold_se = np.sqrt(threshold_var)
        
        n_candidates_var = (n - 1) / n * np.sum((rep_n_candidates - rep_n_candidates.mean()) ** 2)
        n_candidates_se = np.sqrt(n_candidates_var)
        
        jaccard_var = (n - 1) / n * np.sum((rep_jaccard_vs_main - rep_jaccard_vs_main.mean()) ** 2)
        jaccard_se = np.sqrt(jaccard_var)
        
        # 95% CI
        z = self.z_score
        
        return {
            'threshold': {
                'estimate': main_threshold,
                'se': threshold_se,
                'ci_lower': main_threshold - z * threshold_se,
                'ci_upper': main_threshold + z * threshold_se
            },
            'n_candidates': {
                'estimate': main_n_candidates,
                'se': n_candidates_se,
                'ci_lower': main_n_candidates - z * n_candidates_se,
                'ci_upper': main_n_candidates + z * n_candidates_se
            },
            'jaccard': {
                'estimate': rep_jaccard_vs_main.mean(),
                'se': jaccard_se,
                'ci_lower': max(0, rep_jaccard_vs_main.mean() - z * jaccard_se),
                'ci_upper': min(1, rep_jaccard_vs_main.mean() + z * jaccard_se),
                'min': rep_jaccard_vs_main.min(),
                'max': rep_jaccard_vs_main.max()
            }
        }
    
    def compute_jaccard_overlap_with_ci(self,
                                         scores: np.ndarray,
                                         weights: np.ndarray,
                                         replicate_weights: pd.DataFrame,
                                         target_percentile: float = 90) -> Dict[str, Any]:
        """
        Compute Jaccard index and Overlap rate with 95% CI.
        
        Compares weighted vs unweighted candidate lists.
        
        Parameters
        ----------
        scores : array
            Policy scores
        weights : array
            Main weights (NWEIGHT)
        replicate_weights : DataFrame
            Replicate weights (NWEIGHT1-60)
        target_percentile : float
            Target percentile
            
        Returns
        -------
        dict
            Jaccard and Overlap with 95% CIs
        """
        quantile = target_percentile / 100
        z = self.z_score
        
        # Main estimates
        threshold_w = compute_weighted_quantile(scores, weights, quantile)
        threshold_uw = np.quantile(scores[~np.isnan(scores)], quantile)
        
        candidates_w = scores >= threshold_w
        candidates_uw = scores >= threshold_uw
        
        intersection = np.sum(candidates_w & candidates_uw)
        union = np.sum(candidates_w | candidates_uw)
        main_jaccard = intersection / union if union > 0 else 0
        
        # Dice coefficient = 2 * intersection / (|A| + |B|)
        n_w = candidates_w.sum()
        n_uw = candidates_uw.sum()
        main_dice = 2 * intersection / (n_w + n_uw) if (n_w + n_uw) > 0 else 0
        
        # Replicate estimates
        rep_jaccards = []
        rep_dices = []
        
        for col in replicate_weights.columns[:self.n_replicates]:
            rep_w = replicate_weights[col].values
            rep_threshold = compute_weighted_quantile(scores, rep_w, quantile)
            rep_candidates = scores >= rep_threshold
            
            rep_inter = np.sum(rep_candidates & candidates_uw)
            rep_union = np.sum(rep_candidates | candidates_uw)
            rep_jaccard = rep_inter / rep_union if rep_union > 0 else 0
            
            # Dice for replicate
            rep_n_w = rep_candidates.sum()
            rep_dice = 2 * rep_inter / (rep_n_w + n_uw) if (rep_n_w + n_uw) > 0 else 0
            
            rep_jaccards.append(rep_jaccard)
            rep_dices.append(rep_dice)
        
        rep_jaccards = np.array(rep_jaccards)
        rep_dices = np.array(rep_dices)
        n = len(rep_jaccards)
        
        # Jackknife variance (RECS SDR method: multiply by 4/n for successive difference)
        # Standard jackknife: (n-1)/n * sum((x - mean)^2)
        jaccard_var = (n - 1) / n * np.sum((rep_jaccards - rep_jaccards.mean()) ** 2)
        jaccard_se = np.sqrt(jaccard_var)
        
        dice_var = (n - 1) / n * np.sum((rep_dices - rep_dices.mean()) ** 2)
        dice_se = np.sqrt(dice_var)
        
        return {
            'jaccard': {
                'estimate': main_jaccard,
                'se': jaccard_se,
                'ci_lower': max(0, main_jaccard - z * jaccard_se),
                'ci_upper': min(1, main_jaccard + z * jaccard_se)
            },
            'overlap': {  # This is now Dice coefficient, not Containment
                'estimate': main_dice,
                'se': dice_se,
                'ci_lower': max(0, main_dice - z * dice_se),
                'ci_upper': min(1, main_dice + z * dice_se)
            },
            'n_weighted_candidates': int(candidates_w.sum()),
            'n_unweighted_candidates': int(candidates_uw.sum())
        }


def create_targeting_summary_table(analysis_results: Dict[str, Any],
                                    score_name: str = 'high_use',
                                    uncertainty_results: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Create summary table for policy targeting results with uncertainty.
    
    Parameters
    ----------
    analysis_results : dict
        Results from PolicyTargeting.run_full_analysis()
    score_name : str
        Which score to summarize
    uncertainty_results : dict, optional
        Uncertainty results from TargetingUncertainty
        
    Returns
    -------
    DataFrame
        Summary table with CIs
    """
    if score_name not in analysis_results['weighted_vs_unweighted']:
        raise ValueError(f"Score {score_name} not found in results")
    
    score_results = analysis_results['weighted_vs_unweighted'][score_name]
    overlap = score_results['overlap']
    
    rows = []
    
    # Jaccard Index with CI if available
    jaccard_val = f"{overlap['jaccard_index']:.3f}"
    if uncertainty_results and 'jaccard' in uncertainty_results:
        jac = uncertainty_results['jaccard']
        jaccard_val = f"{jac['estimate']:.3f} [{jac['ci_lower']:.3f}, {jac['ci_upper']:.3f}]"
    rows.append({'Metric': 'Jaccard Index', 'Value': jaccard_val, 
                 'Description': 'Intersection / Union of candidate sets'})
    
    # Overlap Rate with CI if available
    overlap_val = f"{overlap['overlap_rate']:.3f}"
    if uncertainty_results and 'overlap' in uncertainty_results:
        ovl = uncertainty_results['overlap']
        overlap_val = f"{ovl['estimate']:.3f} [{ovl['ci_lower']:.3f}, {ovl['ci_upper']:.3f}]"
    rows.append({'Metric': 'Overlap Rate', 'Value': overlap_val,
                 'Description': 'Dice coefficient: 2×|A∩B| / (|A|+|B|)'})
    
    # Recall and Containment metrics
    if 'recall_weighted' in overlap:
        rows.append({'Metric': 'Recall (Weighted)', 
                    'Value': f"{overlap['recall_weighted']:.3f}",
                    'Description': 'Fraction of weighted candidates also in unweighted'})
    if 'containment' in overlap:
        rows.append({'Metric': 'Containment',
                    'Value': f"{overlap['containment']:.3f}",
                    'Description': 'Intersection / min(|A|, |B|)'})
    
    # Other metrics
    rows.extend([
        {'Metric': 'Only in Weighted', 
         'Value': f"{overlap['only_weighted']} ({overlap['pct_only_weighted']:.1f}%)",
         'Description': 'Candidates selected only with weights'},
        {'Metric': 'Only in Unweighted',
         'Value': f"{overlap['only_unweighted']} ({overlap['pct_only_unweighted']:.1f}%)",
         'Description': 'Candidates selected only without weights'},
    ])
    
    # Thresholds with correct units
    if 'high_intensity' in score_name:
        threshold_unit = 'kBTU/ft²'
    else:
        threshold_unit = 'kBTU'
    
    rows.extend([
        {'Metric': 'Weighted Threshold',
         'Value': f"{score_results['weighted_threshold']:,.0f} {threshold_unit}",
         'Description': f"Weighted 90th percentile cutoff"},
        {'Metric': 'Unweighted Threshold',
         'Value': f"{score_results['unweighted_threshold']:,.0f} {threshold_unit}",
         'Description': f"Unweighted 90th percentile cutoff"}
    ])
    
    result_df = pd.DataFrame(rows)
    result_df.attrs['note'] = (
        f"Policy score: {score_name}. Top 10% candidates defined by weighted/unweighted "
        "90th percentile. 95% CIs from replicate-weight jackknife (n=60)."
    )
    
    return result_df


class PolicyMetricsEvaluator:
    """
    Policy-oriented evaluation metrics using TRUE consumption values.
    
    These metrics evaluate how well the model identifies ACTUAL high consumers,
    not just agreement between weighted/unweighted lists.
    
    Key metrics:
    - Precision@k: Of top-k by prediction, what fraction are truly high consumers?
    - Recall@k: Of true high consumers, what fraction are in top-k predictions?
    - Overlap@k: Jaccard between predicted top-k and true top-k
    - Lift@k: How much better than random selection?
    - Cost of underprediction: Asymmetric loss on top decile
    """
    
    def __init__(self, k_percentile: float = 10):
        """
        Initialize evaluator.
        
        Parameters
        ----------
        k_percentile : float
            Top percentage to target (default 10 = top 10%)
        """
        self.k_percentile = k_percentile
        self.k_quantile = 1 - k_percentile / 100  # 0.9 for top 10%
    
    def compute_all_metrics(self,
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            weights: np.ndarray) -> Dict[str, float]:
        """
        Compute all policy-oriented metrics.
        
        Parameters
        ----------
        y_true : array
            True consumption values
        y_pred : array
            Predicted values
        weights : array
            Sample weights (NWEIGHT)
            
        Returns
        -------
        dict
            Dictionary of policy metrics
        """
        results = {}
        
        # Use weighted quantiles to define "true high consumers"
        true_threshold = compute_weighted_quantile(y_true, weights, self.k_quantile)
        pred_threshold = compute_weighted_quantile(y_pred, weights, self.k_quantile)
        
        # Define sets
        true_high = y_true >= true_threshold
        pred_high = y_pred >= pred_threshold
        
        # Basic counts
        n_true_high = np.sum(weights[true_high])  # Weighted count
        n_pred_high = np.sum(weights[pred_high])
        n_both = np.sum(weights[true_high & pred_high])
        
        # Precision@k: Of predicted high, what fraction are truly high?
        precision = n_both / n_pred_high if n_pred_high > 0 else 0
        
        # Recall@k: Of truly high, what fraction were predicted high?
        recall = n_both / n_true_high if n_true_high > 0 else 0
        
        # F1@k: Harmonic mean
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Overlap@k (Jaccard between predicted and true top-k)
        union = np.sum(weights[true_high | pred_high])
        jaccard = n_both / union if union > 0 else 0
        
        # Lift@k: How much better than random?
        # Random baseline: k% of predictions would hit k% × k% = (k%)² of true high
        base_rate = self.k_percentile / 100
        lift = precision / base_rate if base_rate > 0 else 0
        
        # Mean Rank of true high consumers in predictions (lower is better)
        pred_ranks = np.argsort(np.argsort(-y_pred))  # Rank 0 = highest prediction
        mean_rank_true_high = np.average(pred_ranks[true_high], weights=weights[true_high])
        
        # Normalized Discounted Cumulative Gain (NDCG) for ranking quality
        ndcg = self._compute_ndcg(y_true, y_pred, weights)
        
        # Cost of underprediction in top decile
        top_decile_mask = y_true >= true_threshold
        if np.any(top_decile_mask):
            residuals_top = y_pred[top_decile_mask] - y_true[top_decile_mask]
            weights_top = weights[top_decile_mask]
            
            # Weighted mean underprediction (negative = underprediction)
            underpred = np.sum(weights_top * residuals_top) / np.sum(weights_top)
            underpred_pct = underpred / np.average(y_true[top_decile_mask], weights=weights_top) * 100
            
            # Asymmetric loss: penalize underprediction 2x vs overprediction
            asymmetric_loss = np.sum(weights_top * np.where(
                residuals_top < 0, 
                2 * residuals_top**2,  # 2x penalty for under
                residuals_top**2       # 1x for over
            )) / np.sum(weights_top)
        else:
            underpred = 0
            underpred_pct = 0
            asymmetric_loss = 0
        
        results = {
            'precision_at_k': precision,
            'recall_at_k': recall,
            'f1_at_k': f1,
            'jaccard_at_k': jaccard,
            'lift_at_k': lift,
            'mean_rank_true_high': mean_rank_true_high,
            'ndcg': ndcg,
            'top_decile_underpred': underpred,
            'top_decile_underpred_pct': underpred_pct,
            'asymmetric_loss': asymmetric_loss,
            'true_threshold': true_threshold,
            'pred_threshold': pred_threshold,
            'k_percentile': self.k_percentile
        }
        
        return results
    
    def _compute_ndcg(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      weights: np.ndarray, k: int = None) -> float:
        """
        Compute Normalized Discounted Cumulative Gain.
        
        Measures ranking quality: do high true values rank high in predictions?
        
        Parameters
        ----------
        y_true : array
            True values (relevance scores)
        y_pred : array
            Predicted values (used for ranking)
        weights : array
            Sample weights
        k : int, optional
            Only consider top-k positions (default: all)
            
        Returns
        -------
        float
            NDCG score between 0 and 1
        """
        n = len(y_true)
        k = k or int(n * self.k_percentile / 100)
        
        # Rank by predictions (descending)
        pred_order = np.argsort(-y_pred)[:k]
        
        # Ideal ranking (by true values)
        ideal_order = np.argsort(-y_true)[:k]
        
        # DCG: sum of relevance / log2(rank + 2)
        dcg = 0
        for i, idx in enumerate(pred_order):
            rel = y_true[idx] * weights[idx]  # Weighted relevance
            dcg += rel / np.log2(i + 2)
        
        # Ideal DCG
        idcg = 0
        for i, idx in enumerate(ideal_order):
            rel = y_true[idx] * weights[idx]
            idcg += rel / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0
    
    def compute_with_ci(self,
                        y_true: np.ndarray,
                        y_pred: np.ndarray,
                        weights: np.ndarray,
                        replicate_weights: np.ndarray,
                        alpha: float = 0.05) -> Dict[str, Dict[str, float]]:
        """
        Compute policy metrics with confidence intervals using replicate weights.
        
        Parameters
        ----------
        y_true : array
            True consumption values
        y_pred : array
            Predicted values
        weights : array
            Primary sample weights (NWEIGHT)
        replicate_weights : array
            Replicate weights (n_samples × n_replicates)
        alpha : float
            Significance level for CI (default 0.05 for 95% CI)
            
        Returns
        -------
        dict
            Dictionary of {metric: {'estimate': x, 'se': x, 'ci_lower': x, 'ci_upper': x}}
        """
        # Main estimate
        main = self.compute_all_metrics(y_true, y_pred, weights)
        
        # Replicate estimates
        n_reps = replicate_weights.shape[1]
        rep_results = {k: [] for k in main.keys()}
        
        for r in range(n_reps):
            rep_metrics = self.compute_all_metrics(y_true, y_pred, replicate_weights[:, r])
            for k, v in rep_metrics.items():
                rep_results[k].append(v)
        
        # Compute SE and CI using jackknife formula
        results = {}
        z = 1.96 if alpha == 0.05 else 2.576  # 95% or 99% CI
        
        for metric, main_val in main.items():
            reps = np.array(rep_results[metric])
            # Jackknife variance: ((n-1)/n) * sum((rep - main)^2)
            variance = ((n_reps - 1) / n_reps) * np.sum((reps - main_val)**2)
            se = np.sqrt(variance)
            
            results[metric] = {
                'estimate': main_val,
                'se': se,
                'ci_lower': main_val - z * se,
                'ci_upper': main_val + z * se
            }
        
        return results
    
    def create_summary_table(self, metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Create summary table of policy metrics.
        
        Parameters
        ----------
        metrics : dict
            Output from compute_with_ci
            
        Returns
        -------
        DataFrame
            Formatted summary table
        """
        rows = []
        
        key_metrics = [
            ('precision_at_k', 'Precision@k', 'Of predicted top-k%, fraction truly high'),
            ('recall_at_k', 'Recall@k', 'Of truly high, fraction in top-k predictions'),
            ('f1_at_k', 'F1@k', 'Harmonic mean of Precision and Recall'),
            ('jaccard_at_k', 'Jaccard@k', 'Overlap between predicted and true top-k'),
            ('lift_at_k', 'Lift@k', 'Improvement over random (Precision / base rate)'),
            ('ndcg', 'NDCG', 'Ranking quality (1.0 = perfect ranking)'),
            ('top_decile_underpred_pct', 'Top-10% Bias (%)', 'Underprediction in top decile'),
            ('asymmetric_loss', 'Asymmetric Loss', 'Under-prediction penalized 2× over-prediction'),
        ]
        
        for key, name, desc in key_metrics:
            if key in metrics:
                m = metrics[key]
                if key in ['precision_at_k', 'recall_at_k', 'f1_at_k', 'jaccard_at_k', 'ndcg']:
                    val_str = f"{m['estimate']:.3f} [{m['ci_lower']:.3f}, {m['ci_upper']:.3f}]"
                elif key == 'lift_at_k':
                    val_str = f"{m['estimate']:.2f}× [{m['ci_lower']:.2f}, {m['ci_upper']:.2f}]"
                elif key == 'top_decile_underpred_pct':
                    val_str = f"{m['estimate']:+.1f}% [{m['ci_lower']:+.1f}, {m['ci_upper']:+.1f}]"
                else:
                    val_str = f"{m['estimate']:,.0f} [{m['ci_lower']:,.0f}, {m['ci_upper']:,.0f}]"
                
                rows.append({
                    'Metric': name,
                    'Value (95% CI)': val_str,
                    'Description': desc
                })
        
        df = pd.DataFrame(rows)
        k = metrics.get('k_percentile', {}).get('estimate', 10)
        df.attrs['note'] = (
            f"Policy metrics for top-{k:.0f}% targeting. 'True high' defined by weighted "
            f"90th percentile of observed consumption. CIs from replicate-weight jackknife."
        )
        return df
