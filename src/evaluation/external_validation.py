"""External validation helpers.

The core pipeline in this repo uses RECS 2020. To address reviewer requests for
external validation (e.g., against utility billing data), this module provides a
small, explicit interface you can plug into once you have a household-level link.

Input billing DataFrame (minimal):
- id_col: household identifier that also exists in the RECS-derived dataset
- y_bill_col: annual heating-energy proxy (same unit as your model output, or a
  consistently scaled proxy)
Optional:
- weight_col: optional external weights (if any)

This module does NOT attempt to:
- infer units
- disaggregate end-uses
- match across datasets (you must join externally)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.evaluation.metrics import WeightedMetrics
from src.utils.helpers import logger


@dataclass
class ExternalValidationResult:
    n_matched: int
    metrics: Dict[str, float]
    correlation: float


def align_predictions_with_bills(
    df_recs: pd.DataFrame,
    df_bills: pd.DataFrame,
    id_col: str,
    pred_col: str,
    y_bill_col: str,
    weight_col_recs: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Align predictions from RECS frame with external bills.

    Returns (y_bill, y_pred, weights) after inner join on id_col.
    """
    if id_col not in df_recs.columns:
        raise KeyError(f"{id_col} not found in df_recs")
    if id_col not in df_bills.columns:
        raise KeyError(f"{id_col} not found in df_bills")
    if pred_col not in df_recs.columns:
        raise KeyError(f"{pred_col} not found in df_recs")
    if y_bill_col not in df_bills.columns:
        raise KeyError(f"{y_bill_col} not found in df_bills")

    merged = df_recs[[id_col, pred_col] + ([weight_col_recs] if weight_col_recs else [])].merge(
        df_bills[[id_col, y_bill_col]],
        on=id_col,
        how="inner",
    )

    y_bill = merged[y_bill_col].astype(float).values
    y_pred = merged[pred_col].astype(float).values

    w = None
    if weight_col_recs:
        w = merged[weight_col_recs].astype(float).values

    return y_bill, y_pred, w


def compute_external_validation_metrics(
    y_bill: np.ndarray,
    y_pred: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> ExternalValidationResult:
    """Compute validation metrics vs external bills."""
    metrics_calc = WeightedMetrics()

    # Filter invalid
    mask = np.isfinite(y_bill) & np.isfinite(y_pred)
    if weights is not None:
        mask = mask & np.isfinite(weights) & (weights >= 0)

    yb = y_bill[mask]
    yp = y_pred[mask]
    w = weights[mask] if weights is not None else None

    if len(yb) == 0:
        raise ValueError("No valid matched rows for external validation")

    metrics = {
        "rmse": metrics_calc.weighted_rmse(yb, yp, w) if w is not None else float(np.sqrt(np.mean((yb - yp) ** 2))),
        "mae": metrics_calc.weighted_mae(yb, yp, w) if w is not None else float(np.mean(np.abs(yb - yp))),
        "r2": metrics_calc.weighted_r2(yb, yp, w) if w is not None else float(1 - np.sum((yb - yp) ** 2) / np.sum((yb - np.mean(yb)) ** 2)),
    }

    # Pearson correlation (unweighted)
    corr = float(np.corrcoef(yb, yp)[0, 1]) if len(yb) > 1 else float("nan")

    logger.info(f"External validation matched n={len(yb)} rows; rmse={metrics['rmse']:.2f}; r2={metrics['r2']:.3f}; corr={corr:.3f}")

    return ExternalValidationResult(n_matched=int(len(yb)), metrics=metrics, correlation=corr)
