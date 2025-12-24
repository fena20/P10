"""Advanced / state-of-the-art baselines.

These are OPTIONAL baselines intended to address reviewer requests for
comparisons beyond LightGBM/EBM.

Design goals:
- Keep the core pipeline dependency-light (no hard dependency on torch/xgboost/catboost)
- Provide drop-in sklearn-style estimators when optional deps are installed

Notes
-----
1) RECS heating demand prediction here is cross-sectional (household-level), not a time series.
   "Transformer forecasters" are typically for temporal sequences; here we provide a
   *tabular transformer* baseline (MLP/Transformer over features).
2) If dependencies are missing, these classes raise a clear ImportError at fit time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin

from src.utils.helpers import logger


class XGBoostRegressorModel(BaseEstimator, RegressorMixin):
    """XGBoost regressor baseline (optional dependency)."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}
        self.model_ = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        try:
            import xgboost as xgb
        except Exception as e:
            raise ImportError("xgboost is required for XGBoostRegressorModel. Install with: pip install xgboost") from e

        X_arr = X.values if isinstance(X, pd.DataFrame) else X

        default = {
            "n_estimators": 800,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "objective": "reg:squarederror",
            "random_state": 42,
            "n_jobs": -1,
        }
        cfg = default.copy()
        cfg.update(self.params)

        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight

        # Train, with optional GPU->CPU fallback.
        try:
            self.model_ = xgb.XGBRegressor(**cfg)
            self.model_.fit(X_arr, y, **fit_kwargs)
        except Exception as e:
            uses_gpu = (
                str(cfg.get("tree_method", "")).lower().startswith("gpu")
                or "cuda" in str(cfg.get("device", "")).lower()
                or "gpu" in str(cfg.get("predictor", "")).lower()
                or ("gpu_id" in cfg)
            )
            if uses_gpu:
                logger.warning(f"XGBoost GPU training failed ({e}); falling back to CPU.")
                cpu_cfg = dict(cfg)
                for k in ["tree_method", "predictor", "gpu_id", "device"]:
                    cpu_cfg.pop(k, None)
                cpu_cfg.setdefault("tree_method", "hist")
                self.model_ = xgb.XGBRegressor(**cpu_cfg)
                self.model_.fit(X_arr, y, **fit_kwargs)
            else:
                raise
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model is not fitted")
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        return self.model_.predict(X_arr)


class CatBoostRegressorModel(BaseEstimator, RegressorMixin):
    """CatBoost regressor baseline (optional dependency)."""

    def __init__(self, params: Optional[Dict[str, Any]] = None, verbose: bool = False):
        self.params = params or {}
        self.verbose = verbose
        self.model_ = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        try:
            from catboost import CatBoostRegressor
        except Exception as e:
            raise ImportError("catboost is required for CatBoostRegressorModel. Install with: pip install catboost") from e

        X_arr = X.values if isinstance(X, pd.DataFrame) else X

        default = {
            "loss_function": "RMSE",
            "depth": 8,
            "learning_rate": 0.05,
            "iterations": 4000,
            "random_seed": 42,
            "verbose": 200 if self.verbose else False,
        }
        cfg = default.copy()
        cfg.update(self.params)

        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight

        # Train, with optional GPU->CPU fallback.
        try:
            self.model_ = CatBoostRegressor(**cfg)
            self.model_.fit(X_arr, y, **fit_kwargs)
        except Exception as e:
            uses_gpu = str(cfg.get("task_type", "")).upper() == "GPU"
            if uses_gpu:
                logger.warning(f"CatBoost GPU training failed ({e}); falling back to CPU.")
                cpu_cfg = dict(cfg)
                for k in ["task_type", "devices"]:
                    cpu_cfg.pop(k, None)
                cpu_cfg["task_type"] = "CPU"
                self.model_ = CatBoostRegressor(**cpu_cfg)
                self.model_.fit(X_arr, y, **fit_kwargs)
            else:
                raise
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model is not fitted")
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        return self.model_.predict(X_arr)


class TabularTransformerModel(BaseEstimator, RegressorMixin):
    """A lightweight tabular transformer baseline (optional dependency: torch).

    This is intentionally simple (no categorical embedding automation). It expects
    all features already numeric (as produced by the pipeline's FeatureBuilder).

    Use for reviewer-requested "transformer" comparison in a cross-sectional setting.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 512,
        epochs: int = 50,
        random_state: int = 42,
        device: Optional[str] = None,
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state
        self.device = device

        self.model_ = None
        self.n_features_ = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except Exception as e:
            raise ImportError("torch is required for TabularTransformerModel. Install with: pip install torch") from e

        rng = np.random.default_rng(self.random_state)

        X_arr = X.values.astype(np.float32) if isinstance(X, pd.DataFrame) else X.astype(np.float32)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        self.n_features_ = X_arr.shape[1]

        # Tokenize each feature as a 'token' via a learned linear projection.
        class FeatureTokenizer(nn.Module):
            def __init__(self, n_features: int, d_model: int):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
                self.bias = nn.Parameter(torch.zeros(n_features, d_model))

            def forward(self, x):
                # x: (B, F) -> (B, F, D)
                return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)

        class TinyTabTransformer(nn.Module):
            def __init__(self, n_features: int, d_model: int, n_heads: int, n_layers: int, dropout: float):
                super().__init__()
                self.tok = FeatureTokenizer(n_features, d_model)
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
                self.head = nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, 1),
                )

            def forward(self, x):
                z = self.tok(x)  # (B, F, D)
                z = self.enc(z)  # (B, F, D)
                z = z.mean(dim=1)  # pool over features
                return self.head(z)

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.random_state)

        self.model_ = TinyTabTransformer(self.n_features_, self.d_model, self.n_heads, self.n_layers, self.dropout).to(device)

        X_t = torch.from_numpy(X_arr)
        y_t = torch.from_numpy(y_arr)

        if sample_weight is not None:
            w_t = torch.from_numpy(np.asarray(sample_weight, dtype=np.float32).reshape(-1, 1))
        else:
            w_t = torch.ones_like(y_t)

        ds = TensorDataset(X_t, y_t, w_t)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        import torch.nn.functional as F

        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.model_.train()
        for epoch in range(self.epochs):
            total = 0.0
            wsum = 0.0
            for xb, yb, wb in dl:
                xb = xb.to(device)
                yb = yb.to(device)
                wb = wb.to(device)

                pred = self.model_(xb)
                # Weighted MSE
                loss = (wb * (pred - yb) ** 2).sum() / (wb.sum() + 1e-8)

                opt.zero_grad()
                loss.backward()
                opt.step()

                total += float(loss.detach().cpu())
                wsum += 1.0

            if (epoch + 1) % 10 == 0:
                logger.info(f"TabularTransformer epoch {epoch+1}/{self.epochs} - loss={total/max(wsum,1.0):.6f}")

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model is not fitted")

        try:
            import torch
        except Exception as e:
            raise ImportError("torch is required for TabularTransformerModel") from e

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

        X_arr = X.values.astype(np.float32) if isinstance(X, pd.DataFrame) else X.astype(np.float32)
        X_t = torch.from_numpy(X_arr).to(device)

        self.model_.eval()
        with torch.no_grad():
            preds = self.model_(X_t).detach().cpu().numpy().reshape(-1)
        return preds
