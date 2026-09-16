# Ported verbatim from the thesis K2VAE repo (Project/src/plot_data.py) so the
# same diagnostic figures can be produced from this repo's reference K2VAE
# implementation for direct comparison. See K2VAEDiagnosticsCallback for how
# this repo's Lightning training loop feeds this module.
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import torch
from einops import rearrange


@dataclass
class ForecastBatch:
    """Canonical plotting payload.

    Shapes:
        context_true/context_mu: (B, T, V)
        y_true/y_mu/y_sigma: (B, H, V)
    """

    y_true: np.ndarray
    y_mu: np.ndarray
    y_sigma: Optional[np.ndarray] = None
    context_true: Optional[np.ndarray] = None
    context_mu: Optional[np.ndarray] = None
    sample_ids: Optional[List[int]] = None


@dataclass
class HorizonResiduals:
    """Per-horizon error summaries."""

    mae_by_horizon: np.ndarray  # (H,)
    mse_by_horizon: np.ndarray  # (H,)


@dataclass
class DatasetPlotConfig:
    """Metadata required to decode and label plots across datasets."""

    dataset_name: str
    num_features: int
    patch_size: int
    input_window: int
    horizon_window: int
    feature_names: Optional[List[str]] = None
    units: Optional[List[str]] = None
    inverse_scale: bool = True
    default_feature_idx: int = 0
    default_sample_idx: int = 0
    interval_z: float = 1.96


class PlotDataCollector:
    """Converts model/training tensors into plot-ready numpy payloads."""

    def __init__(self, scaler: Optional[Any] = None):
        self.scaler = scaler

    @staticmethod
    def to_numpy(tensor: torch.Tensor) -> np.ndarray:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
        return tensor.detach().cpu().float().numpy()

    @staticmethod
    def validate_patched_shape(x: np.ndarray, num_features: int, patch_size: int) -> None:
        """Validate (B, input_dim, patches), where input_dim == num_features * patch_size."""
        if x.ndim != 3:
            raise ValueError(f"Expected 3D shape (B, input_dim, patches), got {x.shape}")

        input_dim = x.shape[1]
        if input_dim % patch_size != 0:
            raise ValueError(
                f"input_dim={input_dim} must be divisible by patch_size={patch_size}"
            )

        inferred_features = input_dim // patch_size
        if inferred_features != num_features:
            raise ValueError(
                "num_features mismatch: "
                f"inferred={inferred_features}, expected={num_features}"
            )

    @staticmethod
    def unpatch_to_time(x: np.ndarray, num_features: int, patch_size: int) -> np.ndarray:
        """Inverse of k2vae_integration: (B, N*s, p) -> (B, p*s, N)."""
        PlotDataCollector.validate_patched_shape(x, num_features, patch_size)
        return rearrange(x, "B (N s) p -> B (p s) N", N=num_features, s=patch_size)

    def inverse_scale_points(self, arr_btn: np.ndarray) -> np.ndarray:
        """Inverse-scale point estimates (means/targets) in (B, T, N)."""
        if self.scaler is None:
            return arr_btn
        if not hasattr(self.scaler, "inverse_transform"):
            raise AttributeError("Scaler must provide inverse_transform for point estimates")

        b, t, n = arr_btn.shape
        flat = arr_btn.reshape(b * t, n)
        flat_inv = self.scaler.inverse_transform(flat)
        return flat_inv.reshape(b, t, n)

    def inverse_scale_sigma(self, sigma_bhn: np.ndarray) -> np.ndarray:
        """Inverse-scale standard deviation in (B, H, N) using scale only.

        Important: sigma is spread, not location, so we only multiply by scaler.scale_.
        """
        if self.scaler is None:
            return sigma_bhn
        if not hasattr(self.scaler, "scale_"):
            raise AttributeError("Scaler must provide scale_ for sigma inverse scaling")

        b, h, n = sigma_bhn.shape
        flat = sigma_bhn.reshape(b * h, n)
        flat_inv = flat * np.asarray(self.scaler.scale_)
        return flat_inv.reshape(b, h, n)

    def collect_forecast_batch(
        self,
        x_h_true: torch.Tensor,
        x_h_mu: torch.Tensor,
        x_h_sigma: Optional[torch.Tensor] = None,
        x_c_true: Optional[torch.Tensor] = None,
        x_c_mu: Optional[torch.Tensor] = None,
        *,
        num_features: int,
        patch_size: int,
        inverse_scale: bool = True,
        patched: bool = True,
        sample_ids: Optional[List[int]] = None,
    ) -> ForecastBatch:
        """Build plot-ready ForecastBatch payload.

        Steps:
        1) tensor -> numpy
          2) convert patched (B, input_dim, p) or raw (B, time, N)
              tensors to (B, time, N)
        3) optional inverse scaling
           - points use inverse_transform
           - sigma uses scale_ only
        """
        def to_time(tensor: torch.Tensor) -> np.ndarray:
            """Convert one model payload to the canonical time-major shape."""
            values = self.to_numpy(tensor)
            if patched:
                return self.unpatch_to_time(values, num_features, patch_size)
            # Reference K2VAE (and RevIN-integrated K2VAE) already returns
            # temporal layout, so no inverse patch operation is appropriate
            # in this branch.
            if values.ndim != 3 or values.shape[2] != num_features:
                raise ValueError(f"Expected raw shape (B, time, {num_features}), got {values.shape}")
            return values

        y_true = to_time(x_h_true)
        y_mu = to_time(x_h_mu)

        y_sigma = None
        if x_h_sigma is not None:
            y_sigma = to_time(x_h_sigma)

        context_true = None
        if x_c_true is not None:
            context_true = to_time(x_c_true)

        context_mu = None
        if x_c_mu is not None:
            context_mu = to_time(x_c_mu)

        if inverse_scale:
            y_true = self.inverse_scale_points(y_true)
            y_mu = self.inverse_scale_points(y_mu)

            if context_true is not None:
                context_true = self.inverse_scale_points(context_true)
            if context_mu is not None:
                context_mu = self.inverse_scale_points(context_mu)

            if y_sigma is not None:
                y_sigma = self.inverse_scale_sigma(y_sigma)

        return ForecastBatch(
            y_true=y_true,
            y_mu=y_mu,
            y_sigma=y_sigma,
            context_true=context_true,
            context_mu=context_mu,
            sample_ids=sample_ids,
        )

    @staticmethod
    def compute_horizon_residuals(batch: ForecastBatch) -> HorizonResiduals:
        """Compute MAE and MSE per horizon step from y_true and y_mu."""
        if batch.y_true.shape != batch.y_mu.shape:
            raise ValueError(
                f"Shape mismatch: y_true {batch.y_true.shape} vs y_mu {batch.y_mu.shape}"
            )

        err = batch.y_true - batch.y_mu  # (B, H, V)
        mae_h = np.mean(np.abs(err), axis=(0, 2))
        mse_h = np.mean(err ** 2, axis=(0, 2))

        return HorizonResiduals(mae_by_horizon=mae_h, mse_by_horizon=mse_h)

    @staticmethod
    def load_dataset_plot_config(path: Union[str, Path]) -> DatasetPlotConfig:
        """Load and validate dataset plot metadata YAML."""
        import yaml

        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        required = [
            "dataset_name",
            "num_features",
            "patch_size",
            "input_window",
            "horizon_window",
        ]
        missing = [k for k in required if k not in raw]
        if missing:
            raise ValueError(f"Missing required YAML keys: {missing}")

        import dataclasses
        known_fields = {f.name for f in dataclasses.fields(DatasetPlotConfig)}
        cfg = DatasetPlotConfig(**{k: v for k, v in raw.items() if k in known_fields})

        if cfg.input_window % cfg.patch_size != 0:
            raise ValueError("input_window must be divisible by patch_size")
        if cfg.horizon_window % cfg.patch_size != 0:
            raise ValueError("horizon_window must be divisible by patch_size")

        if cfg.feature_names is not None and len(cfg.feature_names) != cfg.num_features:
            raise ValueError("feature_names length must equal num_features")

        if cfg.units is not None and len(cfg.units) != cfg.num_features:
            raise ValueError("units length must equal num_features")

        return cfg
