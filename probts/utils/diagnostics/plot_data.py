from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch


@dataclass
class DatasetPlotConfig:
    dataset_name: str
    num_features: int
    patch_size: int
    input_window: int
    horizon_window: int
    default_sample_idx: int = 0
    interval_z: float = 1.96
    feature_names: Optional[Sequence[str]] = None
    units: Optional[Sequence[str]] = None


@dataclass
class ForecastBatch:
    y_true: np.ndarray
    y_mu: np.ndarray
    y_sigma: Optional[np.ndarray] = None
    context_true: Optional[np.ndarray] = None
    context_mu: Optional[np.ndarray] = None


@dataclass
class HorizonResiduals:
    mae_by_horizon: np.ndarray
    mse_by_horizon: np.ndarray


class PlotDataCollector:
    @staticmethod
    def collect_forecast_batch(
        x_h_true, x_h_mu, x_h_sigma=None, x_c_true=None, x_c_mu=None,
        num_features=None, patch_size=1, inverse_scale=False, patched=False,
    ) -> ForecastBatch:
        if inverse_scale:
            raise ValueError("Inverse-transform tensors with the dataset scaler before collecting plots")

        def convert(value):
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            array = np.asarray(value)
            if patched:
                array = array.reshape(array.shape[0], -1, num_features)
            if array.ndim != 3:
                raise ValueError("Forecast tensors must have shape (batch, time, features)")
            if num_features is not None and array.shape[-1] != num_features:
                raise ValueError("Forecast feature count does not match num_features")
            return array

        batch = ForecastBatch(
            y_true=convert(x_h_true), y_mu=convert(x_h_mu),
            y_sigma=convert(x_h_sigma), context_true=convert(x_c_true),
            context_mu=convert(x_c_mu),
        )
        if batch.y_true.shape != batch.y_mu.shape:
            raise ValueError("Forecast and truth shapes must match")
        if batch.y_sigma is not None and batch.y_sigma.shape != batch.y_mu.shape:
            raise ValueError("Forecast standard deviation shape must match the mean")
        return batch

    @staticmethod
    def compute_horizon_residuals(batch: ForecastBatch) -> HorizonResiduals:
        error = batch.y_mu - batch.y_true
        return HorizonResiduals(
            mae_by_horizon=np.mean(np.abs(error), axis=(0, 2)),
            mse_by_horizon=np.mean(np.square(error), axis=(0, 2)),
        )
