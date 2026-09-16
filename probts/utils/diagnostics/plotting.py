# plot_loss_curves / plot_forecast_vs_truth / plot_residual_by_horizon /
# EpochHistory are ported verbatim from the thesis K2VAE repo
# (Project/src/plotting.py), so forecast/residual figures come out in the
# same format on both sides.
#
# ReferenceEpochHistory and plot_reference_loss_curves are new: this repo's
# ProbTSForecastModule trains K2VAE on the ELBO (see
# probts.model.forecaster.prob_forecaster.k2vae.k2VAEModel.loss) but
# validates/checkpoints on CRPS-family sampling metrics computed by
# Evaluator (see forecast_module.py's evaluate()), not on an ELBO. Those two
# quantities live on different scales and aren't the same objective, so
# unlike the thesis repo's plot_loss_curves (which overlays train_total vs
# val_total on one axis), the val side is plotted on its own panel rather
# than forced onto the same axis as the train ELBO.
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import math
import matplotlib.pyplot as plt
import numpy as np

from .plot_data import DatasetPlotConfig, ForecastBatch, HorizonResiduals


@dataclass
class EpochHistory:
    """Container for train/validation metrics tracked per epoch."""

    epoch: List[int]
    train_total: List[float]
    val_total: List[float]
    train_elbo: Optional[List[float]] = None
    val_elbo: Optional[List[float]] = None
    train_nll: Optional[List[float]] = None
    val_nll: Optional[List[float]] = None
    train_kl: Optional[List[float]] = None
    val_kl: Optional[List[float]] = None
    train_rec: Optional[List[float]] = None
    val_rec: Optional[List[float]] = None
    skipped_batches: Optional[List[int]] = None


@dataclass
class ReferenceEpochHistory:
    """Per-epoch metrics for this repo's K2VAE, as actually trained/validated.

    train_* are the ELBO components from k2VAEModel.loss() (rec_loss,
    post_loss, kld_loss, and their weighted sum). val_* are whatever
    Evaluator computed that epoch and ProbTSForecastModule logged under
    val_<NAME> (see save_utils.update_metrics); CRPS is the one the
    ModelCheckpoint callback in run.py actually monitors, the rest are
    included for context.
    """

    epoch: List[int]
    train_total: List[float]
    train_rec: List[float]
    train_nll: List[float]
    train_kl: List[float]
    val_CRPS: List[float]
    val_MSE: Optional[List[float]] = None
    val_NRMSE: Optional[List[float]] = None
    val_MASE: Optional[List[float]] = None
    val_ND: Optional[List[float]] = None


class PlotManager:
    """Pure plotting layer: consumes payloads, never calls model/trainer internals."""

    def __init__(self, cfg: DatasetPlotConfig, run_id: str, output_root: str = "artifacts/plots"):
        self.cfg = cfg
        self.run_id = run_id
        self.out_dir = Path(output_root) / run_id
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def save(self, fig: plt.Figure, name: str) -> Path:
        """Save figure and close it to avoid memory leaks in long plotting loops."""
        out_path = self.out_dir / f"{name}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def plot_loss_curves(self, history: EpochHistory, show_components: bool = True) -> List[plt.Figure]:
        """Create total loss curve plus optional ELBO component curves."""
        if not (len(history.epoch) == len(history.train_total) == len(history.val_total)):
            raise ValueError("epoch, train_total, and val_total must be the same length")

        figures: List[plt.Figure] = []
        # A one-epoch history has no visible line segment; mark its point so
        # smoke-test plots still show the recorded values.
        marker = "o" if len(history.epoch) == 1 else None

        fig_total, ax = plt.subplots(figsize=(8, 4))
        ax.plot(history.epoch, history.train_total, marker=marker, label="train_total")
        ax.plot(history.epoch, history.val_total, marker=marker, label="val_total")
        if len(history.epoch) == 1:
            ax.set_xlim(history.epoch[0] - 0.5, history.epoch[0] + 0.5)
        ax.set_title(f"{self.cfg.dataset_name}: Total Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)
        ax.legend()
        figures.append(fig_total)

        if show_components:
            comp_names = ["elbo", "nll", "kl", "rec"]
            train_map = {
                "elbo": history.train_elbo,
                "nll": history.train_nll,
                "kl": history.train_kl,
                "rec": history.train_rec,
            }
            val_map = {
                "elbo": history.val_elbo,
                "nll": history.val_nll,
                "kl": history.val_kl,
                "rec": history.val_rec,
            }

            available = [name for name in comp_names if train_map[name] is not None and val_map[name] is not None]
            if available:
                fig_comp, axes = plt.subplots(len(available), 1, figsize=(8, 3 * len(available)), sharex=True)
                if len(available) == 1:
                    axes = [axes]
                for ax_i, name in zip(axes, available):
                    ax_i.plot(history.epoch, train_map[name], marker=marker, label=f"train_{name}")
                    ax_i.plot(history.epoch, val_map[name], marker=marker, label=f"val_{name}")
                    if len(history.epoch) == 1:
                        ax_i.set_xlim(history.epoch[0] - 0.5, history.epoch[0] + 0.5)
                    ax_i.set_ylabel(name)
                    ax_i.grid(alpha=0.3)
                    ax_i.legend()
                axes[-1].set_xlabel("Epoch")
                fig_comp.suptitle(f"{self.cfg.dataset_name}: ELBO Components")
                figures.append(fig_comp)

        return figures

    def plot_reference_loss_curves(self, history: ReferenceEpochHistory) -> List[plt.Figure]:
        """Reference-repo counterpart to plot_loss_curves.

        Produces the same two output figures (so run.py-style
        `loss_curves_{i}` naming lines up with the thesis repo), but split so
        train ELBO and val sampling metrics each get axes that make sense for
        their own scale:
          - figure 0: train total ELBO loss only (no val_total exists here)
          - figure 1: train rec/nll/kl components stacked above a shared
            panel of whichever val_* sampling metrics were recorded
        """
        n = len(history.epoch)
        if not (n == len(history.train_total) == len(history.train_rec)
                == len(history.train_nll) == len(history.train_kl) == len(history.val_CRPS)):
            raise ValueError("epoch and all train_*/val_CRPS series must be the same length")

        figures: List[plt.Figure] = []
        marker = "o" if n == 1 else None

        fig_total, ax = plt.subplots(figsize=(8, 4))
        ax.plot(history.epoch, history.train_total, marker=marker, label="train_total (ELBO)")
        if n == 1:
            ax.set_xlim(history.epoch[0] - 0.5, history.epoch[0] + 0.5)
        ax.set_title(f"{self.cfg.dataset_name}: Train Total Loss (ELBO)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)
        ax.legend()
        figures.append(fig_total)

        val_series: Dict[str, List[float]] = {"CRPS": history.val_CRPS}
        for name in ("MSE", "NRMSE", "MASE", "ND"):
            series = getattr(history, f"val_{name}", None)
            if series is not None:
                val_series[name] = series

        n_panels = 4  # train rec, train nll, train kl, val metrics
        fig_comp, axes = plt.subplots(n_panels, 1, figsize=(8, 3 * n_panels), sharex=True)

        for ax_i, name in zip(axes[:3], ("rec", "nll", "kl")):
            ax_i.plot(history.epoch, getattr(history, f"train_{name}"), marker=marker, label=f"train_{name}")
            if n == 1:
                ax_i.set_xlim(history.epoch[0] - 0.5, history.epoch[0] + 0.5)
            ax_i.set_ylabel(name)
            ax_i.grid(alpha=0.3)
            ax_i.legend()

        ax_val = axes[3]
        for name, series in val_series.items():
            ax_val.plot(history.epoch, series, marker=marker, label=f"val_{name}")
        if n == 1:
            ax_val.set_xlim(history.epoch[0] - 0.5, history.epoch[0] + 0.5)
        ax_val.set_ylabel("val metrics")
        ax_val.set_xlabel("Epoch")
        ax_val.grid(alpha=0.3)
        ax_val.legend()

        fig_comp.suptitle(f"{self.cfg.dataset_name}: Train ELBO Components vs Val Sampling Metrics")
        figures.append(fig_comp)

        return figures

    def plot_forecast_vs_truth(
        self,
        batch: ForecastBatch,
        feature_idx: Union[int, List[int]] = 0,
        sample_idx: Optional[int] = None,
        interval_z: Optional[float] = None,
    ) -> plt.Figure:
        """Plot context + horizon truth/prediction with optional uncertainty bands.

        feature_idx can be int or list[int]. A list generates a subplot grid.
        """
        if sample_idx is None:
            sample_idx = self.cfg.default_sample_idx
        z = self.cfg.interval_z if interval_z is None else interval_z

        features: List[int]
        if isinstance(feature_idx, int):
            features = [feature_idx]
        else:
            features = list(feature_idx)

        num_features = batch.y_true.shape[2]
        for idx in features:
            if idx < 0 or idx >= num_features:
                raise IndexError(f"feature_idx={idx} out of range for V={num_features}")

        if sample_idx < 0 or sample_idx >= batch.y_true.shape[0]:
            raise IndexError(f"sample_idx={sample_idx} out of range for B={batch.y_true.shape[0]}")

        n = len(features)
        ncols = min(2, n)
        nrows = int(math.ceil(n / ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 3.5 * nrows))
        axes = np.array(axes).reshape(-1)

        context_len = 0
        if batch.context_true is not None:
            context_len = batch.context_true.shape[1]

        for i, fidx in enumerate(features):
            ax = axes[i]

            # Context section
            if batch.context_true is not None:
                ctx_true = batch.context_true[sample_idx, :, fidx]
                x_ctx = np.arange(context_len)
                ax.plot(x_ctx, ctx_true, label="context_true", color="tab:blue")

            if batch.context_mu is not None:
                ctx_mu = batch.context_mu[sample_idx, :, fidx]
                x_ctx = np.arange(context_len)
                ax.plot(x_ctx, ctx_mu, label="context_mu", color="tab:cyan", linestyle="--")

            # Horizon section
            y_true = batch.y_true[sample_idx, :, fidx]
            y_mu = batch.y_mu[sample_idx, :, fidx]
            x_h = np.arange(context_len, context_len + y_true.shape[0])

            ax.plot(x_h, y_true, label="horizon_true", color="tab:green")
            ax.plot(x_h, y_mu, label="horizon_mu", color="tab:red")

            # Optional uncertainty band
            if batch.y_sigma is not None:
                y_sigma = batch.y_sigma[sample_idx, :, fidx]
                if np.all(np.isfinite(y_sigma)) and np.all(y_sigma >= 0):
                    lower = y_mu - z * y_sigma
                    upper = y_mu + z * y_sigma
                    ax.fill_between(x_h, lower, upper, alpha=0.2, color="tab:red", label=f"mu +/- {z:.2f}sigma")

            if context_len > 0:
                ax.axvline(context_len - 0.5, color="gray", linestyle="--", alpha=0.7)

            if self.cfg.feature_names and fidx < len(self.cfg.feature_names):
                title = self.cfg.feature_names[fidx]
            else:
                title = f"feature_{fidx}"

            if self.cfg.units and fidx < len(self.cfg.units):
                ylabel = f"Value ({self.cfg.units[fidx]})"
            else:
                ylabel = "Value"

            ax.set_title(title)
            ax.set_xlabel("Time step")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)

        # Hide unused subplots in the grid
        for j in range(n, len(axes)):
            axes[j].axis("off")

        fig.suptitle(f"{self.cfg.dataset_name}: Forecast vs Truth")
        return fig

    def plot_residual_by_horizon(self, residuals: HorizonResiduals) -> plt.Figure:
        """Plot MAE and MSE as a function of horizon step."""
        h = np.arange(1, len(residuals.mae_by_horizon) + 1)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(h, residuals.mae_by_horizon, label="MAE")
        ax.plot(h, residuals.mse_by_horizon, label="MSE")
        ax.set_title(f"{self.cfg.dataset_name}: Residuals by Horizon")
        ax.set_xlabel("Forecast step")
        ax.set_ylabel("Error")
        ax.grid(alpha=0.3)
        ax.legend()
        return fig

    def plot_comparison_table(self, rows: List[Dict], title: str) -> plt.Figure:
        """Generic comparison chart for baseline/ablation rows.

        Expected row keys: model, metric, value.
        """
        required = {"model", "metric", "value"}
        for row in rows:
            missing = required.difference(row.keys())
            if missing:
                raise ValueError(f"Missing keys {missing} in row {row}")

        labels = [f"{row['model']}:{row['metric']}" for row in rows]
        values = [row["value"] for row in rows]

        fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.8), 4))
        ax.bar(labels, values)
        ax.set_title(title)
        ax.set_ylabel("Value")
        ax.tick_params(axis="x", labelrotation=45)
        ax.grid(axis="y", alpha=0.3)
        return fig
