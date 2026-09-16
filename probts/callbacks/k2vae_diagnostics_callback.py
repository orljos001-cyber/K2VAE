import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks.callback import Callback

from probts.data.data_wrapper import ProbTSBatchData
from probts.utils.diagnostics import (
    DatasetPlotConfig,
    PlotDataCollector,
    PlotManager,
    ReferenceEpochHistory,
)

log = logging.getLogger(__name__)


class K2VAEDiagnosticsCallback(Callback):
    """
    Ports the thesis K2VAE repo's 4-figure diagnostic output (loss curves,
    ELBO components, forecast-vs-truth, residuals-by-horizon) onto this
    repo's K2VAE, so the two implementations can be compared side by side.

    No-ops for every other forecaster in this repo (checked in
    on_fit_start), so it is safe to leave permanently wired into run.py's
    callback list rather than only enabling it for K2VAE runs.

    This repo trains K2VAE on the ELBO but validates/checkpoints on
    CRPS-family sampling metrics (see forecast_module.py's evaluate() and
    run.py's ModelCheckpoint monitor), never computing an ELBO on the val
    set. So unlike the thesis repo's EpochHistory, the val side of the loss
    curves here is CRPS/MSE/NRMSE/MASE/ND, not val_elbo/val_nll/val_kl/val_rec
    -- see plotting.ReferenceEpochHistory / plot_reference_loss_curves.
    """

    VAL_METRIC_NAMES = ("CRPS", "MSE", "NRMSE", "MASE", "ND")

    def __init__(
        self,
        output_dir: str = "diagnostics",
        run_id: Optional[str] = None,
        plot_config: Optional[DatasetPlotConfig] = None,
        max_features_plotted: int = 6,
    ):
        self.output_dir = output_dir
        self.run_id = run_id
        self.plot_config = plot_config
        self.max_features_plotted = max_features_plotted

        self._enabled = False
        self._train_history: Dict[int, Dict[str, float]] = {}
        self._val_history: Dict[int, Dict[str, float]] = {}
        self._epoch_sums: Dict[str, float] = {}
        self._epoch_batches = 0

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        forecaster = pl_module.forecaster
        self._enabled = forecaster.name == "k2VAEModel"
        if not self._enabled:
            log.info(
                "K2VAEDiagnosticsCallback: forecaster is %s, not k2VAEModel; skipping diagnostics.",
                forecaster.name,
            )
            return

        if isinstance(forecaster.context_length, list) or isinstance(forecaster.prediction_length, list):
            log.warning(
                "K2VAEDiagnosticsCallback: multi-horizon context/prediction_length is not "
                "supported by this callback; skipping diagnostics."
            )
            self._enabled = False
            return

        if self.run_id is None:
            self.run_id = getattr(trainer, "tag", None) or forecaster.dataset or "k2vae_reference"

        if self.plot_config is None:
            self.plot_config = DatasetPlotConfig(
                dataset_name=forecaster.dataset or self.run_id,
                num_features=forecaster.target_dim,
                patch_size=getattr(forecaster.model, "patch_len", 1),
                input_window=forecaster.context_length,
                horizon_window=forecaster.prediction_length,
            )

    # ---- train side: average the k2VAEModel.loss() side-channel per epoch ----
    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self._enabled:
            return
        self._epoch_sums = {"total": 0.0, "rec": 0.0, "nll": 0.0, "kl": 0.0}
        self._epoch_batches = 0

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        if not self._enabled:
            return
        components = getattr(pl_module.forecaster, "last_loss_components", None)
        if components is None:
            return
        for key in self._epoch_sums:
            value = components.get(key)
            if value is not None:
                self._epoch_sums[key] += float(value)
        self._epoch_batches += 1

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self._enabled or self._epoch_batches == 0:
            return
        n = self._epoch_batches
        self._train_history[trainer.current_epoch] = {
            key: total / n for key, total in self._epoch_sums.items()
        }

    # ---- val side: read the CRPS-family metrics ProbTSForecastModule already logged ----
    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self._enabled or trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        epoch_metrics = {}
        for name in self.VAL_METRIC_NAMES:
            value = metrics.get(f"val_{name}")
            if value is not None:
                epoch_metrics[name] = float(value)
        if epoch_metrics:
            self._val_history[trainer.current_epoch] = epoch_metrics

    # ---- final plots ----
    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self._enabled:
            return
        manager = PlotManager(cfg=self.plot_config, run_id=self.run_id, output_root=self.output_dir)

        if self._train_history:
            loss_figs = manager.plot_reference_loss_curves(self._build_history())
            for i, fig in enumerate(loss_figs):
                manager.save(fig, f"loss_curves_{i}")
        else:
            log.warning("K2VAEDiagnosticsCallback: no train-epoch history recorded, skipping loss curves.")

        self._save_forecast_and_residual_plots(trainer, pl_module, manager)

    def _build_history(self) -> ReferenceEpochHistory:
        epochs = sorted(self._train_history.keys())
        nan = float("nan")

        def train_series(name: str):
            return [self._train_history[e][name] for e in epochs]

        def val_series(name: str):
            return [self._val_history.get(e, {}).get(name, nan) for e in epochs]

        return ReferenceEpochHistory(
            epoch=epochs,
            train_total=train_series("total"),
            train_rec=train_series("rec"),
            train_nll=train_series("nll"),
            train_kl=train_series("kl"),
            val_CRPS=val_series("CRPS"),
            val_MSE=val_series("MSE"),
            val_NRMSE=val_series("NRMSE"),
            val_MASE=val_series("MASE"),
            val_ND=val_series("ND"),
        )

    @torch.no_grad()
    def _save_forecast_and_residual_plots(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", manager: PlotManager
    ) -> None:
        dataloader = trainer.datamodule.val_dataloader()
        try:
            batch = next(iter(dataloader))
        except StopIteration:
            log.warning(
                "K2VAEDiagnosticsCallback: validation dataloader is empty, skipping "
                "forecast/residual plots."
            )
            return

        forecaster = pl_module.forecaster
        scaler = pl_module.scaler

        batch_data = ProbTSBatchData(batch, pl_module.device)
        context = batch_data.past_target_cdf[:, -forecaster.context_length:, :]
        horizon_true = batch_data.future_target_cdf

        forecaster.model.eval()
        norm_context = scaler.transform(context)
        rec, _prior_dist, post_dist = forecaster.forward(norm_context)

        context_mu = scaler.inverse_transform(rec)
        y_mu = scaler.inverse_transform(post_dist.loc)

        # sigma is spread, not location: scale by the scaler's multiplicative
        # factor only, mirroring RevIN.denormalize_sigma in the thesis repo
        # rather than reusing inverse_transform (which would also add the
        # scaler's additive mean/shift, wrong for a standard deviation).
        # Every shipped k2vae.yaml uses `scaler: standard` (StandardScaler,
        # per-feature scale shape (C,), broadcasts against (B,H,C) y_sigma);
        # this has not been checked against the temporal/identity options.
        scale = getattr(scaler, "scale", None)
        if scale is None:
            y_sigma = post_dist.scale
        else:
            epsilon = getattr(scaler, "epsilon", 0.0)
            y_sigma = post_dist.scale * (scale.to(post_dist.scale.device) + epsilon)

        collector = PlotDataCollector()
        forecast_batch = collector.collect_forecast_batch(
            x_h_true=horizon_true,
            x_h_mu=y_mu,
            x_h_sigma=y_sigma,
            x_c_true=context,
            x_c_mu=context_mu,
            num_features=forecaster.target_dim,
            patch_size=self.plot_config.patch_size,
            inverse_scale=False,  # already inverse-transformed above
            patched=False,
        )

        features_to_plot = list(range(min(self.max_features_plotted, forecaster.target_dim)))
        forecast_fig = manager.plot_forecast_vs_truth(forecast_batch, feature_idx=features_to_plot)
        manager.save(forecast_fig, "forecast_vs_truth")

        residuals = PlotDataCollector.compute_horizon_residuals(forecast_batch)
        residual_fig = manager.plot_residual_by_horizon(residuals)
        manager.save(residual_fig, "residual_by_horizon")
