"""Callbacks implementing the toolbox resume and terminal-reporting contracts."""

from __future__ import annotations

import random
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from hsmsfusion.metrics.evaluator import EvaluationResult
from hsmsfusion.visualization import save_srf_curve, save_validation_comparison

_TERMINAL_CONSOLE = Console(
    color_system="auto",
    force_terminal=None,
    legacy_windows=False,
    markup=False,
    emoji=False,
    highlight=False,
)


def _supports_unicode_progress() -> bool:
    """Use Rich's native bar only when the active terminal can encode it."""

    encoding = (sys.stdout.encoding or "").lower().replace("-", "")
    return "utf" in encoding or encoding in {"cp65001", "65001"}


def _console() -> Console:
    """Create a color-capable Rich console with a safe legacy fallback."""

    return _TERMINAL_CONSOLE


class _AsciiBarColumn(ProgressColumn):
    """A styled ASCII fallback for legacy Windows code pages."""

    def __init__(self, width: int = 24) -> None:
        super().__init__()
        self.width = width

    def render(self, task: Any) -> Text:
        total = float(task.total or 0.0)
        fraction = 0.0 if total <= 0.0 else min(max(task.completed / total, 0.0), 1.0)
        filled = int(round(self.width * fraction))
        bar = Text("[", style="dim")
        bar.append("#" * filled, style="bright_cyan")
        bar.append("-" * (self.width - filled), style="grey50")
        bar.append("]", style="dim")
        return bar


def _progress_bar_column() -> ProgressColumn:
    if _supports_unicode_progress():
        return BarColumn(
            bar_width=32,
            style="grey35",
            complete_style="bright_cyan",
            finished_style="bright_green",
            pulse_style="cyan",
        )
    return _AsciiBarColumn(width=32)


def _extract_loss(outputs: Any) -> float | None:
    """Read the scalar batch loss across Lightning's supported output shapes."""

    candidate = outputs.get("loss") if isinstance(outputs, Mapping) else outputs
    if not isinstance(candidate, torch.Tensor) or candidate.numel() != 1:
        return None
    value = float(candidate.detach().cpu())
    return value if np.isfinite(value) else None


class EpochRichProgressCallback(Callback):
    """Show one Rich progress row for the current training epoch."""

    def __init__(self, refresh_rate: int = 1) -> None:
        super().__init__()
        self.refresh_rate = max(1, int(refresh_rate))
        self.console = _console()
        self.progress: Progress | None = None
        self.task_id: Any | None = None
        self.total_batches = 0

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        self.progress = Progress(
            TextColumn("{task.description}", style="bold bright_cyan"),
            TextColumn(
                "Epoch {task.fields[epoch]:>3}/{task.fields[max_epochs]:<3}",
                style="bold white",
            ),
            _progress_bar_column(),
            TaskProgressColumn(show_speed=False),
            TextColumn("{task.completed:>4.0f}/{task.total:<4.0f} batches", style="dim"),
            TextColumn("loss {task.fields[loss]}", style="yellow"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            refresh_per_second=8,
            transient=True,
        )
        self.progress.start()

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        if self.progress is None:
            return
        if self.task_id is not None:
            self.progress.remove_task(self.task_id)
            self.task_id = None
        total_batches = int(trainer.num_training_batches)
        self.total_batches = max(total_batches, 1)
        self.task_id = self.progress.add_task(
            "TRAIN",
            total=self.total_batches,
            epoch=trainer.current_epoch + 1,
            max_epochs=trainer.max_epochs,
            loss="--",
        )

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.progress is not None and self.task_id is not None:
            if (batch_idx + 1) % self.refresh_rate == 0:
                update: dict[str, Any] = {"advance": self.refresh_rate}
                loss = _extract_loss(outputs)
                if loss is not None:
                    update["loss"] = f"{loss:.6f}"
                self.progress.update(self.task_id, **update)

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        if self.progress is not None and self.task_id is not None:
            self.progress.update(self.task_id, completed=self.total_batches)
            self.progress.refresh()
            self.progress.remove_task(self.task_id)
            self.task_id = None

    def on_fit_end(self, trainer: Any, pl_module: Any) -> None:
        self._stop()

    def on_exception(self, trainer: Any, pl_module: Any, exception: BaseException) -> None:
        self._stop()

    def _stop(self) -> None:
        if self.progress is not None:
            self.progress.stop()
            self.progress = None
            self.task_id = None
            self.total_batches = 0


class TrainingVisualizationCallback(Callback):
    """Persist the fixed SRF and one qualitative validation sample per epoch."""

    def __init__(self, run_directory: str | Path) -> None:
        super().__init__()
        self.figures_directory = Path(run_directory) / "logs" / "figures"
        self._srf_written = False

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        if self._srf_written:
            return
        observation_generator = getattr(trainer.datamodule, "observation_generator", None)
        spectral_operator = getattr(observation_generator, "spectral_operator", None)
        self._srf_written = save_srf_curve(
            spectral_operator,
            self.figures_directory / "srf_curve.png",
        )

    def on_validation_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if trainer.sanity_checking or batch_idx != 0 or not isinstance(outputs, Mapping):
            return
        prediction = outputs.get("prediction")
        if not isinstance(prediction, torch.Tensor):
            return
        observation_generator = getattr(trainer.datamodule, "observation_generator", None)
        spectral_operator = getattr(observation_generator, "spectral_operator", None)
        save_validation_comparison(
            batch,
            prediction,
            self.figures_directory
            / "validation"
            / f"epoch_{trainer.current_epoch + 1:04d}.png",
            spectral_operator=spectral_operator,
        )


class EpochBoundaryLastCheckpointCallback(Callback):
    """Persist ``last.ckpt`` after every training epoch, independent of validation."""

    def __init__(self, checkpoint_path: str | Path) -> None:
        super().__init__()
        self.checkpoint_path = Path(checkpoint_path)

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        if trainer.sanity_checking:
            return
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(self.checkpoint_path), weights_only=False)


class PredictTimerCallback(Callback):
    """Record per-batch wall-clock time and batch sizes during predict."""

    def __init__(self) -> None:
        super().__init__()
        self.batch_times: list[float] = []
        self.batch_sizes: list[int] = []
        self._t_start: float | None = None

    def on_predict_batch_start(
        self,
        trainer: Any,
        pl_module: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._t_start = time.perf_counter()

    def on_predict_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self._t_start is not None:
            elapsed = time.perf_counter() - self._t_start
            self.batch_times.append(elapsed)
            self.batch_sizes.append(int(outputs.shape[0]))


class EpochMetricReporter(Callback):
    """Print training and validation summaries outside the live Rich dashboard."""

    validation_metric_labels = (
        ("val/psnr_global", "PSNR", "bright_green"),
        ("val/ssim_band_mean", "SSIM", "bright_magenta"),
        ("val/sam_deg", "SAM (deg)", "bright_yellow"),
        ("val/ergas", "ERGAS", "bright_red"),
        ("val/rmse_global", "RMSE", "bright_cyan"),
        ("val/uiqi", "UIQI", "bright_white"),
    )

    def __init__(self) -> None:
        super().__init__()
        self.console = _console()
        self._epoch_started_at: float | None = None
        self._epoch_train_seconds: float | None = None
        self._total_train_seconds = 0.0
        self._validation_completed_this_epoch = False

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        self._epoch_started_at = None
        self._epoch_train_seconds = None
        self._total_train_seconds = 0.0
        self._validation_completed_this_epoch = False

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        self._epoch_started_at = time.perf_counter()
        self._epoch_train_seconds = None
        self._validation_completed_this_epoch = False

    def on_validation_start(self, trainer: Any, pl_module: Any) -> None:
        if trainer.sanity_checking or self._epoch_started_at is None:
            return
        if self._epoch_train_seconds is None:
            self._epoch_train_seconds = time.perf_counter() - self._epoch_started_at

    def on_validation_end(self, trainer: Any, pl_module: Any) -> None:
        if not trainer.sanity_checking:
            self._validation_completed_this_epoch = True

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        if trainer.sanity_checking:
            return
        train_seconds = self._epoch_train_seconds
        if train_seconds is None and self._epoch_started_at is not None:
            train_seconds = time.perf_counter() - self._epoch_started_at
        train_seconds = max(0.0, train_seconds or 0.0)
        self._total_train_seconds += train_seconds

        train_loss = trainer.callback_metrics.get("train/loss")
        train_loss_text = "--" if train_loss is None else f"{_as_scalar(train_loss):.8f}"
        training_loss_terms = self._training_loss_terms(trainer, pl_module)
        validation_metrics: dict[str, tuple[str, str]] = {}
        val_loss_text = "--"
        if self._validation_completed_this_epoch:
            val_loss = trainer.callback_metrics.get("val/loss")
            if val_loss is not None:
                val_loss_text = f"{_as_scalar(val_loss):.8f}"
            for metric_name, label, color in self.validation_metric_labels:
                value = trainer.callback_metrics.get(metric_name)
                if value is None:
                    continue
                validation_metrics[label] = (f"{_as_scalar(value):.8f}", color)
        self._print_epoch_summary(
            epoch=trainer.current_epoch + 1,
            max_epochs=trainer.max_epochs,
            train_loss=train_loss_text,
            val_loss=val_loss_text,
            train_seconds=train_seconds,
            training_loss_terms=training_loss_terms,
            validation_metrics=validation_metrics,
        )

        self._epoch_started_at = None
        self._epoch_train_seconds = None
        self._validation_completed_this_epoch = False

    def on_fit_end(self, trainer: Any, pl_module: Any) -> None:
        summary = Table.grid(padding=(0, 2))
        summary.add_column(style="bold bright_cyan")
        summary.add_column(style="bold bright_green", justify="right")
        summary.add_row("Total training time", _format_duration(self._total_train_seconds))
        self.console.print(
            Panel(
                summary,
                title=Text("Training complete", style="bold bright_green"),
                border_style="bright_green",
                box=box.ROUNDED if _supports_unicode_progress() else box.ASCII,
                expand=False,
            )
        )

    def _print_epoch_summary(
        self,
        *,
        epoch: int,
        max_epochs: int,
        train_loss: str,
        val_loss: str,
        train_seconds: float,
        training_loss_terms: Sequence[tuple[str, str, str]],
        validation_metrics: Mapping[str, tuple[str, str]],
    ) -> None:
        summary = Table.grid(padding=(0, 2))
        summary.add_column(style="bold bright_cyan", no_wrap=True)
        summary.add_column(justify="right", no_wrap=True)
        summary.add_column(style="bold bright_cyan", no_wrap=True)
        summary.add_column(justify="right", no_wrap=True)
        summary.add_row(
            "Average train loss",
            Text(train_loss, style="bold bright_yellow"),
            "Training time",
            Text(_format_duration(train_seconds), style="bold bright_green"),
        )
        summary.add_row(Text("Training loss terms", style="bold white"), "", "", "")
        if training_loss_terms:
            for alias, value, weighted_value in training_loss_terms:
                summary.add_row(
                    Text(alias, style="bold bright_cyan"),
                    Text(value, style="bright_white"),
                    Text("Weighted contribution", style="bold bright_yellow"),
                    Text(weighted_value, style="bright_yellow"),
                )
        else:
            summary.add_row(Text("--", style="dim"), "", "", "")
        summary.add_row(Text("Validation metrics", style="bold white"), "", "", "")
        summary.add_row(
            Text("Validation loss", style="bold bright_red"),
            Text(val_loss, style="bold bright_red"),
            "",
            "",
        )
        self._add_metric_row(summary, validation_metrics, "PSNR", "SSIM")
        self._add_metric_row(summary, validation_metrics, "SAM (deg)", "ERGAS")
        self._add_metric_row(summary, validation_metrics, "RMSE", "UIQI")
        self.console.print(
            Panel(
                summary,
                title=Text(f"Epoch {epoch:03d}/{max_epochs:03d}", style="bold bright_blue"),
                border_style="bright_blue",
                box=box.ROUNDED if _supports_unicode_progress() else box.ASCII,
                expand=False,
            )
        )

    @staticmethod
    def _training_loss_terms(
        trainer: Any,
        pl_module: Any,
    ) -> list[tuple[str, str, str]]:
        """Read each configured loss term without duplicating ``train/loss``."""

        metrics = trainer.callback_metrics
        aliases = [
            str(component.alias)
            for component in getattr(getattr(pl_module, "loss_manager", None), "components", ())
        ]
        if not aliases:
            aliases = sorted(
                metric_name.removeprefix("train/")
                for metric_name in metrics
                if metric_name.startswith("train/")
                and metric_name not in {"train/loss", "train/total"}
                and not metric_name.endswith("_weighted")
            )

        terms: list[tuple[str, str, str]] = []
        for alias in aliases:
            value = metrics.get(f"train/{alias}")
            if value is None:
                continue
            weighted_value = metrics.get(f"train/{alias}_weighted", value)
            terms.append(
                (
                    alias,
                    f"{_as_scalar(value):.8f}",
                    f"{_as_scalar(weighted_value):.8f}",
                )
            )
        return terms

    @staticmethod
    def _add_metric_row(
        summary: Table,
        metrics: Mapping[str, tuple[str, str]],
        left_label: str,
        right_label: str | None,
    ) -> None:
        left_value, left_color = metrics.get(left_label, ("--", "dim"))
        if right_label is None:
            summary.add_row(
                Text(left_label, style=f"bold {left_color}"),
                Text(left_value, style=f"bold {left_color}"),
                "",
                "",
            )
            return
        right_value, right_color = metrics.get(right_label, ("--", "dim"))
        summary.add_row(
            Text(left_label, style=f"bold {left_color}"),
            Text(left_value, style=f"bold {left_color}"),
            Text(right_label, style=f"bold {right_color}"),
            Text(right_value, style=f"bold {right_color}"),
        )


def _as_scalar(value: Any) -> float:
    return float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)


def _format_duration(seconds: float) -> str:
    hours, remainder = divmod(max(0.0, seconds), 3600.0)
    minutes, seconds = divmod(remainder, 60.0)
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"


def print_training_overview(
    *,
    model_name: str,
    model_class_name: str,
    model_parameters: dict[str, Any],
    total_parameters_million: float,
    trainable_parameters_million: float,
    dataset_name: str,
    dataset_root: str,
    train_dataloader_config: dict[str, Any],
    valid_dataloader_config: dict[str, Any],
    materialization_batch_size: int,
    validation_every_n_epochs: int,
    spectral_config: dict[str, Any],
    spatial_config: dict[str, Any],
    operator_access: str,
    run_directory: str,
) -> None:
    """Render the exact model, dataset, and fixed degradation setup before fit."""

    overview = Table.grid(padding=(0, 2))
    overview.add_column(style="bold bright_cyan", no_wrap=True)
    overview.add_column()
    overview.add_row("Model", Text(f"{model_name} ({model_class_name})", style="bold white"))
    overview.add_row(
        "Parameters",
        Text(
            f"{total_parameters_million:.3f} M total | "
            f"{trainable_parameters_million:.3f} M trainable",
            style="bold bright_green",
        ),
    )
    overview.add_row("Model config", _format_config_items(model_parameters))
    overview.add_row("Dataset", Text(dataset_name, style="bold white"))
    overview.add_row("Dataset root", Text(dataset_root, style="dim"))
    overview.add_row("Materialization", f"batch_size={materialization_batch_size}")
    overview.add_row("Train loader", _format_dataloader(train_dataloader_config))
    overview.add_row("Valid loader", _format_dataloader(valid_dataloader_config))
    overview.add_row("Validation cadence", f"every {validation_every_n_epochs} epoch(s)")
    overview.add_row(
        "Spectral operator",
        _format_spectral_operator(spectral_config),
    )
    response_file = spectral_config.get("response_file")
    if response_file:
        overview.add_row("Measured SRF", Text(str(response_file), style="bold bright_magenta"))
    else:
        overview.add_row(
            "SRF centers",
            f"[{_format_sequence(spectral_config.get('centers_nm', []))}] nm",
        )
        overview.add_row(
            "SRF FWHM",
            f"[{_format_sequence(spectral_config.get('fwhm_nm', []))}] nm",
        )
    overview.add_row("Spatial operator", _format_spatial_operator(spatial_config))
    operator_style = "bright_green" if operator_access == "known" else "bright_yellow"
    overview.add_row("Operator access", Text(operator_access, style=f"bold {operator_style}"))
    overview.add_row("Run directory", Text(run_directory, style="dim"))
    _console().print(
        Panel(
            overview,
            title=Text("Training setup", style="bold bright_blue"),
            border_style="bright_blue",
            box=box.ROUNDED if _supports_unicode_progress() else box.ASCII,
            expand=False,
        )
    )


def print_test_report(
    *,
    result: EvaluationResult,
    model_name: str,
    benchmark_name: str,
    total_wall_seconds: float,
    trimmed_per_sample_ms: float | None,
    sample_count: int,
) -> None:
    """Render per-image metrics, summary statistics, and timing in the terminal."""

    _METRIC_LABELS: dict[str, str] = {
        "psnr_global": "PSNR",
        "ssim_band_mean": "SSIM",
        "sam_deg": "SAM",
        "ergas": "ERGAS",
        "rmse_global": "RMSE",
        "uiqi": "UIQI",
    }

    console = _console()
    use_unicode = _supports_unicode_progress()
    active_box = box.ROUNDED if use_unicode else box.ASCII

    metric_names: list[str] = list(result.summary["metrics"])
    labels: list[str] = [_METRIC_LABELS.get(n, n) for n in metric_names]

    # -- Metrics table (per-image + mean + std) -----------------------------
    tbl = Table(box=active_box, show_header=True, header_style="bold bright_cyan", padding=(0, 1), expand=True)
    tbl.add_column("Image", style="bold bright_cyan", no_wrap=True)
    for label in labels:
        tbl.add_column(label, justify="right")

    for idx, row in enumerate(result.per_image):
        cells: list[str | Text] = [f"image_{idx:04d}"]
        for name in metric_names:
            cells.append(f"{float(row[name]):.4f}")
        tbl.add_row(*cells, style="bright_white")

    # Mean row
    mean_cells: list[str | Text] = [Text("Mean", style="bold bright_green")]
    for name in metric_names:
        val = float(result.summary["metrics"][name]["mean"])
        mean_cells.append(Text(f"{val:.4f}", style="bold bright_green"))
    tbl.add_row(*mean_cells)
    tbl.add_section()

    # Std row
    std_cells: list[str | Text] = [Text("Std", style="bright_yellow")]
    for name in metric_names:
        val = float(result.summary["metrics"][name]["std"])
        std_cells.append(Text(f"{val:.4f}", style="bright_yellow"))
    tbl.add_row(*std_cells)

    # -- Timing line (below table, inside same panel) ------------------------
    timing_parts: list[str] = [
        f"Wall time: {_format_duration(total_wall_seconds)}",
        f"Samples: {sample_count}",
    ]
    if trimmed_per_sample_ms is not None:
        timing_parts.append(f"Trimmed avg: {trimmed_per_sample_ms:.2f} ms/sample")
    timing_line = Text("    ".join(timing_parts), style="bold bright_white")

    # -- Wrap everything in one Panel ---------------------------------------
    console.print(
        Panel(
            Group(tbl, timing_line),
            title=Text(f"Test results — {model_name}  |  {benchmark_name}", style="bold bright_blue"),
            border_style="bright_blue",
            box=active_box,
            expand=False,
        )
    )


def _format_config_items(config: Mapping[str, Any]) -> str:
    return ", ".join(
        f"{key}={_format_value(value)}" for key, value in config.items()
    ) or "--"


def _format_dataloader(config: Mapping[str, Any]) -> str:
    ordering = "shuffle" if config.get("shuffle") else "ordered"
    memory = "pinned" if config.get("pin_memory") else "pageable"
    return (
        f"batch={config.get('batch_size')} | {ordering} | {memory} | "
        f"workers={config.get('num_workers')}"
    )


def _format_spectral_operator(config: Mapping[str, Any]) -> str:
    name = str(config.get("name", "unknown")).replace("_", " ")
    sensor = str(config.get("sensor", "unspecified")).upper()
    hsi_bands = config.get("num_hsi_bands", "?")
    msi_bands = config.get("num_msi_bands", len(config.get("centers_nm", [])))
    start = _format_value(config.get("wavelength_start_nm", "?"))
    end = _format_value(config.get("wavelength_end_nm", "?"))
    return f"{name} | sensor={sensor} | {hsi_bands} HSI -> {msi_bands} MSI bands | {start}-{end} nm"


def _format_spatial_operator(config: Mapping[str, Any]) -> str:
    name = str(config.get("name", "unknown")).replace("_", " ")
    return (
        f"{name} | scale=x{config.get('scale', '?')} | "
        f"kernel={config.get('kernel_size', '?')} | sigma={_format_value(config.get('sigma', '?'))} | "
        f"padding={config.get('padding', '?')} | offset={_format_value(config.get('offset', '?'))}"
    )


def _format_sequence(values: Any) -> str:
    if not _is_sequence(values):
        return _format_value(values)
    return ", ".join(_format_value(value) for value in values)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else f"{value:g}"
    if _is_sequence(value):
        return "[" + _format_sequence(value) + "]"
    return str(value)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


class RNGStateCallback(Callback):
    """Persist RNG states that are not guaranteed by the default checkpoint schema."""

    checkpoint_key = "hsmsfusion_rng_state"

    def on_save_checkpoint(
        self,
        trainer: Any,
        pl_module: Any,
        checkpoint: dict[str, Any],
    ) -> None:
        numpy_state = np.random.get_state()
        state: dict[str, Any] = {
            "python": random.getstate(),
            "numpy": {
                "bit_generator": numpy_state[0],
                "keys": numpy_state[1].tolist(),
                "position": int(numpy_state[2]),
                "has_gauss": int(numpy_state[3]),
                "cached_gaussian": float(numpy_state[4]),
            },
            "torch_cpu": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        checkpoint[self.checkpoint_key] = state

    def on_load_checkpoint(
        self,
        trainer: Any,
        pl_module: Any,
        checkpoint: dict[str, Any],
    ) -> None:
        state = checkpoint.get(self.checkpoint_key)
        if not state:
            return
        random.setstate(state["python"])
        numpy_state = state["numpy"]
        np.random.set_state(
            (
                numpy_state["bit_generator"],
                np.asarray(numpy_state["keys"], dtype=np.uint32),
                numpy_state["position"],
                numpy_state["has_gauss"],
                numpy_state["cached_gaussian"],
            )
        )
        torch.set_rng_state(state["torch_cpu"])
        if torch.cuda.is_available() and "torch_cuda" in state:
            torch.cuda.set_rng_state_all(state["torch_cuda"])
