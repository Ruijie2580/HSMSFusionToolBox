"""Summarize multi-metric test results and draw per-metric box plots.

The toolbox test export uses the following layout::

    output/<dataset>_x<scale>/<method>/metrics.csv       (preferred)
    output/<dataset>_x<scale>/<method>/predictions.h5
    Dataset/<dataset>/test.h5                            (/HRHS, NCHW)

``metrics.csv`` columns may carry a leading ``*`` (e.g. ``*SAM``); headers are
mapped to stable internal names (``psnr``, ``ssim``, ``sam``, ``ergas``,
``rmse``, ``uiqi``).  Only ``image_*`` rows participate in distribution
statistics; the trailing ``Average`` / ``Std`` rows are ignored.

Run the script without arguments to launch the GUI::

    python scripts/plot_metric_summary.py
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

# Allow ``python scripts/plot_metric_summary.py`` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from scripts._shared import (
    infer_target_path,
    registered_model_display_names,
    resolve_benchmark,
    select_predictions,
    setup_matplotlib_font,
)

setup_matplotlib_font()

_FORMAT_SUFFIXES = {"png": "png", "jpg": "jpeg", "jpeg": "jpeg", "pdf": "pdf", "svg": "svg"}
_HIGHLIGHT_COLOR = "#e74c3c"
_DEFAULT_BOX_COLOR = "#4c78a8"
_PANEL_WIDTH_INCHES = 4.6
_PANEL_HEIGHT_INCHES = 3.4
_MAX_PANEL_COLUMNS = 3

# CSV / display header → internal metric key.
_HEADER_TO_METRIC: dict[str, str] = {
    "psnr": "psnr",
    "psnr_global": "psnr",
    "ssim": "ssim",
    "ssim_band_mean": "ssim",
    "sam": "sam",
    "sam_deg": "sam",
    "sam_rad": "sam_rad",
    "ergas": "ergas",
    "rmse": "rmse",
    "rmse_global": "rmse",
    "uiqi": "uiqi",
}

_METRIC_DISPLAY: dict[str, str] = {
    "psnr": "PSNR",
    "ssim": "SSIM",
    "sam": "*SAM",
    "ergas": "*ERGAS",
    "rmse": "RMSE",
    "uiqi": "UIQI",
    "sam_rad": "SAM (rad)",
}

# Metrics where larger values are better.  Everything else is treated as
# lower-is-better and annotated in the panel title.
_HIGHER_IS_BETTER: frozenset[str] = frozenset({"psnr", "ssim", "uiqi"})

_DEFAULT_LEADERBOARD: tuple[str, ...] = (
    "psnr_global",
    "ssim_band_mean",
    "sam_deg",
    "ergas",
    "rmse_global",
    "uiqi",
)

_AGGREGATE_ROW_NAMES = frozenset({"average", "mean", "std", "std_dev", "stddev"})


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricSummaryData:
    """Aligned per-sample metric values for every selected method."""

    benchmark: str
    methods: list[str]
    metrics: list[str]
    values: dict[str, dict[str, np.ndarray]]
    sample_count: int
    warnings: list[str] = field(default_factory=list)

    def method_values(self, method: str, metric: str) -> np.ndarray:
        return self.values[method][metric]


# ---------------------------------------------------------------------------
# Header / CSV parsing
# ---------------------------------------------------------------------------


def normalize_metric_header(header: str) -> str:
    """Map a CSV header (possibly with ``*``) to a stable internal metric key."""

    cleaned = header.strip().lstrip("*").strip().casefold()
    if not cleaned:
        return header.strip()
    return _HEADER_TO_METRIC.get(cleaned, cleaned)


def metric_display_name(metric: str) -> str:
    return _METRIC_DISPLAY.get(metric, metric.upper())


def metric_higher_is_better(metric: str) -> bool:
    return metric in _HIGHER_IS_BETTER


def metric_direction_label(metric: str) -> str:
    return "higher is better" if metric_higher_is_better(metric) else "lower is better"


def _is_aggregate_row(label: str) -> bool:
    return label.strip().casefold() in _AGGREGATE_ROW_NAMES


def parse_metrics_csv(path: Path) -> dict[str, np.ndarray]:
    """Parse one ``metrics.csv`` into ``{metric: per-image float array}``.

    Only rows whose first column looks like ``image_*`` are kept.  ``Average``
    and ``Std`` (and any other aggregate labels) are ignored.
    """

    path = path.expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"metrics.csv does not exist: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f"{path} is empty.") from error
        if not header or header[0].strip().casefold() != "image":
            raise ValueError(
                f"{path} must start with an Image column header, got {header[:1]!r}."
            )

        metric_keys: list[str] = []
        seen: set[str] = set()
        for raw in header[1:]:
            key = normalize_metric_header(raw)
            if not key:
                continue
            if key in seen:
                raise ValueError(f"{path} contains duplicate metric column {raw!r}.")
            seen.add(key)
            metric_keys.append(key)

        columns: dict[str, list[float]] = {key: [] for key in metric_keys}
        for row in reader:
            if not row or not row[0].strip():
                continue
            label = row[0].strip()
            if _is_aggregate_row(label):
                continue
            if not label.casefold().startswith("image"):
                continue
            if len(row) - 1 < len(metric_keys):
                raise ValueError(
                    f"{path} row {label!r} has {len(row) - 1} values, "
                    f"expected {len(metric_keys)}."
                )
            for key, raw_value in zip(metric_keys, row[1:], strict=True):
                columns[key].append(float(raw_value))

    if not metric_keys:
        raise ValueError(f"{path} has no metric columns.")
    if not next(iter(columns.values())):
        raise ValueError(f"{path} contains no image_* data rows.")

    return {key: np.asarray(values, dtype=np.float64) for key, values in columns.items()}


# ---------------------------------------------------------------------------
# Discovery (metrics.csv and/or predictions.h5)
# ---------------------------------------------------------------------------


def discover_metric_methods(output_root: Path) -> dict[str, list[tuple[str, Path]]]:
    """Discover ``{benchmark: [(display_name, method_dir), ...]}``.

    A method directory is included when it contains ``metrics.csv`` and/or
    ``predictions.h5``.  Display names reuse the shared model registry.
    """

    output_root = output_root.expanduser()
    if not output_root.is_dir():
        return {}
    display_names = registered_model_display_names()
    discovered: dict[str, list[tuple[str, Path]]] = {}
    for benchmark_dir in sorted(p for p in output_root.iterdir() if p.is_dir()):
        methods: list[tuple[str, Path]] = []
        for method_dir in sorted(p for p in benchmark_dir.iterdir() if p.is_dir()):
            has_csv = (method_dir / "metrics.csv").is_file()
            has_predictions = (method_dir / "predictions.h5").is_file()
            if not (has_csv or has_predictions):
                continue
            display = display_names.get(method_dir.name.casefold(), method_dir.name)
            methods.append((display, method_dir))
        if methods:
            discovered[benchmark_dir.name] = methods
    return discovered


# ---------------------------------------------------------------------------
# Recompute helpers
# ---------------------------------------------------------------------------


def _scale_from_benchmark(benchmark: str) -> int:
    match = re.fullmatch(r".+_x(\d+)", benchmark, flags=re.IGNORECASE)
    return int(match.group(1)) if match else 4


def default_metric_config() -> dict[str, object]:
    """Metric configuration used when recomputing a missing ``metrics.csv``."""

    return {
        "protocol_version": "hif_metrics_v1",
        "leaderboard": list(_DEFAULT_LEADERBOARD),
        "epsilon": 1e-8,
        "sam_epsilon": 1e-8,
        "ssim": {
            "window_size": 11,
            "sigma": 1.5,
            "padding": "reflect",
            "k1": 0.01,
            "k2": 0.03,
        },
    }


def recompute_metrics_csv(
    *,
    method_dir: Path,
    target_path: Path,
    scale: int,
    metric_config: Mapping[str, object] | None = None,
) -> Path:
    """Evaluate ``predictions.h5`` against the target and write ``metrics.csv``."""

    from hsmsfusion.metrics.evaluator import evaluate_prediction_file

    method_dir = method_dir.expanduser()
    prediction_path = method_dir / "predictions.h5"
    if not prediction_path.is_file():
        raise FileNotFoundError(
            f"Cannot recompute metrics: missing predictions.h5 under {method_dir}."
        )
    if not target_path.expanduser().is_file():
        raise FileNotFoundError(f"Cannot recompute metrics: missing target {target_path}.")

    evaluate_prediction_file(
        prediction_path=prediction_path,
        target_path=target_path.expanduser(),
        metric_config=dict(metric_config or default_metric_config()),
        scale=int(scale),
        output_directory=method_dir,
    )
    return method_dir / "metrics.csv"


def _ensure_method_metrics(
    method_dir: Path,
    *,
    benchmark: str,
    target_path: Path | None,
    dataset_root: Path,
    prefer_csv: bool,
    recompute_missing: bool,
    metric_config: Mapping[str, object] | None = None,
) -> dict[str, np.ndarray]:
    csv_path = method_dir / "metrics.csv"
    if csv_path.is_file() and prefer_csv:
        return parse_metrics_csv(csv_path)

    if not recompute_missing:
        if csv_path.is_file():
            return parse_metrics_csv(csv_path)
        raise FileNotFoundError(
            f"{method_dir} has no metrics.csv and recompute is disabled."
        )

    resolved_target = (
        target_path or infer_target_path(dataset_root, benchmark)
    ).expanduser()
    recompute_metrics_csv(
        method_dir=method_dir,
        target_path=resolved_target,
        scale=_scale_from_benchmark(benchmark),
        metric_config=metric_config,
    )
    return parse_metrics_csv(csv_path)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _align_values(
    per_method: Mapping[str, Mapping[str, np.ndarray]],
    metrics: Sequence[str],
) -> tuple[dict[str, dict[str, np.ndarray]], int, list[str]]:
    warnings: list[str] = []
    counts = {name: int(next(iter(values.values())).shape[0]) for name, values in per_method.items()}
    unique_counts = sorted(set(counts.values()))
    sample_count = unique_counts[0]
    if len(unique_counts) > 1:
        details = ", ".join(f"{name}={count}" for name, count in counts.items())
        warnings.append(
            f"Sample counts differ across methods ({details}); "
            f"using the first {sample_count} image indices."
        )

    aligned: dict[str, dict[str, np.ndarray]] = {}
    for name, values in per_method.items():
        aligned[name] = {
            metric: np.asarray(values[metric][:sample_count], dtype=np.float64)
            if metric in values
            else np.full(sample_count, np.nan, dtype=np.float64)
            for metric in metrics
        }
    return aligned, sample_count, warnings


def load_metric_summary_data(
    *,
    output_root: Path,
    dataset_root: Path,
    benchmark: str | None,
    methods: Sequence[str] | None = None,
    metrics: Sequence[str] | None = None,
    prefer_csv: bool = True,
    recompute_missing: bool = True,
    target_path: Path | None = None,
    metric_config: Mapping[str, object] | None = None,
) -> MetricSummaryData:
    """Load and align per-sample metrics for the selected methods.

    When ``metrics.csv`` is missing and ``recompute_missing`` is true, the
    method is re-evaluated from ``predictions.h5`` and the CSV is written back.
    Multi-method sample counts are aligned on the shared prefix (min length).
    """

    if not prefer_csv and not recompute_missing:
        raise ValueError("At least one of prefer_csv / recompute_missing must be enabled.")

    benchmark_name, candidates = resolve_benchmark(
        discover_metric_methods(output_root), benchmark
    )
    selected = select_predictions(candidates, methods)
    if not selected:
        raise ValueError("At least one method must be selected.")

    resolved_target = (
        target_path or infer_target_path(dataset_root, benchmark_name)
    ).expanduser()

    per_method: dict[str, dict[str, np.ndarray]] = {}
    metric_orders: list[list[str]] = []
    for method_name, method_dir in selected:
        values = _ensure_method_metrics(
            method_dir,
            benchmark=benchmark_name,
            target_path=resolved_target,
            dataset_root=dataset_root,
            prefer_csv=prefer_csv,
            recompute_missing=recompute_missing,
            metric_config=metric_config,
        )
        per_method[method_name] = values
        metric_orders.append(list(values.keys()))

    if metrics is not None:
        ordered_metrics: list[str] = []
        seen: set[str] = set()
        for raw in metrics:
            key = normalize_metric_header(raw)
            if key and key not in seen:
                seen.add(key)
                ordered_metrics.append(key)
        if not ordered_metrics:
            raise ValueError("metrics must contain at least one metric name.")
    else:
        common = set(metric_orders[0])
        for order in metric_orders[1:]:
            common &= set(order)
        ordered_metrics = [key for key in metric_orders[0] if key in common]
        if not ordered_metrics:
            raise ValueError("Selected methods share no common metric columns.")

    missing_requested = [
        metric
        for metric in ordered_metrics
        if any(metric not in per_method[name] for name in per_method)
    ]
    warnings: list[str] = []
    if missing_requested:
        warnings.append(
            "Missing metrics filled with NaN: " + ", ".join(sorted(set(missing_requested)))
        )

    aligned, sample_count, align_warnings = _align_values(per_method, ordered_metrics)
    warnings.extend(align_warnings)

    return MetricSummaryData(
        benchmark=benchmark_name,
        methods=[name for name, _ in selected],
        metrics=ordered_metrics,
        values=aligned,
        sample_count=sample_count,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Statistics and summary table
# ---------------------------------------------------------------------------


def summarize_metric(values: np.ndarray) -> dict[str, float]:
    """Return mean / std / median ignoring non-finite entries."""

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan")}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=0)),
        "median": float(np.median(finite)),
    }


def best_method_for_metric(data: MetricSummaryData, metric: str) -> str | None:
    """Return the method with the best mean for ``metric`` (NaN-aware)."""

    if metric not in data.metrics:
        return None
    higher = metric_higher_is_better(metric)
    best_name: str | None = None
    best_value: float | None = None
    for method in data.methods:
        stats = summarize_metric(data.method_values(method, metric))
        value = stats["mean"]
        if not np.isfinite(value):
            continue
        if best_value is None or (value > best_value if higher else value < best_value):
            best_name = method
            best_value = value
    return best_name


def format_summary_cell(stats: Mapping[str, float], *, include_median: bool = False) -> str:
    mean = stats["mean"]
    std = stats["std"]
    if not np.isfinite(mean):
        return "—"
    if include_median and np.isfinite(stats.get("median", float("nan"))):
        return f"{mean:.4f} ± {std:.4f} (med {stats['median']:.4f})"
    return f"{mean:.4f} ± {std:.4f}"


def write_summary_csv(
    data: MetricSummaryData,
    path: Path,
    *,
    include_median: bool = False,
    highlight_best: bool = True,
) -> Path:
    """Write a method × metric summary table (``mean ± std`` cells)."""

    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["Method", *[metric_display_name(m) for m in data.metrics]]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        best = {
            metric: best_method_for_metric(data, metric)
            for metric in data.metrics
        } if highlight_best else {}
        for method in data.methods:
            row = [method]
            for metric in data.metrics:
                cell = format_summary_cell(
                    summarize_metric(data.method_values(method, metric)),
                    include_median=include_median,
                )
                if highlight_best and best.get(metric) == method:
                    cell = f"{cell} *"
                row.append(cell)
            writer.writerow(row)
        writer.writerow(
            [
                "Best",
                *[
                    best.get(metric) or "—"
                    for metric in data.metrics
                ],
            ]
        )
    return path


# ---------------------------------------------------------------------------
# Box-plot figure
# ---------------------------------------------------------------------------


def _subplot_layout(metric_count: int) -> tuple[int, int]:
    columns = min(metric_count, _MAX_PANEL_COLUMNS)
    rows = (metric_count + columns - 1) // columns
    return rows, columns


def create_boxplot_figure(
    data: MetricSummaryData,
    *,
    highlight_methods: Sequence[str] = (),
    show_mean: bool = True,
    show_median: bool = True,
) -> Figure:
    """Draw one box-plot panel per metric; X-axis follows ``data.methods``."""

    if not data.metrics:
        raise ValueError("At least one metric is required for the box-plot figure.")
    if not data.methods:
        raise ValueError("At least one method is required for the box-plot figure.")

    highlight = {name.casefold() for name in highlight_methods}
    rows, columns = _subplot_layout(len(data.metrics))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(_PANEL_WIDTH_INCHES * columns, _PANEL_HEIGHT_INCHES * rows),
        squeeze=False,
    )

    positions = np.arange(1, len(data.methods) + 1)
    for panel_index, metric in enumerate(data.metrics):
        axis = axes[panel_index // columns][panel_index % columns]
        series = [
            data.method_values(method, metric) for method in data.methods
        ]
        boxes = axis.boxplot(
            series,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            showmeans=show_mean,
            meanprops={
                "marker": "D",
                "markerfacecolor": "#f39c12",
                "markeredgecolor": "#b9770e",
                "markersize": 5,
            },
            medianprops={"color": "#2c3e50", "linewidth": 1.4},
            whiskerprops={"color": "#7f8c8d"},
            capprops={"color": "#7f8c8d"},
        )

        for index, method in enumerate(data.methods):
            box = boxes["boxes"][index]
            is_highlight = method.casefold() in highlight
            box.set(
                facecolor=_HIGHLIGHT_COLOR if is_highlight else _DEFAULT_BOX_COLOR,
                edgecolor=_HIGHLIGHT_COLOR if is_highlight else "#2f4b7c",
                linewidth=2.2 if is_highlight else 0.9,
                alpha=0.55 if is_highlight else 0.75,
            )

        direction = metric_direction_label(metric)
        axis.set_title(f"{metric_display_name(metric)}  ({direction})")
        axis.set_xticks(positions)
        axis.set_xticklabels(data.methods, rotation=30, ha="right")
        axis.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
        axis.set_ylabel(metric_display_name(metric))

    for panel_index in range(len(data.metrics), rows * columns):
        axes[panel_index // columns][panel_index % columns].axis("off")

    figure.suptitle(
        f"{data.benchmark}  ·  N={data.sample_count} test images",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return figure


def _normalize_format(output_format: str) -> str:
    key = output_format.strip().casefold()
    if key not in _FORMAT_SUFFIXES:
        raise ValueError(f"Unsupported format {output_format!r}.")
    return _FORMAT_SUFFIXES[key]


def _save_figure(
    figure: Figure,
    path: Path,
    *,
    output_format: str,
    dpi: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if output_format in {"png", "jpeg"}:
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
    else:
        figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def render_metric_summary(
    data: MetricSummaryData,
    *,
    output_directory: Path,
    output_format: str = "png",
    dpi: int = 300,
    highlight_methods: Sequence[str] = (),
    show_mean: bool = True,
    show_median: bool = True,
    include_median_in_csv: bool = False,
    export_summary_csv: bool = True,
) -> list[Path]:
    """Render the box-plot figure and optionally the summary CSV."""

    normalized = _normalize_format(output_format)
    output_directory = output_directory.expanduser()
    output_directory.mkdir(parents=True, exist_ok=True)

    figure = create_boxplot_figure(
        data,
        highlight_methods=highlight_methods,
        show_mean=show_mean,
        show_median=show_median,
    )
    figure_path = output_directory / f"metric_summary_{data.benchmark}.{normalized}"
    _save_figure(figure, figure_path, output_format=normalized, dpi=dpi)

    written = [figure_path]
    if export_summary_csv:
        written.append(
            write_summary_csv(
                data,
                output_directory / f"metric_summary_{data.benchmark}_summary.csv",
                include_median=include_median_in_csv,
            )
        )
    return written


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------


class _GuiTaskRunner:
    """Background calculation threads for MetricSummaryGui."""

    def __init__(self, root: object) -> None:
        self._root = root
        self._busy = threading.Event()
        self._buttons: list[object] = []

    @property
    def is_busy(self) -> bool:
        return self._busy.is_set()

    def register_button(self, button: object) -> None:
        self._buttons.append(button)

    def _set_buttons_state(self, state: str) -> None:
        for btn in self._buttons:
            btn.configure(state=state)

    def run_export(
        self,
        settings: dict[str, object],
        on_done: Callable[[list[Path], MetricSummaryData | None, str | None], None],
    ) -> None:
        if self._busy.is_set():
            raise ValueError("A calculation is already running.")
        self._busy.set()
        self._set_buttons_state("disabled")
        threading.Thread(
            target=self._export_worker, args=(settings, on_done), daemon=True
        ).start()

    def run_preview(
        self,
        settings: dict[str, object],
        on_done: Callable[[MetricSummaryData | None, dict[str, object], str | None], None],
    ) -> None:
        if self._busy.is_set():
            raise ValueError("A calculation is already running.")
        self._busy.set()
        self._set_buttons_state("disabled")
        threading.Thread(
            target=self._preview_worker, args=(settings, on_done), daemon=True
        ).start()

    def _export_worker(
        self,
        settings: dict[str, object],
        on_done: Callable[[list[Path], MetricSummaryData | None, str | None], None],
    ) -> None:
        try:
            data = load_metric_summary_data(
                output_root=settings["output_root"],
                dataset_root=settings["dataset_root"],
                benchmark=settings["benchmark"],
                methods=settings["methods"],
                metrics=settings["metrics"],
                prefer_csv=settings["prefer_csv"],
                recompute_missing=settings["recompute_missing"],
            )
            written = render_metric_summary(
                data,
                output_directory=settings["output_directory"],
                output_format=settings["output_format"],
                dpi=settings["dpi"],
                highlight_methods=settings["highlights"],
                show_mean=settings["show_mean"],
                show_median=settings["show_median"],
                include_median_in_csv=settings["include_median"],
                export_summary_csv=settings["export_summary_csv"],
            )
        except (FileNotFoundError, IndexError, KeyError, OSError, ValueError) as error:
            self._root.after(0, on_done, [], None, str(error))
            return
        finally:
            self._busy.clear()
            self._root.after(0, self._set_buttons_state, "normal")
        self._root.after(0, on_done, written, data, None)

    def _preview_worker(
        self,
        settings: dict[str, object],
        on_done: Callable[[MetricSummaryData | None, dict[str, object], str | None], None],
    ) -> None:
        try:
            data = load_metric_summary_data(
                output_root=settings["output_root"],
                dataset_root=settings["dataset_root"],
                benchmark=settings["benchmark"],
                methods=settings["methods"],
                metrics=settings["metrics"],
                prefer_csv=settings["prefer_csv"],
                recompute_missing=settings["recompute_missing"],
            )
        except (FileNotFoundError, IndexError, KeyError, OSError, ValueError) as error:
            self._root.after(0, on_done, None, settings, str(error))
            return
        finally:
            self._busy.clear()
            self._root.after(0, self._set_buttons_state, "normal")
        self._root.after(0, on_done, data, settings, None)


class MetricSummaryGui:
    """Tkinter front end for multi-metric summary box plots."""

    _PREVIEW_DPI = 100

    def __init__(self, *, output_root: Path, dataset_root: Path) -> None:
        try:
            import tkinter as tk
            from tkinter import filedialog, messagebox, ttk

            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except ImportError as error:  # pragma: no cover - platform-owned dependency
            raise RuntimeError("The metric-summary GUI requires Tkinter.") from error

        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.figure_canvas_type = FigureCanvasTkAgg
        self.root = tk.Tk()
        self.root.title("HSMSFusion Metric Summary")
        self.root.geometry("1280x980")
        self.root.minsize(1000, 720)

        self.output_root = tk.StringVar(value=str(output_root))
        self.dataset_root = tk.StringVar(value=str(dataset_root))
        self.dataset_name = tk.StringVar()
        self.figure_directory = tk.StringVar()
        self.export_format = tk.StringVar(value="png")
        self.export_dpi = tk.IntVar(value=300)
        self.prefer_csv = tk.BooleanVar(value=True)
        self.recompute_missing = tk.BooleanVar(value=True)
        self.show_mean = tk.BooleanVar(value=True)
        self.show_median = tk.BooleanVar(value=True)
        self.include_median = tk.BooleanVar(value=False)
        self.export_summary_csv = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="Scan an output directory to begin.")

        self.candidates: dict[str, list[tuple[str, Path]]] = {}
        self.method_order: list[str] = []
        self._included_methods: set[str] = set()
        self._highlighted_methods: set[str] = set()
        self._drag_index: int | None = None
        self._dragging_method = False
        self._metric_variables: dict[str, object] = {}
        self._all_metric_keys = [
            normalize_metric_header(name) for name in _DEFAULT_LEADERBOARD
        ]

        self._task_runner = _GuiTaskRunner(self.root)
        self.preview_canvas = None
        self.preview_figure: Figure | None = None
        self.preview_placeholder: object | None = None

        self._build_layout()
        self._task_runner.register_button(self.generate_button)
        self._task_runner.register_button(self.preview_button)
        self.scan_outputs()

    def _build_layout(self) -> None:
        controls = self.ttk.Frame(self.root, padding=10)
        controls.pack(fill="x")
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(5, weight=1)

        self._build_path_row(controls)
        self._build_dataset_row(controls)
        self._build_method_frame(controls)
        self._build_metric_and_export(controls)

        preview_frame = self.ttk.LabelFrame(
            self.root, text="Box-plot preview (one panel per selected metric)", padding=6
        )
        preview_frame.pack(fill="both", expand=True, padx=10, pady=(0, 4))
        self.preview_container = self.ttk.Frame(preview_frame)
        self.preview_container.pack(fill="both", expand=True)
        self.preview_container.bind("<Configure>", self._on_preview_container_resize)
        self.preview_placeholder = self.ttk.Label(
            self.preview_container,
            text="Select methods and metrics, then click 'Preview' to draw box plots.",
            anchor="center",
        )
        self.preview_placeholder.pack(fill="both", expand=True)

        table_frame = self.ttk.LabelFrame(
            self.root, text="Summary table (mean ± std; * = best mean)", padding=6
        )
        table_frame.pack(fill="x", padx=10, pady=(0, 4))
        self.summary_tree = self.ttk.Treeview(
            table_frame, height=5, show="headings"
        )
        self.summary_tree.pack(fill="x")

        self.ttk.Label(self.root, textvariable=self.status, anchor="w", padding=(12, 7)).pack(
            fill="x"
        )

    def _build_path_row(self, controls: object) -> None:
        self.ttk.Label(controls, text="Output root").grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.ttk.Entry(controls, textvariable=self.output_root).grid(row=0, column=1, sticky="ew")
        self.ttk.Button(controls, text="Browse", command=self._browse_output_root).grid(
            row=0, column=2, sticky="w", padx=6
        )
        self.ttk.Button(controls, text="Scan", command=self.scan_outputs).grid(
            row=0, column=3, sticky="w"
        )
        self.ttk.Label(controls, text="Dataset root").grid(
            row=0, column=4, sticky="w", padx=(18, 6)
        )
        self.ttk.Entry(controls, textvariable=self.dataset_root).grid(
            row=0, column=5, sticky="ew"
        )
        self.ttk.Button(controls, text="Browse", command=self._browse_dataset_root).grid(
            row=0, column=6, sticky="w", padx=6
        )

    def _build_dataset_row(self, controls: object) -> None:
        self.ttk.Label(controls, text="Dataset-scale").grid(
            row=1, column=0, sticky="w", pady=(10, 0)
        )
        self.dataset_combo = self.ttk.Combobox(
            controls, textvariable=self.dataset_name, state="readonly", width=25
        )
        self.dataset_combo.grid(row=1, column=1, sticky="w", pady=(10, 0))
        self.dataset_combo.bind("<<ComboboxSelected>>", self._on_dataset_changed)

    def _build_method_frame(self, controls: object) -> None:
        self.method_frame = self.ttk.LabelFrame(
            controls,
            text=(
                "Methods — list order = box / legend order; "
                "selection = include; drag or Up/Down to reorder"
            ),
            padding=7,
        )
        self.method_frame.grid(row=2, column=0, columnspan=7, sticky="ew", pady=(12, 0))
        self.method_frame.columnconfigure(0, weight=1)
        self.method_frame.columnconfigure(3, weight=1)

        self.ttk.Label(self.method_frame, text="Box order (selected = plot)").grid(
            row=0, column=0, sticky="w"
        )
        list_frame = self.ttk.Frame(self.method_frame)
        list_frame.grid(row=1, column=0, sticky="nsew", pady=(4, 0))
        self.method_listbox = self.tk.Listbox(
            list_frame, selectmode="extended", height=7, exportselection=False
        )
        self.method_listbox.pack(side="left", fill="both", expand=True)
        method_scroll = self.ttk.Scrollbar(
            list_frame, orient="vertical", command=self.method_listbox.yview
        )
        method_scroll.pack(side="right", fill="y")
        self.method_listbox.configure(yscrollcommand=method_scroll.set)
        self.method_listbox.bind("<Button-1>", self._on_method_drag_start, add="+")
        self.method_listbox.bind("<B1-Motion>", self._on_method_drag_motion, add="+")
        self.method_listbox.bind("<ButtonRelease-1>", self._on_method_drag_end, add="+")
        self.method_listbox.bind("<Double-Button-1>", self._toggle_highlight_on_double_click)

        order_buttons = self.ttk.Frame(self.method_frame)
        order_buttons.grid(row=1, column=1, sticky="n", padx=(8, 0), pady=(4, 0))
        self.ttk.Button(order_buttons, text="Up", width=8, command=self._move_method_up).pack(
            pady=2
        )
        self.ttk.Button(order_buttons, text="Down", width=8, command=self._move_method_down).pack(
            pady=2
        )
        self.ttk.Button(
            order_buttons, text="Select all", width=8, command=self._select_all_methods
        ).pack(pady=2)
        self.ttk.Button(
            order_buttons, text="Clear", width=8, command=self._clear_method_selection
        ).pack(pady=2)

        self.ttk.Label(self.method_frame, text="Highlight (bold edge / accent)").grid(
            row=0, column=3, sticky="w", padx=(18, 0)
        )
        highlight_frame = self.ttk.Frame(self.method_frame)
        highlight_frame.grid(row=1, column=3, sticky="nsew", padx=(18, 0), pady=(4, 0))
        self.highlight_listbox = self.tk.Listbox(
            highlight_frame, selectmode="extended", height=7, exportselection=False
        )
        self.highlight_listbox.pack(side="left", fill="both", expand=True)
        highlight_scroll = self.ttk.Scrollbar(
            highlight_frame, orient="vertical", command=self.highlight_listbox.yview
        )
        highlight_scroll.pack(side="right", fill="y")
        self.highlight_listbox.configure(yscrollcommand=highlight_scroll.set)

        highlight_buttons = self.ttk.Frame(self.method_frame)
        highlight_buttons.grid(row=1, column=4, sticky="n", padx=(8, 0), pady=(4, 0))
        self.ttk.Button(
            highlight_buttons, text="Highlight", width=10, command=self._highlight_selected_methods
        ).pack(pady=2)
        self.ttk.Button(
            highlight_buttons,
            text="Unhighlight",
            width=10,
            command=self._unhighlight_selected_methods,
        ).pack(pady=2)
        self.ttk.Button(
            highlight_buttons, text="Clear", width=10, command=self._clear_highlights
        ).pack(pady=2)

    def _build_metric_and_export(self, controls: object) -> None:
        metric_frame = self.ttk.LabelFrame(controls, text="Metrics (default all)", padding=7)
        metric_frame.grid(row=3, column=0, columnspan=3, sticky="nsew", pady=(12, 0))
        for index, key in enumerate(self._all_metric_keys):
            variable = self.tk.BooleanVar(value=True)
            self._metric_variables[key] = variable
            self.ttk.Checkbutton(
                metric_frame, text=metric_display_name(key), variable=variable
            ).grid(row=index // 3, column=index % 3, sticky="w", padx=4, pady=2)

        options = self.ttk.LabelFrame(controls, text="Data options", padding=7)
        options.grid(row=3, column=3, columnspan=2, sticky="nsew", padx=(12, 0), pady=(12, 0))
        self.ttk.Checkbutton(
            options, text="Prefer metrics.csv", variable=self.prefer_csv
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        self.ttk.Checkbutton(
            options,
            text="Recompute missing from predictions.h5",
            variable=self.recompute_missing,
        ).grid(row=1, column=0, columnspan=2, sticky="w")
        self.ttk.Checkbutton(options, text="Show mean", variable=self.show_mean).grid(
            row=2, column=0, sticky="w", pady=(6, 0)
        )
        self.ttk.Checkbutton(options, text="Show median", variable=self.show_median).grid(
            row=2, column=1, sticky="w", pady=(6, 0)
        )

        export_frame = self.ttk.LabelFrame(controls, text="Export", padding=7)
        export_frame.grid(row=3, column=5, columnspan=2, sticky="nsew", padx=(12, 0), pady=(12, 0))
        export_frame.columnconfigure(1, weight=1)
        self.ttk.Label(export_frame, text="Figure directory").grid(row=0, column=0, sticky="w")
        self.ttk.Entry(export_frame, textvariable=self.figure_directory).grid(
            row=0, column=1, columnspan=2, sticky="ew", padx=6
        )
        self.ttk.Button(export_frame, text="Browse", command=self._browse_figure_directory).grid(
            row=0, column=3, sticky="w"
        )
        self.ttk.Label(export_frame, text="Format").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.ttk.Combobox(
            export_frame,
            textvariable=self.export_format,
            values=("png", "jpeg", "pdf", "svg"),
            state="readonly",
            width=8,
        ).grid(row=1, column=1, sticky="w", padx=6, pady=(8, 0))
        self.ttk.Label(export_frame, text="DPI").grid(
            row=1, column=2, sticky="w", padx=(10, 4), pady=(8, 0)
        )
        self.ttk.Spinbox(
            export_frame, from_=1, to=2400, textvariable=self.export_dpi, width=8
        ).grid(row=1, column=3, sticky="w", pady=(8, 0))
        self.ttk.Checkbutton(
            export_frame, text="Median in CSV", variable=self.include_median
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.ttk.Checkbutton(
            export_frame, text="Export summary CSV", variable=self.export_summary_csv
        ).grid(row=2, column=2, columnspan=2, sticky="w", pady=(6, 0))
        self.preview_button = self.ttk.Button(
            export_frame, text="Preview", command=self.start_preview
        )
        self.preview_button.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        self.generate_button = self.ttk.Button(
            export_frame, text="Export figure + CSV", command=self.start_export
        )
        self.generate_button.grid(row=3, column=2, columnspan=2, sticky="ew", padx=(8, 0), pady=(12, 0))

    def run(self) -> None:
        self.root.mainloop()

    def _browse_output_root(self) -> None:
        selected = self.filedialog.askdirectory(
            title="Select output root", initialdir=self.output_root.get()
        )
        if selected:
            self.output_root.set(selected)
            self.scan_outputs()

    def _browse_dataset_root(self) -> None:
        selected = self.filedialog.askdirectory(
            title="Select Dataset root", initialdir=self.dataset_root.get()
        )
        if selected:
            self.dataset_root.set(selected)

    def _browse_figure_directory(self) -> None:
        selected = self.filedialog.askdirectory(
            title="Select figure output directory",
            initialdir=self.figure_directory.get() or self.output_root.get(),
        )
        if selected:
            self.figure_directory.set(selected)

    def scan_outputs(self) -> None:
        self.candidates = discover_metric_methods(Path(self.output_root.get()))
        dataset_names = list(self.candidates)
        self.dataset_combo.configure(values=dataset_names)
        if not dataset_names:
            self.dataset_name.set("")
            self._rebuild_method_checkboxes([])
            self.status.set("No metrics.csv / predictions.h5 files were found under the output root.")
            return
        current = self.dataset_name.get()
        self.dataset_name.set(current if current in self.candidates else dataset_names[0])
        self._on_dataset_changed()

    def _on_dataset_changed(self, _event: object | None = None) -> None:
        benchmark = self.dataset_name.get()
        candidates = self.candidates.get(benchmark, [])
        self._rebuild_method_checkboxes(candidates)
        if not benchmark:
            return
        self.figure_directory.set(str(Path(self.output_root.get()) / benchmark / "metric_summary"))
        self.status.set(
            f"Loaded {len(candidates)} method(s) for {benchmark}. "
            "Select metrics and click Preview."
        )

    def _rebuild_method_checkboxes(self, candidates: Sequence[tuple[str, Path]]) -> None:
        new_names = [name for name, _path in candidates]
        if not new_names:
            self.method_order = []
            self._included_methods = set()
            self._highlighted_methods = set()
            self._refresh_method_listboxes()
            return

        previous_order = list(self.method_order)
        known = set(new_names)
        retained = [name for name in previous_order if name in known]
        appended = [name for name in new_names if name not in set(retained)]
        self.method_order = retained + appended

        if not previous_order:
            self._included_methods = set(new_names)
        else:
            self._included_methods = {name for name in self._included_methods if name in known}
            self._included_methods |= set(appended)
        self._highlighted_methods = {
            name for name in self._highlighted_methods if name in known
        }
        self._refresh_method_listboxes()

    def _refresh_method_listboxes(self) -> None:
        for listbox in (self.method_listbox, self.highlight_listbox):
            listbox.delete(0, "end")
        for name in self.method_order:
            prefix = "★ " if name in self._highlighted_methods else "    "
            self.method_listbox.insert("end", f"{prefix}{name}")
            if name in self._included_methods:
                self.method_listbox.selection_set(self.method_listbox.size() - 1)
            self.highlight_listbox.insert("end", name)
            if name in self._highlighted_methods:
                self.highlight_listbox.selection_set(self.highlight_listbox.size() - 1)

    def _display_method_name(self, text: str) -> str:
        return text[2:] if text.startswith("★ ") else text.lstrip()

    def _on_method_drag_start(self, event: object) -> None:
        index = int(self.method_listbox.nearest(getattr(event, "y", 0)))
        size = self.method_listbox.size()
        self._drag_index = index if 0 <= index < size else None
        self._dragging_method = self._drag_index is not None

    def _on_method_drag_motion(self, event: object) -> None:
        if self._drag_index is None:
            return
        target = int(self.method_listbox.nearest(getattr(event, "y", 0)))
        size = self.method_listbox.size()
        if not 0 <= target < size or target == self._drag_index:
            return
        self._reorder_method(self._drag_index, target)
        self._drag_index = target

    def _on_method_drag_end(self, _event: object) -> None:
        if not self._dragging_method:
            return
        self._dragging_method = False
        self._drag_index = None

    def _reorder_method(self, source: int, target: int) -> None:
        if source == target or not 0 <= source < len(self.method_order):
            return
        if not 0 <= target < len(self.method_order):
            return
        name = self.method_order.pop(source)
        self.method_order.insert(target, name)
        self._refresh_method_listboxes()

    def _capture_method_selection(self) -> None:
        selected = set(self.method_listbox.curselection())
        self._included_methods = {
            self._display_method_name(self.method_listbox.get(index))
            for index in selected
            if 0 <= index < self.method_listbox.size()
        }

    def _move_method_up(self) -> None:
        self._capture_method_selection()
        selected = list(self.method_listbox.curselection())
        if not selected or selected[0] == 0:
            return
        for index in selected:
            if index <= 0:
                continue
            self.method_order[index], self.method_order[index - 1] = (
                self.method_order[index - 1],
                self.method_order[index],
            )
        self._refresh_method_listboxes()

    def _move_method_down(self) -> None:
        self._capture_method_selection()
        selected = list(self.method_listbox.curselection())
        last = len(self.method_order) - 1
        if not selected or selected[-1] >= last:
            return
        for index in reversed(selected):
            if index >= last:
                continue
            self.method_order[index], self.method_order[index + 1] = (
                self.method_order[index + 1],
                self.method_order[index],
            )
        self._refresh_method_listboxes()

    def _select_all_methods(self) -> None:
        self.method_listbox.selection_set(0, "end")
        self._included_methods = set(self.method_order)

    def _clear_method_selection(self) -> None:
        self.method_listbox.selection_clear(0, "end")
        self._included_methods = set()

    def _toggle_highlight_on_double_click(self, event: object) -> None:
        index = int(self.method_listbox.nearest(getattr(event, "y", 0)))
        if not 0 <= index < len(self.method_order):
            return
        name = self.method_order[index]
        if name in self._highlighted_methods:
            self._highlighted_methods.discard(name)
        else:
            self._highlighted_methods.add(name)
        self._refresh_method_listboxes()

    def _highlight_selected_methods(self) -> None:
        for index in self.highlight_listbox.curselection():
            if 0 <= index < len(self.method_order):
                self._highlighted_methods.add(self.method_order[index])
        self._refresh_method_listboxes()

    def _unhighlight_selected_methods(self) -> None:
        for index in self.highlight_listbox.curselection():
            if 0 <= index < len(self.method_order):
                self._highlighted_methods.discard(self.method_order[index])
        self._refresh_method_listboxes()

    def _clear_highlights(self) -> None:
        self._highlighted_methods.clear()
        self._refresh_method_listboxes()

    def _selected_metrics(self) -> list[str] | None:
        selected = [key for key, var in self._metric_variables.items() if var.get()]
        return selected or None

    def _selected_methods(self) -> list[str]:
        self._capture_method_selection()
        return [name for name in self.method_order if name in self._included_methods]

    def _settings(self) -> dict[str, object]:
        benchmark = self.dataset_name.get()
        if not benchmark:
            raise ValueError("Choose a dataset-scale output folder first.")
        methods = self._selected_methods()
        if not methods:
            raise ValueError("Select at least one method.")
        return {
            "output_root": Path(self.output_root.get()),
            "dataset_root": Path(self.dataset_root.get()),
            "benchmark": benchmark,
            "methods": methods,
            "metrics": self._selected_metrics(),
            "prefer_csv": bool(self.prefer_csv.get()),
            "recompute_missing": bool(self.recompute_missing.get()),
            "output_directory": Path(
                self.figure_directory.get()
                or Path(self.output_root.get()) / benchmark / "metric_summary"
            ),
            "output_format": self.export_format.get(),
            "dpi": int(self.export_dpi.get()),
            "highlights": sorted(self._highlighted_methods),
            "show_mean": bool(self.show_mean.get()),
            "show_median": bool(self.show_median.get()),
            "include_median": bool(self.include_median.get()),
            "export_summary_csv": bool(self.export_summary_csv.get()),
        }

    def start_preview(self) -> None:
        try:
            settings = self._settings()
        except ValueError as error:
            self.status.set(str(error))
            self.messagebox.showwarning("Metric summary", str(error))
            return
        self.status.set("Computing metric summary…")
        self._task_runner.run_preview(settings, self._on_preview_done)

    def start_export(self) -> None:
        try:
            settings = self._settings()
        except ValueError as error:
            self.status.set(str(error))
            self.messagebox.showwarning("Metric summary", str(error))
            return
        self.status.set("Exporting box plots…")
        self._task_runner.run_export(settings, self._on_export_done)

    def _fill_summary_table(self, data: MetricSummaryData) -> None:
        columns = ["Method", *[metric_display_name(m) for m in data.metrics]]
        self.summary_tree.configure(columns=columns, show="headings")
        for column in columns:
            self.summary_tree.heading(column, text=column)
            self.summary_tree.column(column, width=110, anchor="center")
        for item in self.summary_tree.get_children():
            self.summary_tree.delete(item)
        best = {metric: best_method_for_metric(data, metric) for metric in data.metrics}
        for method in data.methods:
            row = [method]
            for metric in data.metrics:
                cell = format_summary_cell(summarize_metric(data.method_values(method, metric)))
                if best.get(metric) == method:
                    cell = f"{cell} *"
                row.append(cell)
            self.summary_tree.insert("", "end", values=row)

    def _on_preview_done(
        self,
        data: MetricSummaryData | None,
        settings: dict[str, object],
        error: str | None,
    ) -> None:
        if error is not None or data is None:
            message = error or "Preview failed."
            self.status.set(message)
            self.messagebox.showerror("Preview failed", message)
            return

        figure = create_boxplot_figure(
            data,
            highlight_methods=settings["highlights"],
            show_mean=settings["show_mean"],
            show_median=settings["show_median"],
        )
        self._replace_preview(figure)
        self._fill_summary_table(data)
        warning_suffix = f"  |  {'; '.join(data.warnings)}" if data.warnings else ""
        self.status.set(
            f"{data.benchmark}: {len(data.methods)} methods × {len(data.metrics)} metrics, "
            f"N={data.sample_count}{warning_suffix}"
        )

    def _on_export_done(
        self,
        written: list[Path],
        data: MetricSummaryData | None,
        error: str | None,
    ) -> None:
        if error is not None:
            self.status.set(f"Export failed: {error}")
            self.messagebox.showerror("Export failed", error)
            return
        if data is not None:
            self._fill_summary_table(data)
        paths = ", ".join(str(path) for path in written)
        self.status.set(f"Exported {len(written)} file(s): {paths}")

    def _on_preview_container_resize(self, _event: object) -> None:
        if self.preview_figure is None or self.preview_canvas is None:
            return
        width = self.preview_container.winfo_width()
        height = self.preview_container.winfo_height()
        if width < 100 or height < 100:
            return
        self.preview_figure.set_size_inches(
            width / self._PREVIEW_DPI, height / self._PREVIEW_DPI
        )
        self.preview_canvas.draw_idle()

    def _replace_preview(self, figure: Figure) -> None:
        if self.preview_canvas is not None:
            self.preview_canvas.get_tk_widget().destroy()
            self.preview_canvas = None
        if self.preview_figure is not None:
            self.preview_figure.clear()
            self.preview_figure = None
        if self.preview_placeholder is not None:
            self.preview_placeholder.destroy()
            self.preview_placeholder = None
        self.preview_figure = figure
        self.preview_canvas = self.figure_canvas_type(figure, master=self.preview_container)
        self.preview_canvas.draw()
        self.preview_canvas.get_tk_widget().pack(fill="both", expand=True)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def launch_gui(*, output_root: Path, dataset_root: Path) -> None:
    """Launch the interactive metric-summary front end."""

    matplotlib.use("TkAgg")
    MetricSummaryGui(output_root=output_root, dataset_root=dataset_root).run()


def main(argv: Sequence[str] | None = None) -> int:
    """Launch the GUI, or render headlessly when CLI flags are given."""

    parser = argparse.ArgumentParser(
        description="HSMSFusion multi-metric summary and box-plot tool"
    )
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--dataset-root", type=Path, default=Path("Dataset"))
    parser.add_argument("--benchmark", help="Dataset-scale folder name, e.g. chikusei_x4")
    parser.add_argument("--method", action="append", dest="methods", help="Method name (repeatable)")
    parser.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        help="Metric key or CSV header (repeatable; default all common)",
    )
    parser.add_argument("--highlight", action="append", dest="highlights", help="Highlighted method")
    parser.add_argument(
        "--output-dir",
        "--output-directory",
        type=Path,
        dest="output_directory",
        default=Path("figures/metrics"),
    )
    parser.add_argument("--format", default="png", choices=sorted(_FORMAT_SUFFIXES))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--no-prefer-csv",
        action="store_true",
        help="Recompute metrics even when metrics.csv exists",
    )
    parser.add_argument(
        "--no-recompute",
        action="store_true",
        help="Fail instead of recomputing a missing metrics.csv",
    )
    parser.add_argument("--no-mean", action="store_true", help="Hide mean markers")
    parser.add_argument("--median-in-csv", action="store_true", help="Include median in summary CSV")
    parser.add_argument("--no-summary-csv", action="store_true", help="Skip summary CSV export")
    parser.add_argument("--gui", action="store_true", help="Force the interactive GUI")

    arguments = parser.parse_args(argv)

    use_gui = arguments.gui or arguments.benchmark is None
    if use_gui:
        launch_gui(
            output_root=arguments.output_root, dataset_root=arguments.dataset_root
        )
        return 0

    try:
        data = load_metric_summary_data(
            output_root=arguments.output_root,
            dataset_root=arguments.dataset_root,
            benchmark=arguments.benchmark,
            methods=arguments.methods,
            metrics=arguments.metrics,
            prefer_csv=not arguments.no_prefer_csv,
            recompute_missing=not arguments.no_recompute,
        )
        written = render_metric_summary(
            data,
            output_directory=arguments.output_directory,
            output_format=arguments.format,
            dpi=arguments.dpi,
            highlight_methods=arguments.highlights or (),
            show_mean=not arguments.no_mean,
            show_median=True,
            include_median_in_csv=arguments.median_in_csv,
            export_summary_csv=not arguments.no_summary_csv,
        )
    except (FileNotFoundError, IndexError, KeyError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    for path in written:
        print(path)
    if data.warnings:
        for warning in data.warnings:
            print(f"warning: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
