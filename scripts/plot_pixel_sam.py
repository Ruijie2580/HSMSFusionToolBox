"""Plot sorted per-pixel SAM curves for multiple fusion methods and samples.

The toolbox test export uses the following layout::

    output/<dataset>_x<scale>/<method>/predictions.h5  (/prediction, NCHW)
    Dataset/<dataset>/test.h5                           (/HRHS, NCHW)

For every selected sample the script computes the spectral angle of each
spatial pixel between a fused cube and the ground-truth HRHS, converts it to
degrees, sorts the values in ascending order, and draws one curve per method
against ``Pixel Index (sorted)``.  Multiple samples are laid out as a shared
grid (up to 2 x 3 panels) with one figure-level legend; a highlighted method
is drawn as a thick solid red line while the remaining methods are dashed.

Run the script without arguments to launch the GUI::

    python scripts/plot_pixel_sam.py
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

# Allow ``python scripts/plot_pixel_sam.py`` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, LogLocator, NullLocator

from scripts._shared import (
    discover_output_predictions,
    infer_target_path,
    resolve_benchmark,
    sample_count,
    select_predictions,
    setup_matplotlib_font,
    validate_hsi_dataset,
)

setup_matplotlib_font()

_FORMAT_SUFFIXES = {"png": "png", "jpg": "jpeg", "jpeg": "jpeg", "pdf": "pdf", "svg": "svg"}
_SAM_EPSILON = 1e-8
_MAX_SAMPLES = 6
_GRID_COLUMNS = 3
_PANEL_WIDTH_INCHES = 5.4
_PANEL_HEIGHT_INCHES = 3.4
# Log ticks that match the paper-style SAM panel in the reference figure.
_DEFAULT_LOG_TICKS = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
_HIGHLIGHT_COLOR = "#e74c3c"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PixelSamData:
    """Sorted per-pixel SAM curves for one benchmark across methods."""

    benchmark: str
    target_path: Path
    methods: list[str]
    per_image: dict[str, np.ndarray]
    image_indices: list[int]

    @property
    def sample_count(self) -> int:
        return len(self.image_indices)

    @property
    def pixel_count(self) -> int:
        return int(next(iter(self.per_image.values())).shape[1])


@dataclass(frozen=True)
class SamplePixelSam:
    """Sorted SAM curves of one sample for every compared method."""

    benchmark: str
    image_index: int
    curves: dict[str, np.ndarray]

    @property
    def pixel_count(self) -> int:
        return int(next(iter(self.curves.values())).shape[0])


@dataclass(frozen=True)
class SampleComparisonData:
    """Sorted SAM of up to 6 samples that may come from different benchmarks."""

    methods: list[str]
    samples: list[SamplePixelSam]
    benchmark_data: dict[str, PixelSamData]


# ---------------------------------------------------------------------------
# SAM computation
# ---------------------------------------------------------------------------


def pixel_sam_degrees(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    clip_to_unit: bool = True,
    epsilon: float = _SAM_EPSILON,
) -> np.ndarray:
    """Return the per-pixel spectral angle in degrees for one CHW sample.

    Both cubes must share the same CHW shape.  Values are optionally clipped
    to ``[0, 1]`` before the angle is evaluated so the visualization matches
    the residual-comparison tool.
    """

    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction shape {prediction.shape} does not match target shape {target.shape}."
        )
    if prediction.ndim != 3:
        raise ValueError(f"Expected CHW samples, got shape {prediction.shape}.")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Prediction and target values must be finite.")
    if epsilon <= 0.0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be a finite positive value.")

    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if clip_to_unit:
        pred = np.clip(pred, 0.0, 1.0)
        truth = np.clip(truth, 0.0, 1.0)

    channels = pred.shape[0]
    height, width = pred.shape[1], pred.shape[2]
    pred_flat = pred.reshape(channels, -1)
    truth_flat = truth.reshape(channels, -1)
    numerator = np.sum(pred_flat * truth_flat, axis=0)
    denominator = np.linalg.norm(pred_flat, axis=0) * np.linalg.norm(truth_flat, axis=0)
    cosine = numerator / np.maximum(denominator, epsilon)
    cosine = np.clip(cosine, -1.0, 1.0)
    angles = (np.arccos(cosine) * (180.0 / np.pi)).astype(np.float32)
    return angles.reshape(height, width)


def sorted_pixel_sam_degrees(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    clip_to_unit: bool = True,
    epsilon: float = _SAM_EPSILON,
) -> np.ndarray:
    """Return the ascending-sorted 1-D SAM curve (degrees) of one sample."""

    values = pixel_sam_degrees(
        prediction, target, clip_to_unit=clip_to_unit, epsilon=epsilon
    )
    return np.sort(values.reshape(-1))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _validate_indices(
    total_sample_count: int,
    image_indices: Sequence[int] | None,
    *,
    benchmark_name: str,
    max_samples: int | None = None,
) -> list[int]:
    if image_indices is None:
        selected_indices = list(range(total_sample_count))
        if max_samples is not None and len(selected_indices) > max_samples:
            raise ValueError(
                f"At most {max_samples} samples can be compared at once; "
                f"{benchmark_name} has {total_sample_count}."
            )
        return selected_indices

    selected_indices = list(image_indices)
    if not selected_indices:
        raise ValueError("image_indices must contain at least one image index.")
    if len(set(selected_indices)) != len(selected_indices):
        raise ValueError("image_indices must not contain duplicates.")
    if max_samples is not None and len(selected_indices) > max_samples:
        raise ValueError(f"At most {max_samples} samples can be compared at once.")
    invalid = [i for i in selected_indices if not 0 <= i < total_sample_count]
    if invalid:
        raise IndexError(
            f"image index {invalid[0]} is outside [0, {total_sample_count - 1}] "
            f"for {benchmark_name}."
        )
    return selected_indices


def load_pixel_sam_data(
    *,
    output_root: Path,
    dataset_root: Path,
    benchmark: str | None,
    methods: Sequence[str] | None = None,
    target_path: Path | None = None,
    target_key: str = "HRHS",
    prediction_key: str = "prediction",
    image_indices: Sequence[int] | None = None,
    clip_to_unit: bool = True,
    epsilon: float = _SAM_EPSILON,
) -> PixelSamData:
    """Read selected exports and compute sorted per-pixel SAM for each sample."""

    if epsilon <= 0.0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be a finite positive value.")

    benchmark_name, candidates = resolve_benchmark(
        discover_output_predictions(output_root), benchmark
    )
    selected = select_predictions(candidates, methods)
    if not selected:
        raise ValueError("At least one method must be selected.")

    resolved_target_path = (
        target_path or infer_target_path(dataset_root, benchmark_name)
    ).expanduser()
    total_sample_count = sample_count(resolved_target_path, dataset_key=target_key)
    selected_indices = _validate_indices(
        total_sample_count, image_indices, benchmark_name=benchmark_name
    )

    with h5py.File(resolved_target_path, "r") as target_handle:
        if target_key not in target_handle:
            raise KeyError(f"{resolved_target_path} does not contain {target_key!r}.")
        target_dataset = target_handle[target_key]
        target_shape = validate_hsi_dataset(
            target_dataset, path=resolved_target_path, key=target_key
        )
        target_samples = {
            image_index: np.asarray(target_dataset[image_index], dtype=np.float32)
            for image_index in selected_indices
        }

    pixel_count = int(target_shape[2] * target_shape[3])
    per_image: dict[str, np.ndarray] = {}
    for method_name, prediction_path in selected:
        with h5py.File(prediction_path, "r") as prediction_handle:
            if prediction_key not in prediction_handle:
                raise KeyError(f"{prediction_path} does not contain {prediction_key!r}.")
            prediction_dataset = prediction_handle[prediction_key]
            prediction_shape = validate_hsi_dataset(
                prediction_dataset, path=prediction_path, key=prediction_key
            )
            if prediction_shape[1:] != target_shape[1:]:
                raise ValueError(
                    f"{method_name} prediction shape {prediction_shape} does not match "
                    f"target shape {target_shape}."
                )
            if prediction_shape[0] < total_sample_count:
                raise IndexError(
                    f"{method_name} prediction has {prediction_shape[0]} samples "
                    f"but target has {total_sample_count}."
                )
            values = np.empty((len(selected_indices), pixel_count), dtype=np.float32)
            for row_index, image_index in enumerate(selected_indices):
                prediction_sample = np.asarray(
                    prediction_dataset[image_index], dtype=np.float32
                )
                values[row_index] = sorted_pixel_sam_degrees(
                    prediction_sample,
                    target_samples[image_index],
                    clip_to_unit=clip_to_unit,
                    epsilon=epsilon,
                )
            per_image[method_name] = values

    return PixelSamData(
        benchmark=benchmark_name,
        target_path=resolved_target_path,
        methods=[name for name, _ in selected],
        per_image=per_image,
        image_indices=selected_indices,
    )


def load_sample_pixel_sam_data(
    *,
    output_root: Path,
    dataset_root: Path,
    selections: Sequence[tuple[str, int]],
    methods: Sequence[str] | None = None,
    clip_to_unit: bool = True,
    epsilon: float = _SAM_EPSILON,
) -> SampleComparisonData:
    """Load sorted SAM curves for (benchmark, image) selections across benchmarks.

    Only the selected samples are read.  The comparison keeps the methods
    available in every selected benchmark, matched case-insensitively, so all
    panels share one legend.
    """

    if not selections:
        raise ValueError("At least one sample selection is required.")
    if len(set(selections)) != len(selections):
        raise ValueError("Sample selections must not contain duplicates.")
    if len(selections) > _MAX_SAMPLES:
        raise ValueError(f"At most {_MAX_SAMPLES} samples can be compared at once.")

    benchmarks = sorted({benchmark for benchmark, _ in selections})
    all_candidates = discover_output_predictions(output_root)
    benchmark_data: dict[str, PixelSamData] = {}
    for benchmark in benchmarks:
        indices = sorted({index for name, index in selections if name == benchmark})
        wanted = None
        if methods is not None:
            available = {name.casefold() for name, _ in all_candidates.get(benchmark, [])}
            wanted = [name for name in methods if name.casefold() in available]
            if not wanted:
                raise ValueError("The compared datasets share no fusion method.")
        benchmark_data[benchmark] = load_pixel_sam_data(
            output_root=output_root,
            dataset_root=dataset_root,
            benchmark=benchmark,
            methods=wanted,
            image_indices=indices,
            clip_to_unit=clip_to_unit,
            epsilon=epsilon,
        )

    common: dict[str, str] = {}
    for name in benchmark_data[benchmarks[0]].methods:
        common.setdefault(name.casefold(), name)
    for benchmark in benchmarks[1:]:
        available = {name.casefold() for name in benchmark_data[benchmark].methods}
        common = {key: value for key, value in common.items() if key in available}
    common_names = list(common.values())
    if not common_names:
        raise ValueError("The compared datasets share no fusion method.")

    samples = []
    for benchmark, image_index in selections:
        single = benchmark_data[benchmark]
        row = single.image_indices.index(image_index)
        samples.append(
            SamplePixelSam(
                benchmark=benchmark,
                image_index=image_index,
                curves={name: single.per_image[name][row] for name in common_names},
            )
        )
    return SampleComparisonData(
        methods=common_names, samples=samples, benchmark_data=benchmark_data
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _method_style(
    method: str,
    index: int,
    highlights: set[str],
    *,
    method_count: int,
) -> dict[str, object]:
    if method.casefold() in highlights:
        return {
            "color": _HIGHLIGHT_COLOR,
            "linewidth": 2.4,
            "linestyle": "-",
            "zorder": 10,
        }
    if method_count <= 20:
        colors = plt.get_cmap("tab20").colors
        color = colors[index % len(colors)]
    else:
        cmap = plt.get_cmap("viridis")
        color = cmap(index / max(method_count - 1, 1))
    return {
        "color": color,
        "linewidth": 1.35,
        "linestyle": "--",
        "zorder": 1,
    }


_LEGEND_HANDLE_INCHES = 0.6
_LEGEND_CHAR_INCHES = 0.085


def _legend_layout(labels: Sequence[str], *, figure_width: float) -> tuple[int, float]:
    longest = max((len(label) for label in labels), default=0)
    column_width = _LEGEND_HANDLE_INCHES + _LEGEND_CHAR_INCHES * longest
    columns = max(1, min(11, len(labels), int(figure_width // column_width)))
    rows = math.ceil(len(labels) / columns)
    return columns, 0.45 + 0.28 * rows


def _add_shared_legend(
    figure: Figure,
    handles: Sequence[object],
    labels: Sequence[str],
    *,
    columns: int,
    font_scale: float = 1.0,
) -> None:
    figure.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=columns,
        fontsize=10 * font_scale,
        frameon=True,
        fancybox=False,
        edgecolor="#bbbbbb",
    )


def _sample_log_ticks(values: Sequence[np.ndarray]) -> list[float]:
    """Pick a readable log-tick subset that covers the observed SAM range."""

    flat = np.concatenate([np.asarray(curve, dtype=np.float64).ravel() for curve in values])
    if flat.size == 0:
        return list(_DEFAULT_LOG_TICKS)
    positive = flat[flat > 0.0]
    if positive.size == 0:
        return list(_DEFAULT_LOG_TICKS)
    low = float(np.percentile(positive, 0.5))
    high = float(np.percentile(positive, 99.5))
    ticks = [tick for tick in _DEFAULT_LOG_TICKS if low * 0.5 <= tick <= high * 2.0]
    if not ticks:
        # Fall back to a dense log locator for unusual ranges.
        return []
    return ticks


def _draw_curves(
    axis: plt.Axes,
    methods: Sequence[str],
    values: dict[str, np.ndarray],
    *,
    highlights: set[str],
    ylabel: str,
    log_scale: bool,
    font_scale: float = 1.0,
) -> None:
    curves = [np.asarray(values[method], dtype=np.float64) for method in methods]
    pixel_count = max(curve.size for curve in curves)
    use_log = log_scale and any(np.any(curve > 0.0) for curve in curves)
    for index, method in enumerate(methods):
        style = _method_style(method, index, highlights, method_count=len(methods))
        curve = curves[index]
        plot_curve = np.maximum(curve, 1e-4) if use_log else curve
        axis.plot(np.arange(plot_curve.size), plot_curve, label=method, **style)

    axis.set_xlabel("Pixel Index (sorted)", fontsize=13 * font_scale)
    axis.set_ylabel(ylabel, fontsize=13 * font_scale)
    axis.set_xlim(0, max(pixel_count - 1, 1))
    axis.grid(True, which="major", alpha=0.35)
    if use_log:
        axis.set_yscale("log")
        ticks = _sample_log_ticks(curves)
        if ticks:
            axis.yaxis.set_major_locator(FixedLocator(ticks))
            axis.yaxis.set_minor_locator(NullLocator())
        else:
            axis.yaxis.set_major_locator(LogLocator(base=10.0))
    axis.set_axisbelow(True)
    axis.tick_params(labelsize=10 * font_scale)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def _panel_grid(sample_count_value: int) -> tuple[int, int]:
    columns = min(_GRID_COLUMNS, sample_count_value)
    rows = math.ceil(sample_count_value / columns)
    return rows, columns


def create_sample_comparison_figure(
    samples: Sequence[SamplePixelSam],
    *,
    methods: Sequence[str],
    highlight_methods: Sequence[str] = (),
    log_scale: bool = True,
    ylabel: str = "SAM (°)",
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Lay out 1-6 samples in a shared grid with one figure-level legend."""

    if not samples:
        raise ValueError("At least one sample is required.")
    if len(samples) > _MAX_SAMPLES:
        raise ValueError(f"At most {_MAX_SAMPLES} samples can be compared in one figure.")
    if not methods:
        raise ValueError("At least one method is required.")

    highlights = {method.casefold() for method in highlight_methods}
    method_list = list(methods)
    rows, columns = _panel_grid(len(samples))

    subplot_w = _PANEL_WIDTH_INCHES
    subplot_h = _PANEL_HEIGHT_INCHES
    fig_w = subplot_w * columns if figsize is None else float(figsize[0])
    legend_columns, legend_height = _legend_layout(method_list, figure_width=fig_w)
    natural_height = subplot_h * rows + legend_height
    fig_h = natural_height if figsize is None else float(figsize[1])
    if figsize is not None:
        legend_height *= fig_h / natural_height
    font_scale = 1.0 if figsize is None else max(0.55, min(1.0, fig_h / natural_height))

    figure = Figure(figsize=(fig_w, fig_h), layout="constrained")
    grid = figure.add_gridspec(rows, columns)

    for panel_index, sample in enumerate(samples):
        axis = figure.add_subplot(grid[panel_index // columns, panel_index % columns])
        _draw_curves(
            axis,
            method_list,
            sample.curves,
            highlights=highlights,
            ylabel=ylabel,
            log_scale=log_scale,
            font_scale=font_scale,
        )

    handles, labels = figure.axes[0].get_legend_handles_labels()
    _add_shared_legend(
        figure,
        handles,
        labels,
        columns=legend_columns,
        font_scale=font_scale,
    )
    return figure


def _save_figure(
    figure: plt.Figure, path: Path, *, output_format: str, dpi: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path, format=output_format, dpi=dpi, bbox_inches="tight", facecolor="white"
    )
    plt.close(figure)


def render_sample_comparison_figure(
    data: SampleComparisonData,
    *,
    output_directory: Path,
    output_format: str,
    dpi: int,
    highlight_methods: Sequence[str] = (),
    log_scale: bool = True,
) -> Path:
    """Render and save the multi-sample sorted-SAM figure; return its path."""

    if dpi < 1:
        raise ValueError("dpi must be a positive integer.")
    normalized_format = _FORMAT_SUFFIXES.get(output_format.casefold())
    if normalized_format is None:
        raise ValueError(f"Unsupported output format: {output_format!r}.")
    output_directory = output_directory.expanduser()
    figure = create_sample_comparison_figure(
        data.samples,
        methods=data.methods,
        highlight_methods=highlight_methods,
        log_scale=log_scale,
    )
    stem = "_".join(
        f"{sample.benchmark}-{sample.image_index}" for sample in data.samples
    )
    path = output_directory / f"pixel_sam_comparison_{stem}.{normalized_format}"
    _save_figure(figure, path, output_format=normalized_format, dpi=dpi)
    return path


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------


class _GuiTaskRunner:
    """Manages background calculation threads for PixelSamGui."""

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
        on_done: Callable[[list[Path], str | None], None],
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
        on_done: Callable[[SampleComparisonData | None, dict[str, object], str | None], None],
    ) -> None:
        if self._busy.is_set():
            raise ValueError("A calculation is already running.")
        self._busy.set()
        self._set_buttons_state("disabled")
        threading.Thread(
            target=self._preview_worker, args=(settings, on_done), daemon=True
        ).start()

    def _export_worker(
        self, settings: dict[str, object], on_done: Callable
    ) -> None:
        try:
            data = load_sample_pixel_sam_data(
                output_root=settings["output_root"],
                dataset_root=settings["dataset_root"],
                selections=settings["selections"],
                methods=settings["methods"],
                clip_to_unit=settings["clip_to_unit"],
            )
            written = [
                render_sample_comparison_figure(
                    data,
                    output_directory=settings["output_directory"],
                    output_format=settings["output_format"],
                    dpi=settings["dpi"],
                    highlight_methods=settings["highlights"],
                    log_scale=settings["log_scale"],
                )
            ]
        except (FileNotFoundError, IndexError, KeyError, OSError, ValueError) as error:
            self._root.after(0, on_done, [], str(error))
            return
        finally:
            self._busy.clear()
            self._root.after(0, self._set_buttons_state, "normal")
        self._root.after(0, on_done, written, None)

    def _preview_worker(
        self, settings: dict[str, object], on_done: Callable
    ) -> None:
        try:
            data = load_sample_pixel_sam_data(
                output_root=settings["output_root"],
                dataset_root=settings["dataset_root"],
                selections=settings["selections"],
                methods=settings["methods"],
                clip_to_unit=settings["clip_to_unit"],
            )
        except (FileNotFoundError, IndexError, KeyError, OSError, ValueError) as error:
            self._root.after(0, on_done, None, settings, str(error))
            return
        finally:
            self._busy.clear()
            self._root.after(0, self._set_buttons_state, "normal")
        self._root.after(0, on_done, data, settings, None)


class PixelSamGui:
    """Tkinter front end for selecting methods, samples, and export options."""

    _PREVIEW_DPI = 100

    def __init__(self, *, output_root: Path, dataset_root: Path) -> None:
        try:
            import tkinter as tk
            from tkinter import filedialog, messagebox, ttk

            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except ImportError as error:  # pragma: no cover - platform-owned dependency
            raise RuntimeError("The pixel-SAM GUI requires Tkinter.") from error

        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.figure_canvas_type = FigureCanvasTkAgg
        self.root = tk.Tk()
        self.root.title("HSMSFusion Pixel SAM Comparison")
        self.root.geometry("1280x960")
        self.root.minsize(1000, 720)
        self.output_root = tk.StringVar(value=str(output_root))
        self.dataset_root = tk.StringVar(value=str(dataset_root))
        self.dataset_name = tk.StringVar()
        self.target_path = tk.StringVar()
        self.figure_directory = tk.StringVar()
        self.export_format = tk.StringVar(value="png")
        self.export_dpi = tk.IntVar(value=300)
        self.log_scale = tk.BooleanVar(value=True)
        self.clip_to_unit = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="Scan an output directory to begin.")
        self.candidates: dict[str, list[tuple[str, Path]]] = {}
        self.method_order: list[str] = []
        self._included_methods: set[str] = set()
        self._highlighted_methods: set[str] = set()
        self._drag_index: int | None = None
        self._dragging_method = False
        self.comparison_selections: list[tuple[str, int]] = []
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
        controls.columnconfigure(4, weight=1)

        self._build_path_row(controls)
        self._build_dataset_row(controls)
        self._build_method_frame(controls)
        self._build_image_and_export(controls)

        preview_frame = self.ttk.LabelFrame(
            self.root,
            text="Sorted per-pixel SAM preview (double-click an image to add it)",
            padding=6,
        )
        preview_frame.pack(fill="both", expand=True, padx=10, pady=(0, 4))
        self.preview_container = self.ttk.Frame(preview_frame)
        self.preview_container.pack(fill="both", expand=True)
        self.preview_container.bind("<Configure>", self._on_preview_container_resize)
        self.preview_placeholder = self.ttk.Label(
            self.preview_container,
            text=(
                f"Add up to {_MAX_SAMPLES} samples (datasets may differ) and click "
                "'Preview SAM' to compare sorted per-pixel SAM curves."
            ),
            anchor="center",
        )
        self.preview_placeholder.pack(fill="both", expand=True)

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
        self.ttk.Label(controls, text="Dataset-scale").grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.dataset_combo = self.ttk.Combobox(
            controls, textvariable=self.dataset_name, state="readonly", width=25
        )
        self.dataset_combo.grid(row=1, column=1, sticky="w", pady=(10, 0))
        self.dataset_combo.bind("<<ComboboxSelected>>", self._on_dataset_changed)
        self.ttk.Label(controls, text="Target HRHS").grid(
            row=1, column=4, sticky="w", padx=(18, 6), pady=(10, 0)
        )
        self.ttk.Entry(controls, textvariable=self.target_path).grid(
            row=1, column=5, sticky="ew", pady=(10, 0)
        )
        self.ttk.Button(controls, text="Browse", command=self._browse_target).grid(
            row=1, column=6, sticky="w", padx=6, pady=(10, 0)
        )
        self.ttk.Button(controls, text="Load images", command=self.refresh_images).grid(
            row=1, column=7, sticky="w", padx=(2, 0), pady=(10, 0)
        )

    def _build_method_frame(self, controls: object) -> None:
        self.method_frame = self.ttk.LabelFrame(
            controls,
            text=(
                "Methods — list order = legend / curve order; "
                "selection = include; drag or Up/Down to reorder"
            ),
            padding=7,
        )
        self.method_frame.grid(row=2, column=0, columnspan=8, sticky="ew", pady=(12, 0))
        self.method_frame.columnconfigure(0, weight=1)
        self.method_frame.columnconfigure(3, weight=1)

        self.ttk.Label(self.method_frame, text="Legend order (selected = plot)").grid(
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
        self.ttk.Button(
            order_buttons, text="Down", width=8, command=self._move_method_down
        ).pack(pady=2)
        self.ttk.Button(
            order_buttons, text="Select all", width=8, command=self._select_all_methods
        ).pack(pady=2)
        self.ttk.Button(
            order_buttons, text="Clear", width=8, command=self._clear_method_selection
        ).pack(pady=2)

        self.ttk.Label(self.method_frame, text="Highlight (solid red line)").grid(
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
            highlight_buttons,
            text="Highlight",
            width=10,
            command=self._highlight_selected_methods,
        ).pack(pady=2)
        self.ttk.Button(
            highlight_buttons,
            text="Unhighlight",
            width=10,
            command=self._unhighlight_selected_methods,
        ).pack(pady=2)
        self.ttk.Button(
            highlight_buttons,
            text="Clear",
            width=10,
            command=self._clear_highlights,
        ).pack(pady=2)

    def _build_image_and_export(self, controls: object) -> None:
        image_frame = self.ttk.LabelFrame(
            controls, text=f"Comparison samples (max {_MAX_SAMPLES}, datasets may differ)", padding=7
        )
        image_frame.grid(row=3, column=0, columnspan=4, sticky="nsew", pady=(12, 0))
        controls.rowconfigure(3, weight=1)
        image_frame.columnconfigure(0, weight=1)

        self.ttk.Label(image_frame, text="Dataset samples").grid(row=0, column=0, sticky="w")
        self.image_list = self.tk.Listbox(
            image_frame, selectmode="extended", height=9, exportselection=False
        )
        self.image_list.grid(row=1, column=0, rowspan=4, sticky="nsew")
        image_scrollbar = self.ttk.Scrollbar(
            image_frame, orient="vertical", command=self.image_list.yview
        )
        image_scrollbar.grid(row=1, column=1, rowspan=4, sticky="ns", padx=(4, 0))
        self.image_list.configure(yscrollcommand=image_scrollbar.set)
        self.image_list.bind(
            "<Double-Button-1>", lambda _event: self.add_selected_to_comparison()
        )

        self.ttk.Button(
            image_frame, text="Add to comparison", command=self.add_selected_to_comparison
        ).grid(row=1, column=2, sticky="ew", padx=10)
        self.ttk.Button(
            image_frame, text="Remove selected", command=self.remove_comparison_selection
        ).grid(row=2, column=2, sticky="ew", padx=10, pady=4)
        self.ttk.Button(
            image_frame, text="Clear comparison", command=self.clear_comparison
        ).grid(row=3, column=2, sticky="ew", padx=10)
        self.ttk.Label(image_frame, text="Double-click adds one sample.").grid(
            row=4, column=2, sticky="sw", padx=10
        )

        self.ttk.Label(image_frame, text="Comparison order").grid(
            row=0, column=3, columnspan=2, sticky="w"
        )
        self.comparison_list = self.tk.Listbox(
            image_frame, selectmode="extended", height=9, exportselection=False, width=28
        )
        self.comparison_list.grid(row=1, column=3, rowspan=4, columnspan=2, sticky="nsew")

        export_frame = self.ttk.LabelFrame(controls, text="Export", padding=7)
        export_frame.grid(row=3, column=4, columnspan=4, sticky="nsew", padx=(12, 0), pady=(12, 0))
        export_frame.columnconfigure(1, weight=1)
        self.ttk.Label(export_frame, text="Figure directory").grid(row=0, column=0, sticky="w")
        self.ttk.Entry(export_frame, textvariable=self.figure_directory).grid(
            row=0, column=1, sticky="ew", padx=6
        )
        self.ttk.Button(export_frame, text="Browse", command=self._browse_figure_directory).grid(
            row=0, column=2, sticky="w"
        )
        self.ttk.Checkbutton(
            export_frame, text="Log Y axis", variable=self.log_scale
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.ttk.Checkbutton(
            export_frame, text="Clip values to [0, 1]", variable=self.clip_to_unit
        ).grid(row=1, column=2, columnspan=2, sticky="w", pady=(8, 0))
        self.ttk.Label(export_frame, text="Format").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.ttk.Combobox(
            export_frame,
            textvariable=self.export_format,
            values=("png", "jpeg", "pdf", "svg"),
            state="readonly",
            width=8,
        ).grid(row=2, column=1, sticky="w", padx=6, pady=(8, 0))
        self.ttk.Label(export_frame, text="DPI").grid(
            row=2, column=2, sticky="w", padx=(10, 4), pady=(8, 0)
        )
        self.ttk.Spinbox(
            export_frame, from_=1, to=2400, textvariable=self.export_dpi, width=8
        ).grid(row=2, column=3, sticky="w", pady=(8, 0))
        self.preview_button = self.ttk.Button(
            export_frame, text="Preview SAM", command=self.start_preview
        )
        self.preview_button.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(16, 0))
        self.generate_button = self.ttk.Button(
            export_frame, text="Calculate and export", command=self.start_export
        )
        self.generate_button.grid(
            row=3, column=2, columnspan=2, sticky="ew", padx=(8, 0), pady=(16, 0)
        )

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
            self._on_dataset_changed()

    def _browse_target(self) -> None:
        selected = self.filedialog.askopenfilename(
            title="Select test HDF5 target",
            initialdir=str(Path(self.target_path.get() or self.dataset_root.get()).parent),
            filetypes=(("HDF5", "*.h5 *.hdf5"), ("All files", "*.*")),
        )
        if selected:
            self.target_path.set(selected)
            self.refresh_images()

    def _browse_figure_directory(self) -> None:
        selected = self.filedialog.askdirectory(
            title="Select figure output directory",
            initialdir=self.figure_directory.get() or self.output_root.get(),
        )
        if selected:
            self.figure_directory.set(selected)

    def scan_outputs(self) -> None:
        self.candidates = discover_output_predictions(Path(self.output_root.get()))
        dataset_names = list(self.candidates)
        self.dataset_combo.configure(values=dataset_names)
        if not dataset_names:
            self.dataset_name.set("")
            self._rebuild_method_checkboxes([])
            self._replace_image_choices([])
            self.status.set("No predictions.h5 files were found under the output root.")
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
        target = infer_target_path(Path(self.dataset_root.get()), benchmark)
        self.target_path.set(str(target))
        self.figure_directory.set(str(Path(self.output_root.get()) / benchmark / "pixel_sam"))
        self.refresh_images()

    def _rebuild_method_checkboxes(self, candidates: Sequence[tuple[str, Path]]) -> None:
        """Refresh method listboxes, keeping order/include/highlight when possible."""

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
            self._included_methods = {
                name for name in self._included_methods if name in known
            }
            self._included_methods |= set(appended)
        self._highlighted_methods = {
            name for name in self._highlighted_methods if name in known
        }
        self._refresh_method_listboxes()

    def _refresh_method_listboxes(self) -> None:
        """Rewrite both listboxes from method_order and the state sets."""

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

    def _replace_image_choices(self, image_indices: Sequence[int]) -> None:
        self.image_list.delete(0, self.tk.END)
        for image_index in image_indices:
            self.image_list.insert(self.tk.END, f"Image {image_index:04d}")

    def refresh_images(self) -> None:
        self._replace_image_choices([])
        target = Path(self.target_path.get()).expanduser()
        try:
            count = sample_count(target)
        except (FileNotFoundError, KeyError, OSError, ValueError) as error:
            self.status.set(f"Cannot load test images: {error}")
            return
        self._replace_image_choices(range(count))
        self.status.set(
            f"Loaded {count} test images. Double-click images (or use 'Add to comparison') "
            f"to compare up to {_MAX_SAMPLES} samples."
        )

    def add_selected_to_comparison(self) -> None:
        benchmark = self.dataset_name.get()
        if not benchmark:
            self.status.set("Choose a dataset-scale output folder first.")
            return
        added = 0
        for index in self.image_list.curselection():
            if len(self.comparison_selections) >= _MAX_SAMPLES:
                self.status.set(f"At most {_MAX_SAMPLES} samples can be compared at once.")
                break
            entry = (benchmark, int(index))
            if entry in self.comparison_selections:
                continue
            self.comparison_selections.append(entry)
            added += 1
        self._refresh_comparison_list()
        if added:
            self.status.set(
                f"Comparison now holds {len(self.comparison_selections)} sample(s)."
            )

    def remove_comparison_selection(self) -> None:
        for position in sorted(self.comparison_list.curselection(), reverse=True):
            del self.comparison_selections[position]
        self._refresh_comparison_list()

    def clear_comparison(self) -> None:
        self.comparison_selections.clear()
        self._refresh_comparison_list()

    def _refresh_comparison_list(self) -> None:
        self.comparison_list.delete(0, self.tk.END)
        for benchmark, index in self.comparison_selections:
            self.comparison_list.insert(self.tk.END, f"{benchmark}  #{index:04d}")

    def _comparison_methods(self) -> list[str]:
        """Included methods in current listbox order, restricted to common names."""

        self._capture_method_selection()
        involved = {benchmark for benchmark, _ in self.comparison_selections}
        common: set[str] | None = None
        for benchmark in involved:
            names = {method.casefold() for method, _ in self.candidates.get(benchmark, [])}
            common = names if common is None else common & names
        if not common:
            return []
        return [
            name
            for name in self.method_order
            if name in self._included_methods and name.casefold() in common
        ]

    def _export_settings(self) -> dict[str, object]:
        if self._task_runner.is_busy:
            raise ValueError("A calculation is already running.")
        if not self.comparison_selections:
            raise ValueError(
                "Add at least one sample to the comparison (double-click an image or use "
                "'Add to comparison')."
            )
        methods = self._comparison_methods()
        if not methods:
            raise ValueError("Select at least one method available in every compared dataset.")
        try:
            dpi = int(self.export_dpi.get())
        except (self.tk.TclError, TypeError, ValueError) as error:
            raise ValueError("DPI must be a positive integer.") from error
        if dpi < 1:
            raise ValueError("DPI must be a positive integer.")
        figure_directory = Path(self.figure_directory.get()).expanduser()
        if not str(figure_directory):
            raise ValueError("Choose a figure output directory.")
        highlights = [name for name in methods if name in self._highlighted_methods]
        return {
            "output_root": Path(self.output_root.get()),
            "dataset_root": Path(self.dataset_root.get()),
            "selections": list(self.comparison_selections),
            "methods": methods,
            "output_directory": figure_directory,
            "output_format": self.export_format.get(),
            "dpi": dpi,
            "highlights": highlights,
            "log_scale": bool(self.log_scale.get()),
            "clip_to_unit": bool(self.clip_to_unit.get()),
        }

    def start_export(self) -> None:
        try:
            settings = self._export_settings()
        except ValueError as error:
            self.messagebox.showerror("Invalid selection", str(error))
            return
        self.status.set("Calculating selected sorted per-pixel SAM curves...")
        self._task_runner.run_export(settings, self._on_export_done)

    def start_preview(self) -> None:
        try:
            settings = self._export_settings()
        except ValueError as error:
            self.messagebox.showerror("Invalid selection", str(error))
            return
        self.status.set("Calculating per-pixel SAM preview...")
        self._task_runner.run_preview(settings, self._on_preview_done)

    def _on_preview_done(
        self,
        data: SampleComparisonData | None,
        settings: dict[str, object],
        error: str | None,
    ) -> None:
        if error is not None:
            self.status.set(f"Preview failed: {error}")
            self.messagebox.showerror("Preview failed", error)
            return
        assert data is not None
        figure = create_sample_comparison_figure(
            data.samples,
            methods=data.methods,
            highlight_methods=settings["highlights"],
            log_scale=settings["log_scale"],
            figsize=self._preview_figsize(),
        )
        self._replace_preview(figure)
        self.status.set(
            f"Previewed {len(data.samples)} sample(s) using {len(data.methods)} method(s)."
        )

    def _preview_figsize(self) -> tuple[float, float]:
        self.preview_container.update_idletasks()
        width = max(self.preview_container.winfo_width(), 640)
        height = max(self.preview_container.winfo_height(), 360)
        return (width / self._PREVIEW_DPI, height / self._PREVIEW_DPI)

    def _on_preview_container_resize(self, _event: object | None = None) -> None:
        if self.preview_canvas is None or self.preview_figure is None:
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

    def _on_export_done(self, written: list[Path], error: str | None) -> None:
        if error is not None:
            self.status.set(f"Export failed: {error}")
            self.messagebox.showerror("Export failed", error)
            return
        paths = ", ".join(str(path) for path in written)
        self.status.set(f"Exported {len(written)} file(s): {paths}")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def launch_gui(*, output_root: Path, dataset_root: Path) -> None:
    """Launch the interactive desktop front end."""

    matplotlib.use("TkAgg")
    PixelSamGui(output_root=output_root, dataset_root=dataset_root).run()


def main(argv: Sequence[str] | None = None) -> int:
    """Launch the GUI, or render a headless comparison when CLI flags are given."""

    parser = argparse.ArgumentParser(
        description="HSMSFusion sorted per-pixel SAM comparison tool"
    )
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--dataset-root", type=Path, default=Path("Dataset"))
    parser.add_argument("--benchmark", help="Dataset-scale folder name, e.g. chikusei_x4")
    parser.add_argument("--method", action="append", dest="methods", help="Method name (repeatable)")
    parser.add_argument(
        "--selection",
        action="append",
        dest="selections",
        metavar="BENCHMARK:INDEX",
        help="Sample selection such as chikusei_x4:0 (repeatable, max 6)",
    )
    parser.add_argument(
        "--highlight", action="append", dest="highlights", help="Highlighted method name"
    )
    parser.add_argument("--output-directory", type=Path, default=Path("figures/pixel_sam"))
    parser.add_argument("--format", default="png", choices=sorted(_FORMAT_SUFFIXES))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--linear-y", action="store_true", help="Use a linear Y axis")
    parser.add_argument("--no-clip", action="store_true", help="Do not clip values to [0, 1]")
    parser.add_argument("--gui", action="store_true", help="Force the interactive GUI")
    parser.add_argument(
        "--render-only", action="store_true",
        help="Skip the GUI even when selections are incomplete (headless check)",
    )

    arguments = parser.parse_args(argv)

    selections: list[tuple[str, int]] = []
    for item in arguments.selections or []:
        if ":" not in item:
            raise SystemExit(f"Invalid --selection {item!r}; expected BENCHMARK:INDEX.")
        name, raw_index = item.rsplit(":", maxsplit=1)
        try:
            selections.append((name.strip(), int(raw_index)))
        except ValueError as error:
            raise SystemExit(f"Invalid --selection index in {item!r}.") from error

    use_gui = arguments.gui or (not selections and arguments.benchmark is None)
    if use_gui and not arguments.render_only:
        launch_gui(
            output_root=arguments.output_root, dataset_root=arguments.dataset_root
        )
        return 0

    if not selections:
        if arguments.benchmark is None:
            raise SystemExit("Provide --benchmark and --selection, or run with --gui.")
        # Headless fallback: first sample of the chosen benchmark.
        selections = [(arguments.benchmark, 0)]

    try:
        data = load_sample_pixel_sam_data(
            output_root=arguments.output_root,
            dataset_root=arguments.dataset_root,
            selections=selections,
            methods=arguments.methods,
            clip_to_unit=not arguments.no_clip,
        )
        path = render_sample_comparison_figure(
            data,
            output_directory=arguments.output_directory,
            output_format=arguments.format,
            dpi=arguments.dpi,
            highlight_methods=arguments.highlights or (),
            log_scale=not arguments.linear_y,
        )
    except (FileNotFoundError, IndexError, KeyError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
