"""Plot per-band PSNR curves from exported HDF5 predictions.

The toolbox test export uses the following layout::

    output/<dataset>_x<scale>/<method>/predictions.h5  (/prediction, NCHW)
    Dataset/<dataset>/test.h5                           (/HRHS, NCHW)

Run the script to launch a GUI where up to 3 samples - possibly from
different datasets - can be compared side by side.  The GUI scans available
output folders; only the selected samples are loaded into memory.
"""

from __future__ import annotations

import csv
import math
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

# Allow ``python scripts/plot_band_psnr.py`` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from scripts._shared import (
    discover_output_predictions,
    infer_target_path,
    load_target_cube,
    resolve_benchmark,
    sample_count,
    select_predictions,
    setup_matplotlib_font,
    validate_hsi_dataset,
)

if TYPE_CHECKING:
    from tkinter import ttk

setup_matplotlib_font()

# Backward-compatible alias for tests
target_sample_count = sample_count

_PSNR_EPSILON = 1e-8
_FORMAT_SUFFIXES = {"png": "png", "jpg": "jpeg", "jpeg": "jpeg", "pdf": "pdf", "svg": "svg"}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BandPsnrData:
    """Per-image per-band PSNR values for all selected methods."""

    benchmark: str
    target_path: Path
    methods: list[str]
    per_image: dict[str, np.ndarray]
    image_indices: list[int]

    @property
    def sample_count(self) -> int:
        return len(self.image_indices)

    @property
    def band_count(self) -> int:
        return int(next(iter(self.per_image.values())).shape[1])


@dataclass(frozen=True)
class SampleBandPsnr:
    """Per-band PSNR curves of one sample for every compared method."""

    benchmark: str
    image_index: int
    curves: dict[str, np.ndarray]

    @property
    def band_count(self) -> int:
        return int(next(iter(self.curves.values())).shape[0])


@dataclass(frozen=True)
class SampleComparisonData:
    """Per-band PSNR of up to 3 samples that may come from different benchmarks."""

    methods: list[str]
    samples: list[SampleBandPsnr]
    benchmark_data: dict[str, BandPsnrData]


# ---------------------------------------------------------------------------
# PSNR computation
# ---------------------------------------------------------------------------


def _per_band_psnr(prediction: np.ndarray, target: np.ndarray, *, epsilon: float) -> np.ndarray:
    """Compute per-band PSNR between prediction and target.

    Both arrays are clipped to [0, 1] before MSE computation.
    Returns shape (C,) float32 PSNR values in dB.
    """
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction shape {prediction.shape} does not match target shape {target.shape}.")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Prediction and target values must be finite.")
    clipped_prediction = np.clip(prediction, 0.0, 1.0)
    clipped_target = np.clip(target, 0.0, 1.0)
    mse = np.mean((clipped_prediction - clipped_target) ** 2, axis=(-2, -1), dtype=np.float64)
    return (-10.0 * np.log10(np.maximum(mse, epsilon))).astype(np.float32)


# ---------------------------------------------------------------------------
# Data loading (target cached in memory)
# ---------------------------------------------------------------------------


def load_band_psnr_data(
    *,
    output_root: Path,
    dataset_root: Path,
    benchmark: str | None,
    methods: Sequence[str] | None = None,
    target_path: Path | None = None,
    target_key: str = "HRHS",
    prediction_key: str = "prediction",
    epsilon: float = _PSNR_EPSILON,
    image_indices: Sequence[int] | None = None,
) -> BandPsnrData:
    """Read selected exports and calculate per-image PSNR for every spectral band.

    The target cube is loaded into memory once and reused for every method.
    """

    if epsilon <= 0.0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be a finite positive value.")

    benchmark_name, candidates = resolve_benchmark(
        discover_output_predictions(output_root), benchmark
    )
    selected = select_predictions(candidates, methods)
    if not selected:
        raise ValueError("At least one method must be selected.")

    resolved_target_path = (target_path or infer_target_path(dataset_root, benchmark_name)).expanduser()

    # Only read the shape (N) without loading data into memory
    total_sample_count = sample_count(resolved_target_path, dataset_key=target_key)

    if image_indices is None:
        selected_indices = list(range(total_sample_count))
    else:
        selected_indices = list(image_indices)
        if not selected_indices:
            raise ValueError("image_indices must contain at least one image index.")
        if len(set(selected_indices)) != len(selected_indices):
            raise ValueError("image_indices must not contain duplicates.")
        invalid = [i for i in selected_indices if not 0 <= i < total_sample_count]
        if invalid:
            raise IndexError(
                f"image index {invalid[0]} is outside [0, {total_sample_count - 1}] "
                f"for {benchmark_name}."
            )

    # Load only the selected subset of the target cube
    cached_target = load_target_cube(
        resolved_target_path, dataset_key=target_key, image_indices=selected_indices
    )
    band_count = cached_target.band_count

    per_image: dict[str, np.ndarray] = {}
    for method_name, prediction_path in selected:
        with h5py.File(prediction_path, "r") as prediction_handle:
            if prediction_key not in prediction_handle:
                raise KeyError(f"{prediction_path} does not contain {prediction_key!r}.")
            prediction_dataset = prediction_handle[prediction_key]
            prediction_shape = validate_hsi_dataset(
                prediction_dataset, path=prediction_path, key=prediction_key
            )
            # Compare C, H, W dimensions (N differs because we load a subset)
            if prediction_shape[1:] != cached_target.data.shape[1:]:
                raise ValueError(
                    f"{method_name} prediction shape {prediction_shape} does not match "
                    f"target shape (N={len(selected_indices)}, *{cached_target.data.shape[1:]})."
                )
            if prediction_shape[0] < total_sample_count:
                raise IndexError(
                    f"{method_name} prediction has {prediction_shape[0]} samples "
                    f"but target has {total_sample_count}."
                )
            values = np.empty((len(selected_indices), band_count), dtype=np.float32)
            for row_index, image_index in enumerate(selected_indices):
                prediction_sample = np.asarray(prediction_dataset[image_index], dtype=np.float32)
                target_sample = cached_target.data[row_index]
                values[row_index] = _per_band_psnr(
                    prediction_sample, target_sample, epsilon=epsilon
                )
            per_image[method_name] = values

    return BandPsnrData(
        benchmark=benchmark_name,
        target_path=resolved_target_path,
        methods=[name for name, _ in selected],
        per_image=per_image,
        image_indices=selected_indices,
    )


def load_sample_band_psnr_data(
    *,
    output_root: Path,
    dataset_root: Path,
    selections: Sequence[tuple[str, int]],
    methods: Sequence[str] | None = None,
) -> SampleComparisonData:
    """Load per-band PSNR for (benchmark, image) selections across benchmarks.

    Only the selected samples are read.  The comparison keeps the methods
    available in every selected benchmark, matched case-insensitively, so all
    panels share one legend.
    """
    if not selections:
        raise ValueError("At least one sample selection is required.")
    if len(set(selections)) != len(selections):
        raise ValueError("Sample selections must not contain duplicates.")

    benchmarks = sorted({benchmark for benchmark, _ in selections})
    all_candidates = discover_output_predictions(output_root)
    benchmark_data: dict[str, BandPsnrData] = {}
    for benchmark in benchmarks:
        indices = sorted({index for name, index in selections if name == benchmark})
        wanted = None
        if methods is not None:
            # Skip methods this benchmark does not offer; the intersection
            # below decides what the comparison keeps.
            available = {name.casefold() for name, _ in all_candidates.get(benchmark, [])}
            wanted = [name for name in methods if name.casefold() in available]
            if not wanted:
                raise ValueError("The compared datasets share no fusion method.")
        benchmark_data[benchmark] = load_band_psnr_data(
            output_root=output_root,
            dataset_root=dataset_root,
            benchmark=benchmark,
            methods=wanted,
            image_indices=indices,
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
            SampleBandPsnr(
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


def _method_style(method: str, index: int, highlights: set[str], *, method_count: int) -> dict[str, object]:
    if method.casefold() in highlights:
        return {"color": "#e74c3c", "linewidth": 2.5, "linestyle": "-", "zorder": 10}
    if method_count <= 20:
        colors = plt.get_cmap("tab20").colors
        color = colors[index % len(colors)]
        linestyle = "--"
    else:
        cmap = plt.get_cmap("viridis")
        color = cmap(index / max(method_count - 1, 1))
        # Cycle dash patterns so neighbouring viridis colours stay distinguishable.
        linestyle = ("--", "-.", ":")[index % 3]
    return {
        "color": color,
        "linewidth": 1.25,
        "linestyle": linestyle,
        "zorder": 1,
    }


_LEGEND_HANDLE_INCHES = 0.6  # line sample + padding in front of each label
_LEGEND_CHAR_INCHES = 0.085  # rough glyph width at legend fontsize 10


def _legend_layout(labels: Sequence[str], *, figure_width: float) -> tuple[int, float]:
    """Pick legend columns that fit the figure width; return (columns, row height)."""

    longest = max((len(label) for label in labels), default=0)
    column_width = _LEGEND_HANDLE_INCHES + _LEGEND_CHAR_INCHES * longest
    columns = max(1, min(8, len(labels), int(figure_width // column_width)))
    rows = math.ceil(len(labels) / columns)
    return columns, 0.45 + 0.25 * rows


def _add_shared_legend(
    figure: Figure,
    handles: Sequence[object],
    labels: Sequence[str],
    *,
    columns: int,
    font_scale: float = 1.0,
) -> None:
    """Draw the shared method legend centred above all panels.

    Requires a constrained layout engine, which reserves the legend space and
    keeps fixed-aspect (4:3) panels from colliding with their neighbours.
    """

    figure.legend(
        handles, labels,
        loc="outside upper center",
        ncol=columns, fontsize=10 * font_scale, frameon=False,
    )


def _draw_curves(
    axis: plt.Axes,
    methods: Sequence[str],
    values: dict[str, np.ndarray],
    *,
    highlights: set[str],
    ylabel: str,
    font_scale: float = 1.0,
) -> None:
    band_count = max(len(values[method]) for method in methods)
    for index, method in enumerate(methods):
        style = _method_style(method, index, highlights, method_count=len(methods))
        curve = np.asarray(values[method])
        axis.plot(np.arange(1, curve.size + 1), curve, label=method, **style)
    axis.set_xlabel("Band Index", fontsize=14 * font_scale)
    axis.set_ylabel(ylabel, fontsize=14 * font_scale)
    # Half-band padding keeps the first/last markers fully inside the axes.
    axis.set_xlim(0.5, band_count + 0.5)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    # Enforce an exact 4:3 plot box (width:height) on every curve panel.
    # (Keep matplotlib's default 'datalim' adjustable: 'box' breaks with
    # aspect='auto' on matplotlib 3.11, and explicit x limits are preserved.)
    axis.set_box_aspect(3 / 4)
    axis.grid(True, alpha=0.3)
    axis.set_axisbelow(True)
    axis.tick_params(labelsize=11 * font_scale)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def _save_figure(figure: plt.Figure, path: Path, *, output_format: str, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format=output_format, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def create_sample_comparison_figure(
    samples: Sequence[SampleBandPsnr],
    *,
    methods: Sequence[str],
    highlight_methods: Sequence[str] = (),
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Lay out 1-3 samples side by side in one row, tagged (a)/(b)/(c).

    Samples may come from different benchmarks; every panel keeps an exact
    4:3 plot box, the shared method legend is placed above all panels and a
    ``(a)``/``(b)``/``(c)`` tag sits below each panel.  Pass ``figsize`` to
    override the natural figure size (the GUI uses this to fit its
    container); panel and legend proportions are preserved.
    """
    if not samples:
        raise ValueError("At least one sample is required.")
    if len(samples) > 3:
        raise ValueError("At most 3 samples can be compared in one figure.")
    if not methods:
        raise ValueError("At least one method is required.")
    highlights = {method.casefold() for method in highlight_methods}
    method_list = list(methods)

    ncols = len(samples)
    # Each subplot cell is 4:3 → width:height = 6.0:4.5 inches.
    subplot_w, subplot_h = 6.0, 4.5
    fig_w = subplot_w * ncols if figsize is None else float(figsize[0])
    columns, legend_height = _legend_layout(method_list, figure_width=fig_w)
    natural_height = subplot_h + legend_height
    fig_h = natural_height if figsize is None else float(figsize[1])
    if figsize is not None:
        legend_height *= fig_h / natural_height
    # Shrink text with the overridden figure so labels never outgrow panels.
    font_scale = 1.0 if figsize is None else max(0.6, min(1.0, fig_h / natural_height))

    figure = Figure(figsize=(fig_w, fig_h), layout="constrained")
    gs = figure.add_gridspec(1, ncols)

    for panel_index, sample in enumerate(samples):
        axis = figure.add_subplot(gs[0, panel_index])
        _draw_curves(
            axis, method_list, sample.curves,
            highlights=highlights, ylabel="PSNR (dB)", font_scale=font_scale,
        )
        tag = "abc"[panel_index]
        axis.text(
            0.5, -0.24, f"({tag})",
            transform=axis.transAxes, ha="center", va="top",
            fontsize=13 * font_scale,
        )

    handles, labels = figure.axes[0].get_legend_handles_labels()
    _add_shared_legend(
        figure, handles, labels,
        columns=columns,
        font_scale=font_scale,
    )
    return figure


def render_sample_comparison_figure(
    data: SampleComparisonData,
    *,
    output_directory: Path,
    output_format: str,
    dpi: int,
    highlight_methods: Sequence[str] = (),
) -> Path:
    """Render and save the multi-sample comparison figure; return its path."""

    if dpi < 1:
        raise ValueError("dpi must be a positive integer.")
    normalized_format = _FORMAT_SUFFIXES.get(output_format.casefold())
    if normalized_format is None:
        raise ValueError(f"Unsupported output format: {output_format!r}.")
    output_directory = output_directory.expanduser()
    figure = create_sample_comparison_figure(
        data.samples, methods=data.methods, highlight_methods=highlight_methods,
    )
    stem = "_".join(f"{sample.benchmark}-{sample.image_index}" for sample in data.samples)
    path = output_directory / f"band_psnr_comparison_{stem}.{normalized_format}"
    _save_figure(figure, path, output_format=normalized_format, dpi=dpi)
    return path


def create_band_psnr_preview_figure(
    data: BandPsnrData,
    *,
    image_indices: Sequence[int],
    highlight_methods: Sequence[str] = (),
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Single-benchmark convenience wrapper around the comparison figure."""

    if not image_indices:
        raise ValueError("At least one image index is required.")
    for idx in image_indices:
        if idx not in data.image_indices:
            available = ", ".join(str(i) for i in data.image_indices)
            raise IndexError(f"image index {idx} was not calculated. Available: {available}.")
    samples = [
        SampleBandPsnr(
            benchmark=data.benchmark,
            image_index=idx,
            curves={
                method: data.per_image[method][data.image_indices.index(idx)]
                for method in data.methods
            },
        )
        for idx in image_indices
    ]
    return create_sample_comparison_figure(
        samples, methods=data.methods, highlight_methods=highlight_methods, figsize=figsize,
    )


def write_per_band_csv(data: BandPsnrData, path: Path) -> Path:
    """Write unrounded per-image per-band values for later statistical analysis."""

    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[list[str]] = []
    for method in data.methods:
        for image_index, psnr_values in zip(data.image_indices, data.per_image[method], strict=True):
            for band_index, value in enumerate(psnr_values, start=1):
                rows.append([method, image_index, band_index, f"{float(value):.6f}"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Method", "ImageIndex", "Band", "PSNR"])
        writer.writerows(rows)
    return path


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------


class _GuiTaskRunner:
    """Manages background calculation threads for BandPsnrGui."""

    def __init__(self, root: object) -> None:
        self._root = root
        self._busy = threading.Event()
        self._buttons: list[object] = []

    @property
    def is_busy(self) -> bool:
        return self._busy.is_set()

    def register_button(self, button: ttk.Button) -> None:
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
        on_done: Callable[[BandPsnrData | None, dict[str, object], str | None], None],
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
            data = load_sample_band_psnr_data(
                output_root=settings["output_root"],
                dataset_root=settings["dataset_root"],
                selections=settings["selections"],
                methods=settings["methods"],
            )
            written = [
                render_sample_comparison_figure(
                    data,
                    output_directory=settings["output_directory"],
                    output_format=settings["output_format"],
                    dpi=settings["dpi"],
                    highlight_methods=settings["highlights"],
                )
            ]
            if settings["save_csv"]:
                for benchmark_data in data.benchmark_data.values():
                    written.append(
                        write_per_band_csv(
                            benchmark_data,
                            settings["output_directory"]
                            / f"{benchmark_data.benchmark}_per_band_psnr.csv",
                        )
                    )
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
            data = load_sample_band_psnr_data(
                output_root=settings["output_root"],
                dataset_root=settings["dataset_root"],
                selections=settings["selections"],
                methods=settings["methods"],
            )
        except (FileNotFoundError, IndexError, KeyError, OSError, ValueError) as error:
            self._root.after(0, on_done, None, settings, str(error))
            return
        finally:
            self._busy.clear()
            self._root.after(0, self._set_buttons_state, "normal")
        self._root.after(0, on_done, data, settings, None)


class BandPsnrGui:
    """Tkinter front end for selecting benchmark methods and test images."""

    _PREVIEW_DPI = 100

    def __init__(self, *, output_root: Path, dataset_root: Path) -> None:
        try:
            import tkinter as tk
            from tkinter import filedialog, messagebox, ttk

            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except ImportError as error:  # pragma: no cover - platform-owned dependency
            raise RuntimeError("The per-band PSNR GUI requires Tkinter.") from error

        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.figure_canvas_type = FigureCanvasTkAgg
        self.root = tk.Tk()
        self.root.title("HSMSFusion Per-band PSNR")
        self.root.geometry("1120x960")
        self.root.minsize(900, 720)
        self.output_root = tk.StringVar(value=str(output_root))
        self.dataset_root = tk.StringVar(value=str(dataset_root))
        self.dataset_name = tk.StringVar()
        self.target_path = tk.StringVar()
        self.figure_directory = tk.StringVar()
        self.export_format = tk.StringVar(value="png")
        self.export_dpi = tk.IntVar(value=300)
        self.save_csv = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Scan an output directory to begin.")
        self.candidates: dict[str, list[tuple[str, Path]]] = {}
        self.method_variables: dict[str, object] = {}
        self.highlight_variables: dict[str, object] = {}
        self._method_state: dict[str, bool] = {}
        self._highlight_state: dict[str, bool] = {}
        self.comparison_selections: list[tuple[str, int]] = []
        self._task_runner = _GuiTaskRunner(self.root)
        self.preview_canvas = None
        self.preview_figure: Figure | None = None
        self.preview_placeholder: object | None = None

        self._build_layout()
        self._task_runner.register_button(self.generate_button)
        self._task_runner.register_button(self.preview_button)
        self.scan_outputs()

    # -- Layout split into focused methods ----------------------------------

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
            text="Per-band PSNR preview (double-click an image to add it to the comparison)",
            padding=6,
        )
        preview_frame.pack(fill="both", expand=True, padx=10, pady=(0, 4))
        self.preview_container = self.ttk.Frame(preview_frame)
        self.preview_container.pack(fill="both", expand=True)
        self.preview_container.bind("<Configure>", self._on_preview_container_resize)
        self.preview_placeholder = self.ttk.Label(
            self.preview_container,
            text="Add up to 3 samples (datasets may differ) and click 'Preview PSNR' "
                 "to compare their per-band PSNR (4:3 panels).",
            anchor="center",
        )
        self.preview_placeholder.pack(fill="both", expand=True)

        self.ttk.Label(self.root, textvariable=self.status, anchor="w", padding=(12, 7)).pack(fill="x")

    def _build_path_row(self, controls: object) -> None:
        self.ttk.Label(controls, text="Output root").grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.ttk.Entry(controls, textvariable=self.output_root).grid(row=0, column=1, sticky="ew")
        self.ttk.Button(controls, text="Browse", command=self._browse_output_root).grid(
            row=0, column=2, sticky="w", padx=6
        )
        self.ttk.Button(controls, text="Scan", command=self.scan_outputs).grid(row=0, column=3, sticky="w")
        self.ttk.Label(controls, text="Dataset root").grid(row=0, column=4, sticky="w", padx=(18, 6))
        self.ttk.Entry(controls, textvariable=self.dataset_root).grid(row=0, column=5, sticky="ew")
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
            controls, text="Methods (use the right checkbox to highlight)", padding=7
        )
        self.method_frame.grid(row=2, column=0, columnspan=8, sticky="ew", pady=(12, 0))

    def _build_image_and_export(self, controls: object) -> None:
        image_frame = self.ttk.LabelFrame(
            controls, text="Comparison samples (max 3, datasets may differ)", padding=7
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
        self.image_list.bind("<Double-Button-1>", lambda _event: self.add_selected_to_comparison())

        self.ttk.Button(
            image_frame, text="Add to comparison", command=self.add_selected_to_comparison
        ).grid(row=1, column=2, sticky="ew", padx=10)
        self.ttk.Button(
            image_frame, text="Remove selected", command=self.remove_comparison_selection
        ).grid(row=2, column=2, sticky="ew", padx=10, pady=4)
        self.ttk.Button(
            image_frame, text="Clear comparison", command=self.clear_comparison
        ).grid(row=3, column=2, sticky="ew", padx=10)
        self.ttk.Label(
            image_frame, text="Double-click adds one sample."
        ).grid(row=4, column=2, sticky="sw", padx=10)

        self.ttk.Label(image_frame, text="Comparison (a)(b)(c)").grid(
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
        self.ttk.Checkbutton(export_frame, text="Save PSNR CSV", variable=self.save_csv).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        self.ttk.Label(export_frame, text="Format").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.ttk.Combobox(
            export_frame, textvariable=self.export_format,
            values=("png", "jpeg", "pdf", "svg"), state="readonly", width=8,
        ).grid(row=2, column=1, sticky="w", padx=6, pady=(8, 0))
        self.ttk.Label(export_frame, text="DPI").grid(row=2, column=2, sticky="w", padx=(10, 4), pady=(8, 0))
        self.ttk.Spinbox(export_frame, from_=1, to=2400, textvariable=self.export_dpi, width=8).grid(
            row=2, column=3, sticky="w", pady=(8, 0)
        )
        self.preview_button = self.ttk.Button(
            export_frame, text="Preview PSNR", command=self.start_preview
        )
        self.preview_button.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(16, 0))
        self.generate_button = self.ttk.Button(
            export_frame, text="Calculate and export", command=self.start_export
        )
        self.generate_button.grid(row=3, column=2, columnspan=2, sticky="ew", padx=(8, 0), pady=(16, 0))

    # -- Logic --------------------------------------------------------------

    def run(self) -> None:
        self.root.mainloop()

    def _browse_output_root(self) -> None:
        selected = self.filedialog.askdirectory(title="Select output root", initialdir=self.output_root.get())
        if selected:
            self.output_root.set(selected)
            self.scan_outputs()

    def _browse_dataset_root(self) -> None:
        selected = self.filedialog.askdirectory(title="Select Dataset root", initialdir=self.dataset_root.get())
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
        self.figure_directory.set(str(Path(self.output_root.get()) / benchmark / "band_psnr"))
        self.refresh_images()

    def _rebuild_method_checkboxes(self, candidates: Sequence[tuple[str, Path]]) -> None:
        # Persist the checkbox states so they survive switching datasets.
        for method, variable in self.method_variables.items():
            self._method_state[method] = bool(variable.get())
        for method, variable in self.highlight_variables.items():
            self._highlight_state[method] = bool(variable.get())
        for child in self.method_frame.winfo_children():
            child.destroy()
        self.method_variables.clear()
        self.highlight_variables.clear()
        if not candidates:
            self.ttk.Label(self.method_frame, text="No method predictions are available.").grid(
                row=0, column=0, sticky="w"
            )
            return
        self.ttk.Label(self.method_frame, text="Method").grid(row=0, column=0, sticky="w", padx=(0, 16))
        self.ttk.Label(self.method_frame, text="Highlight").grid(row=0, column=1, sticky="w")
        for row, (method, _path) in enumerate(candidates, start=1):
            selected = self.tk.BooleanVar(value=self._method_state.get(method, True))
            highlighted = self.tk.BooleanVar(value=self._highlight_state.get(method, False))
            self.method_variables[method] = selected
            self.highlight_variables[method] = highlighted
            self.ttk.Checkbutton(self.method_frame, text=method, variable=selected).grid(
                row=row, column=0, sticky="w"
            )
            self.ttk.Checkbutton(self.method_frame, variable=highlighted).grid(
                row=row, column=1, sticky="w"
            )

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
            f"to compare up to 3 samples."
        )

    def add_selected_to_comparison(self) -> None:
        benchmark = self.dataset_name.get()
        if not benchmark:
            self.status.set("Choose a dataset-scale output folder first.")
            return
        added = 0
        for index in self.image_list.curselection():
            if len(self.comparison_selections) >= 3:
                self.status.set("At most 3 samples can be compared at once.")
                break
            entry = (benchmark, index)
            if entry in self.comparison_selections:
                continue
            self.comparison_selections.append(entry)
            added += 1
        self._refresh_comparison_list()
        if added:
            self.status.set(f"Comparison now holds {len(self.comparison_selections)} sample(s).")

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
        """Checked methods that exist in every dataset of the comparison."""

        involved = {benchmark for benchmark, _ in self.comparison_selections}
        common: set[str] | None = None
        for benchmark in involved:
            names = {method.casefold() for method, _ in self.candidates.get(benchmark, [])}
            common = names if common is None else common & names
        if not common:
            return []
        browsed = self.candidates.get(self.dataset_name.get(), [])
        return [
            method
            for method, _path in browsed
            if method.casefold() in common and self._method_state.get(method, True)
        ]

    def _export_settings(self) -> dict[str, object]:
        if self._task_runner.is_busy:
            raise ValueError("A calculation is already running.")
        if not self.comparison_selections:
            raise ValueError(
                "Add at least one sample to the comparison (double-click an image or use 'Add to comparison')."
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
        highlights = [
            method for method in methods if self._highlight_state.get(method, False)
        ]
        return {
            "output_root": Path(self.output_root.get()),
            "dataset_root": Path(self.dataset_root.get()),
            "selections": list(self.comparison_selections),
            "methods": methods,
            "output_directory": figure_directory,
            "output_format": self.export_format.get(),
            "dpi": dpi,
            "highlights": highlights,
            "save_csv": bool(self.save_csv.get()),
        }

    def start_export(self) -> None:
        try:
            settings = self._export_settings()
        except ValueError as error:
            self.messagebox.showerror("Invalid selection", str(error))
            return
        self.status.set("Calculating selected per-band PSNR curves...")
        self._task_runner.run_export(settings, self._on_export_done)

    def start_preview(self) -> None:
        try:
            settings = self._export_settings()
        except ValueError as error:
            self.messagebox.showerror("Invalid selection", str(error))
            return
        self.status.set("Calculating per-band PSNR preview...")
        self._task_runner.run_preview(settings, self._on_preview_done)

    def _on_preview_done(
        self, data: SampleComparisonData | None, settings: dict[str, object], error: str | None
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
            figsize=self._preview_figsize(),
        )
        self._replace_preview(figure)
        self.status.set(
            f"Previewed {len(data.samples)} sample(s) using {len(data.methods)} method(s)."
        )

    def _preview_figsize(self) -> tuple[float, float]:
        """Match the preview figure to the available container pixels."""

        self.preview_container.update_idletasks()
        width = max(self.preview_container.winfo_width(), 640)
        height = max(self.preview_container.winfo_height(), 360)
        return (width / self._PREVIEW_DPI, height / self._PREVIEW_DPI)

    def _on_preview_container_resize(self, _event: object | None = None) -> None:
        """Re-fit the preview figure whenever the container changes size."""

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
        """Replace the preview canvas, cleaning up old resources."""

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
        self.status.set(f"Exported {len(written)} file(s).")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def launch_gui(*, output_root: Path, dataset_root: Path) -> None:
    """Launch the interactive per-band PSNR export tool."""

    matplotlib.use("TkAgg")
    BandPsnrGui(output_root=output_root, dataset_root=dataset_root).run()


def main() -> int:
    """Launch the GUI using the repository-default output and Dataset roots."""

    launch_gui(output_root=Path("output"), dataset_root=Path("Dataset"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
