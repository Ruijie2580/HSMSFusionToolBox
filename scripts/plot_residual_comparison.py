"""Interactive desktop tool for RGB and residual comparison of HSI predictions."""

from __future__ import annotations

import argparse
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

# Allow ``python scripts/plot_residual_comparison.py`` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from scripts._shared import (
    _FIGURE_FONT,
    discover_output_predictions,
    infer_target_path,
    read_hsi_sample,
    sample_count,
    setup_matplotlib_font,
)

setup_matplotlib_font()


# ---------------------------------------------------------------------------
# Configuration dataclass (replaces argparse.Namespace for internal use)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResidualComparisonConfig:
    """Typed configuration for residual comparison rendering."""

    target: Path
    target_key: str = "HRHS"
    prediction: list[str] = field(default_factory=list)
    prediction_key: str = "prediction"
    sample_index: int = 0
    rgb_bands: tuple[int, int, int] = (54, 36, 21)
    residual_vmin: float = 0.0
    residual_vmax: float | None = None
    cmap: str = "viridis"
    contrast_stretch: bool = True
    stretch_percentiles: tuple[float, float] = (1.0, 99.0)
    include_sam: bool = False
    include_gt: bool = False
    sam_vmin: float = 0.0
    sam_vmax: float | None = None
    sam_cmap: str = "magma"
    zoom: tuple[int, int, int, int] | None = None
    zoom_corner: str = "lower right"
    title: str = ""
    hide_row_labels: bool = False
    dpi: int = 300
    output: Path | None = None


@dataclass(frozen=True)
class ComparisonData:
    """One target sample and the selected method predictions to compare."""

    target: np.ndarray
    methods: list[tuple[str, np.ndarray]]
    residuals: list[np.ndarray]
    sam_maps: list[np.ndarray] | None
    residual_vmin: float
    residual_vmax: float
    sam_vmin: float | None
    sam_vmax: float | None
    rgb_images: list[np.ndarray] | None = None
    rgb_params: tuple[tuple[int, int, int], bool, tuple[float, float]] | None = None


# ---------------------------------------------------------------------------
# Input parsing (shared by the CLI and the GUI)
# ---------------------------------------------------------------------------


def _parse_predictions(items: Sequence[str]) -> list[tuple[str, Path]]:
    predictions: list[tuple[str, Path]] = []
    seen_names: set[str] = set()
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --prediction value {item!r}; expected METHOD=PATH.")
        name, raw_path = item.split("=", maxsplit=1)
        name = name.strip()
        if not name or not raw_path.strip():
            raise ValueError(f"Invalid --prediction value {item!r}; expected METHOD=PATH.")
        if name in seen_names:
            raise ValueError(f"Duplicate method name in --prediction: {name!r}.")
        seen_names.add(name)
        predictions.append((name, Path(raw_path).expanduser()))
    return predictions


def parse_rgb_bands(text: str) -> tuple[int, int, int]:
    """Parse an ``R, G, B`` band string into a validated index tuple."""

    try:
        bands = tuple(int(value.strip()) for value in text.split(","))
    except ValueError as error:
        raise ValueError("RGB bands must be three comma-separated integer indices.") from error
    if len(bands) != 3:
        raise ValueError("RGB bands must contain exactly three indices, e.g. 54, 36, 21.")
    return bands


def parse_metric_range(minimum: str, maximum: str, *, label: str) -> tuple[float, float | None]:
    """Parse a ``min``/``max`` pair where ``max`` may be empty or ``auto``."""

    try:
        vmin = float(minimum.strip())
    except ValueError as error:
        raise ValueError(f"{label} minimum must be a number.") from error
    maximum = maximum.strip().casefold()
    if maximum in {"", "auto"}:
        return vmin, None
    try:
        vmax = float(maximum)
    except ValueError as error:
        raise ValueError(f"{label} maximum must be a number or 'auto'.") from error
    if vmax <= vmin:
        raise ValueError(f"{label} maximum must exceed its minimum.")
    return vmin, vmax


def parse_stretch_percentiles(low: str, high: str) -> tuple[float, float]:
    """Parse the contrast-stretch percentile pair."""

    try:
        low_value = float(low.strip())
        high_value = float(high.strip())
    except ValueError as error:
        raise ValueError("Stretch percentiles must be numeric.") from error
    if not 0.0 <= low_value < high_value <= 100.0:
        raise ValueError("Stretch percentiles must satisfy 0 <= low < high <= 100.")
    return low_value, high_value


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------


def _rgb_image(
    cube: np.ndarray,
    bands: Sequence[int],
    *,
    contrast_stretch: bool,
    percentiles: Sequence[float],
) -> np.ndarray:
    """Extract an RGB image from a CHW HSI cube with optional contrast stretch."""

    if cube.ndim != 3:
        raise ValueError(f"Expected CHW HSI sample, got {cube.shape}.")
    if any(band < 0 or band >= cube.shape[0] for band in bands):
        raise ValueError(f"RGB bands {tuple(bands)} are invalid for {cube.shape[0]} spectral channels.")
    rgb = np.moveaxis(cube[list(bands)], 0, -1).copy()
    if not contrast_stretch:
        return np.clip(rgb, 0.0, 1.0)

    low, high = float(percentiles[0]), float(percentiles[1])
    if not 0.0 <= low < high <= 100.0:
        raise ValueError("stretch-percentiles must satisfy 0 <= LOW < HIGH <= 100.")
    # Vectorized: compute percentiles for all channels at once
    lowers, uppers = np.percentile(rgb, (low, high), axis=(0, 1))
    denom = np.maximum(uppers - lowers, 1e-8)
    rgb = (rgb - lowers) / denom
    return np.clip(rgb, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Zoom utilities
# ---------------------------------------------------------------------------


def _zoom_bounds(zoom: Sequence[int] | None, shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    if zoom is None:
        return None
    row, column, height, width = zoom
    image_height, image_width = shape
    if height < 1 or width < 1:
        raise ValueError("Zoom height and width must be positive.")
    if height != width:
        raise ValueError("Zoom height and width must be equal so the selected region is square.")
    if row < 0 or column < 0 or row + height > image_height or column + width > image_width:
        raise ValueError(f"Zoom {tuple(zoom)} lies outside the image shape {shape}.")
    return row, column, height, width


def selection_to_zoom_bounds(
    x0: float | None,
    y0: float | None,
    x1: float | None,
    y1: float | None,
    *,
    shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """Convert a Matplotlib drag in image coordinates to a validated CHW crop."""

    if None in (x0, y0, x1, y1):
        return None
    left, right = sorted((int(np.floor(min(x0, x1))), int(np.ceil(max(x0, x1)))))
    top, bottom = sorted((int(np.floor(min(y0, y1))), int(np.ceil(max(y0, y1)))))
    if right - left < 2 or bottom - top < 2:
        return None
    side = min(max(right - left, bottom - top), min(shape))
    center_row = (top + bottom) / 2.0
    center_column = (left + right) / 2.0
    top = int(round(center_row - side / 2.0))
    left = int(round(center_column - side / 2.0))
    top = min(max(top, 0), shape[0] - side)
    left = min(max(left, 0), shape[1] - side)
    return _zoom_bounds((top, left, side, side), shape)


def _inset_bbox(corner: str) -> tuple[float, float, float, float]:
    padding = 0.03
    width = height = 0.38
    positions = {
        "upper right": (1.0 - width - padding, 1.0 - height - padding),
        "upper left": (padding, 1.0 - height - padding),
        "lower right": (1.0 - width - padding, padding),
        "lower left": (padding, padding),
    }
    if corner not in positions:
        valid = ", ".join(sorted(positions))
        raise ValueError(f"Unknown zoom corner {corner!r}; expected one of: {valid}.")
    left, bottom = positions[corner]
    return left, bottom, width, height


def _add_zoom(
    axis: plt.Axes,
    image: np.ndarray,
    *,
    bounds: tuple[int, int, int, int] | None,
    is_rgb: bool,
    cmap: str,
    value_vmin: float,
    value_vmax: float,
    corner: str,
) -> None:
    if bounds is None:
        return
    row, column, height, width = bounds
    crop = image[row : row + height, column : column + width]
    inset = axis.inset_axes(_inset_bbox(corner))
    if is_rgb:
        inset.imshow(crop, rasterized=True)
    else:
        inset.imshow(crop, cmap=cmap, vmin=value_vmin, vmax=value_vmax, rasterized=True)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_color("red")
        spine.set_linewidth(1.6)
    axis.add_patch(
        patches.Rectangle(
            (column, row),
            width,
            height,
            linewidth=1.6,
            edgecolor="red",
            facecolor="none",
        )
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_comparison_data(config: ResidualComparisonConfig) -> ComparisonData:
    """Load one target/prediction sample per method and derive its residual maps."""

    predictions = _parse_predictions(config.prediction)
    if not predictions:
        raise ValueError("Select at least one prediction (METHOD=PATH).")

    target = np.clip(
        read_hsi_sample(config.target, config.target_key, config.sample_index), 0.0, 1.0
    )
    methods: list[tuple[str, np.ndarray]] = []
    for method_name, prediction_path in predictions:
        prediction = np.clip(
            read_hsi_sample(prediction_path, config.prediction_key, config.sample_index),
            0.0,
            1.0,
        )
        if prediction.shape != target.shape:
            raise ValueError(
                f"{method_name} prediction shape {prediction.shape} does not match target {target.shape}."
            )
        methods.append((method_name, prediction))

    residuals = [np.mean(np.abs(target - pred), axis=0) for _, pred in methods]
    residual_vmin, residual_vmax = _display_range(
        residuals, vmin=config.residual_vmin, vmax=config.residual_vmax, metric_name="Residual"
    )
    sam_maps = None
    sam_vmin = sam_vmax = None
    if config.include_sam:
        sam_maps = [_pixel_sam(pred, target) for _, pred in methods]
        sam_vmin, sam_vmax = _display_range(
            sam_maps, vmin=config.sam_vmin, vmax=config.sam_vmax, metric_name="SAM"
        )

    if config.include_gt:
        methods.append(("GT", target.copy()))
        residuals.append(np.zeros(target.shape[1:], dtype=np.float32))
        if config.include_sam:
            assert sam_maps is not None
            sam_maps.append(np.zeros(target.shape[1:], dtype=np.float32))

    # Precompute the RGB composites once so zoom redraws reuse them.
    rgb_params = (
        tuple(config.rgb_bands),
        bool(config.contrast_stretch),
        tuple(config.stretch_percentiles),
    )
    rgb_images = [
        _rgb_image(
            cube,
            config.rgb_bands,
            contrast_stretch=config.contrast_stretch,
            percentiles=config.stretch_percentiles,
        )
        for _, cube in methods
    ]

    return ComparisonData(
        target=target,
        methods=methods,
        residuals=residuals,
        sam_maps=sam_maps,
        residual_vmin=residual_vmin,
        residual_vmax=residual_vmax,
        sam_vmin=sam_vmin,
        sam_vmax=sam_vmax,
        rgb_images=rgb_images,
        rgb_params=rgb_params,
    )


def _display_range(
    maps: Sequence[np.ndarray],
    *,
    vmin: float,
    vmax: float | None,
    metric_name: str,
) -> tuple[float, float]:
    if not np.isfinite(vmin):
        raise ValueError(f"{metric_name} display minimum must be finite.")
    if vmax is None:
        values = np.concatenate([item.ravel() for item in maps])
        resolved_vmax = float(np.percentile(values, 99.0))
        return vmin, max(resolved_vmax, vmin + 1e-8)
    resolved_vmax = float(vmax)
    if not np.isfinite(resolved_vmax) or resolved_vmax <= vmin:
        raise ValueError(f"{metric_name} display maximum must be finite and exceed its minimum.")
    return vmin, resolved_vmax


def _pixel_sam(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute a two-dimensional spectral-angle map in radians."""

    dot_product = np.sum(prediction * target, axis=0)
    denominator = np.linalg.norm(prediction, axis=0) * np.linalg.norm(target, axis=0)
    cosine = dot_product / np.maximum(denominator, 1e-8)
    return np.arccos(np.clip(cosine, -1.0, 1.0)).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Figure creation
# ---------------------------------------------------------------------------


def _ideal_figsize(method_count: int, row_count: int) -> tuple[float, float]:
    """Natural figure size for the given panel counts, shared by CLI and GUI."""

    return (max(4.8, 3.1 * method_count), 2.9 * row_count)


def _populate_comparison_figure(
    figure: plt.Figure,
    comparison: ComparisonData,
    config: ResidualComparisonConfig,
    *,
    zoom_bounds: tuple[int, int, int, int] | None,
) -> np.ndarray:
    """Draw the RGB/residual/SAM panels into ``figure`` and return its panel axes."""

    method_count = len(comparison.methods)
    if method_count == 0:
        raise ValueError("Select at least one fusion method to display.")
    has_sam = comparison.sam_maps is not None
    row_count = 3 if has_sam else 2
    figure.clear()
    figure.set_size_inches(*_ideal_figsize(method_count, row_count))

    grid = figure.add_gridspec(
        row_count,
        method_count + 1,
        width_ratios=[*[1.0] * method_count, 0.055],
    )
    axes = np.empty((row_count, method_count), dtype=object)
    for row in range(row_count):
        for column in range(method_count):
            axes[row, column] = figure.add_subplot(grid[row, column])
    residual_colorbar_axis = figure.add_subplot(grid[1, -1])
    sam_colorbar_axis = figure.add_subplot(grid[2, -1]) if has_sam else None

    rgb_params = (
        tuple(config.rgb_bands),
        bool(config.contrast_stretch),
        tuple(config.stretch_percentiles),
    )
    rgb_cached = comparison.rgb_images if comparison.rgb_params == rgb_params else None

    residual_images: list[object] = []
    sam_images: list[object] = []
    for column, ((method_name, prediction), residual) in enumerate(
        zip(comparison.methods, comparison.residuals, strict=True)
    ):
        # RGB row
        rgb_axis = axes[0, column]
        if rgb_cached is not None:
            rgb = rgb_cached[column]
        else:
            rgb = _rgb_image(
                prediction,
                config.rgb_bands,
                contrast_stretch=config.contrast_stretch,
                percentiles=config.stretch_percentiles,
            )
        rgb_axis.imshow(rgb, rasterized=True)
        rgb_axis.set_axis_off()
        rgb_axis.set_title(
            method_name, fontsize=13, fontweight="normal", fontfamily=_FIGURE_FONT, pad=6
        )
        _add_zoom(
            rgb_axis, rgb, bounds=zoom_bounds, is_rgb=True,
            cmap=config.cmap, value_vmin=0.0, value_vmax=1.0, corner=config.zoom_corner,
        )

        # Residual row
        residual_axis = axes[1, column]
        residual_images.append(
            residual_axis.imshow(
                residual, cmap=config.cmap,
                vmin=comparison.residual_vmin, vmax=comparison.residual_vmax, rasterized=True,
            )
        )
        residual_axis.set_axis_off()
        _add_zoom(
            residual_axis, residual, bounds=zoom_bounds, is_rgb=False,
            cmap=config.cmap,
            value_vmin=comparison.residual_vmin, value_vmax=comparison.residual_vmax,
            corner=config.zoom_corner,
        )

        # SAM row (optional)
        if has_sam:
            assert comparison.sam_maps is not None
            assert comparison.sam_vmin is not None and comparison.sam_vmax is not None
            sam_axis = axes[2, column]
            sam_images.append(
                sam_axis.imshow(
                    comparison.sam_maps[column], cmap=config.sam_cmap,
                    vmin=comparison.sam_vmin, vmax=comparison.sam_vmax, rasterized=True,
                )
            )
            sam_axis.set_axis_off()
            _add_zoom(
                sam_axis, comparison.sam_maps[column], bounds=zoom_bounds, is_rgb=False,
                cmap=config.sam_cmap,
                value_vmin=comparison.sam_vmin, value_vmax=comparison.sam_vmax,
                corner=config.zoom_corner,
            )

    # Row labels
    if not config.hide_row_labels:
        labels = ["RGB", "MAE"]
        if has_sam:
            labels.append("SAM")
        for axis, label in zip(
            (axes[row, 0] for row in range(row_count)), labels, strict=True
        ):
            axis.text(
                -0.06, 0.5, label, transform=axis.transAxes,
                rotation=90, va="center", ha="right",
                fontsize=13, fontweight="normal", fontfamily=_FIGURE_FONT, clip_on=False,
            )

    # Title
    if config.title:
        figure.suptitle(config.title, fontsize=15, fontweight="normal", fontfamily=_FIGURE_FONT)

    # Colorbars
    residual_colorbar = figure.colorbar(residual_images[0], cax=residual_colorbar_axis)
    _style_colorbar(residual_colorbar)
    if has_sam:
        assert sam_colorbar_axis is not None
        sam_colorbar = figure.colorbar(sam_images[0], cax=sam_colorbar_axis)
        _style_colorbar(sam_colorbar)

    return axes


def create_residual_comparison_figure(
    comparison: ComparisonData,
    config: ResidualComparisonConfig,
    *,
    zoom_bounds: tuple[int, int, int, int] | None,
    figure: plt.Figure | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Create (or repopulate an existing) figure with the RGB/residual panels."""

    if figure is None:
        figure = plt.figure(layout="constrained")
    axes = _populate_comparison_figure(figure, comparison, config, zoom_bounds=zoom_bounds)
    return figure, axes


def _style_colorbar(colorbar: object) -> None:
    for tick in colorbar.ax.get_yticklabels():
        tick.set_fontfamily(_FIGURE_FONT)


def render_residual_comparison(config: ResidualComparisonConfig) -> Path:
    """Load one target/prediction sample per method and save a comparison figure."""

    comparison = load_comparison_data(config)
    bounds = _zoom_bounds(config.zoom, comparison.target.shape[-2:])
    figure, _ = create_residual_comparison_figure(comparison, config, zoom_bounds=bounds)

    assert config.output is not None
    output_path = config.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=config.dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output_path


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------


class _GuiTaskRunner:
    """Runs background loading work for ResidualComparisonGui on daemon threads."""

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
        for button in self._buttons:
            button.configure(state=state)

    def run(
        self,
        work: Callable[[], object],
        on_done: Callable[[object], None],
        on_error: Callable[[str], None],
    ) -> None:
        if self._busy.is_set():
            return
        self._busy.set()
        self._set_buttons_state("disabled")

        def worker() -> None:
            try:
                result = work()
            except Exception as error:  # GUI boundary: surface any load failure without freezing
                self._root.after(0, on_error, str(error) or type(error).__name__)
                return
            finally:
                self._busy.clear()
                self._root.after(0, self._set_buttons_state, "normal")
            self._root.after(0, on_done, result)

        threading.Thread(target=worker, daemon=True).start()


class ResidualComparisonGui:
    """Small Tkinter front end for discovering outputs and selecting a zoom region."""

    def __init__(self, *, output_root: Path, dataset_root: Path) -> None:
        try:
            import tkinter as tk
            from tkinter import filedialog, messagebox, ttk

            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
            from matplotlib.widgets import RectangleSelector
        except ImportError as error:  # pragma: no cover - platform-owned dependency
            raise RuntimeError("The residual-comparison GUI requires Tkinter.") from error

        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.figure_canvas_type = FigureCanvasTkAgg
        self.toolbar_type = NavigationToolbar2Tk
        self.rectangle_selector_type = RectangleSelector

        self.root = tk.Tk()
        self.root.title("HSMSFusion Residual Comparison")
        self.root.geometry("1460x980")
        self.root.minsize(1100, 760)
        self.output_root = tk.StringVar(value=str(output_root))
        self.dataset_root = tk.StringVar(value=str(dataset_root))
        self.dataset_name = tk.StringVar()
        self.target_path = tk.StringVar()
        self.sample_index = tk.IntVar(value=0)
        self.rgb_bands = tk.StringVar(value="54, 36, 21")
        self.contrast_stretch = tk.BooleanVar(value=True)
        self.stretch_low = tk.StringVar(value="1")
        self.stretch_high = tk.StringVar(value="99")
        self.residual_vmin = tk.StringVar(value="0")
        self.residual_vmax = tk.StringVar(value="auto")
        self.include_sam = tk.BooleanVar(value=False)
        self.include_gt = tk.BooleanVar(value=False)
        self.sam_vmin = tk.StringVar(value="0")
        self.sam_vmax = tk.StringVar(value="auto")
        self.export_format = tk.StringVar(value="png")
        self.export_dpi = tk.IntVar(value=300)
        self.status = tk.StringVar(value="Scan an output directory to begin.")
        self.candidates: dict[str, list[tuple[str, Path]]] = {}
        self.comparison: ComparisonData | None = None
        self.comparison_config: ResidualComparisonConfig | None = None
        self.zoom_bounds: tuple[int, int, int, int] | None = None
        self.figure: plt.Figure | None = None
        self.canvas = None
        self.toolbar = None
        self.selectors: list[object] = []
        self._task_runner = _GuiTaskRunner(self.root)
        self._drag_index: int | None = None
        self._dragging_method = False
        self._index_limit_job: object | None = None
        self._ideal_figsize: tuple[float, float] | None = None

        self._build_layout()
        self._task_runner.register_button(self.visualize_button)
        self._task_runner.register_button(self.prev_button)
        self._task_runner.register_button(self.next_button)
        self._task_runner.register_button(self.clear_zoom_button)
        self._task_runner.register_button(self.save_button)
        self.root.bind("<Left>", self._make_arrow_handler(-1))
        self.root.bind("<Right>", self._make_arrow_handler(1))
        self.scan_outputs()

    # -- Layout split into focused methods ----------------------------------

    def _build_layout(self) -> None:
        controls = self.ttk.Frame(self.root, padding=10)
        controls.pack(fill="x")
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(3, weight=1)
        controls.columnconfigure(4, weight=1)

        self._build_path_row(controls)
        self._build_dataset_row(controls)
        self._build_method_and_image_rows(controls)
        self._build_display_row(controls)
        self._build_action_row(controls)

        self.figure_frame = self.ttk.Frame(self.root, padding=(10, 0, 10, 10))
        self.figure_frame.pack(fill="both", expand=True)

    def _build_path_row(self, controls: object) -> None:
        self.ttk.Label(controls, text="Output root").grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.ttk.Entry(controls, textvariable=self.output_root).grid(row=0, column=1, sticky="ew")
        self.ttk.Button(controls, text="Scan output", command=self.scan_outputs).grid(
            row=0, column=2, sticky="w", padx=6
        )
        self.ttk.Label(controls, text="Dataset root").grid(row=0, column=3, sticky="w", padx=(18, 6))
        self.ttk.Entry(controls, textvariable=self.dataset_root).grid(row=0, column=4, sticky="ew")

    def _build_dataset_row(self, controls: object) -> None:
        self.ttk.Label(controls, text="Candidate dataset").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.dataset_combo = self.ttk.Combobox(
            controls, textvariable=self.dataset_name, state="readonly"
        )
        self.dataset_combo.grid(row=1, column=1, sticky="ew", pady=(8, 0))
        self.dataset_combo.bind("<<ComboboxSelected>>", self._on_dataset_changed)
        self.ttk.Label(controls, text="Target HRHS").grid(
            row=1, column=3, sticky="w", padx=(18, 6), pady=(8, 0)
        )
        self.ttk.Entry(controls, textvariable=self.target_path).grid(
            row=1, column=4, sticky="ew", pady=(8, 0)
        )
        self.ttk.Button(controls, text="Browse", command=self._browse_target).grid(
            row=1, column=5, sticky="w", padx=(6, 0), pady=(8, 0)
        )

    def _build_method_and_image_rows(self, controls: object) -> None:
        self.method_frame = self.ttk.LabelFrame(controls, text="Fusion methods (drag or use buttons to reorder)", padding=6)
        self.method_frame.grid(row=2, column=0, columnspan=7, sticky="ew", pady=(10, 0))

        listbox_frame = self.ttk.Frame(self.method_frame)
        listbox_frame.pack(side="left", fill="both", expand=True)
        self.method_listbox = self.tk.Listbox(
            listbox_frame, selectmode="extended", height=4, exportselection=False,
        )
        self.method_listbox.pack(side="left", fill="both", expand=True)
        scrollbar = self.ttk.Scrollbar(listbox_frame, orient="vertical", command=self.method_listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.method_listbox.configure(yscrollcommand=scrollbar.set)

        button_frame = self.ttk.Frame(self.method_frame)
        button_frame.pack(side="left", padx=(6, 0))
        self.ttk.Button(button_frame, text="Up", width=6, command=self._move_method_up).pack(pady=2)
        self.ttk.Button(button_frame, text="Down", width=6, command=self._move_method_down).pack(pady=2)

        self.method_listbox.bind("<Button-1>", self._on_method_drag_start, add="+")
        self.method_listbox.bind("<B1-Motion>", self._on_method_drag_motion, add="+")
        self.method_listbox.bind("<ButtonRelease-1>", self._on_method_drag_end, add="+")
        self.method_listbox.bind("<<ListboxSelect>>", self._on_method_selection_changed, add="+")

        self.ttk.Label(controls, text="Image index").grid(row=3, column=0, sticky="w", pady=(8, 0))
        self.index_spinbox = self.ttk.Spinbox(
            controls, from_=0, to=0, textvariable=self.sample_index, width=8
        )
        self.index_spinbox.grid(row=3, column=1, sticky="w", pady=(8, 0))
        self.ttk.Label(controls, text="RGB bands (R,G,B)").grid(
            row=3, column=2, sticky="e", padx=(18, 6), pady=(8, 0)
        )
        self.ttk.Entry(controls, textvariable=self.rgb_bands, width=16).grid(
            row=3, column=3, sticky="w", pady=(8, 0)
        )
        self.ttk.Checkbutton(controls, text="Contrast stretch", variable=self.contrast_stretch).grid(
            row=3, column=4, sticky="w", padx=(18, 6), pady=(8, 0)
        )
        self.ttk.Label(controls, text="Percentiles").grid(row=3, column=5, sticky="e", pady=(8, 0))
        percentile_frame = self.ttk.Frame(controls)
        percentile_frame.grid(row=3, column=6, sticky="w", padx=(4, 0), pady=(8, 0))
        self.ttk.Entry(percentile_frame, textvariable=self.stretch_low, width=5).pack(side="left")
        self.ttk.Label(percentile_frame, text="to").pack(side="left", padx=3)
        self.ttk.Entry(percentile_frame, textvariable=self.stretch_high, width=5).pack(side="left")

    def _build_display_row(self, controls: object) -> None:
        display_frame = self.ttk.LabelFrame(controls, text="Metric ranges and export", padding=6)
        display_frame.grid(row=4, column=0, columnspan=7, sticky="ew", pady=(10, 0))
        self.ttk.Label(display_frame, text="Residual min / max").grid(row=0, column=0, sticky="w")
        self.ttk.Entry(display_frame, textvariable=self.residual_vmin, width=8).grid(
            row=0, column=1, sticky="w", padx=(6, 2)
        )
        self.ttk.Entry(display_frame, textvariable=self.residual_vmax, width=8).grid(
            row=0, column=2, sticky="w"
        )
        self.ttk.Checkbutton(display_frame, text="Include SAM row", variable=self.include_sam).grid(
            row=0, column=3, sticky="w", padx=(20, 6)
        )
        self.ttk.Checkbutton(display_frame, text="Include GT", variable=self.include_gt).grid(
            row=0, column=4, sticky="w", padx=(6, 6)
        )
        self.ttk.Label(display_frame, text="SAM min / max").grid(row=0, column=5, sticky="w")
        self.ttk.Entry(display_frame, textvariable=self.sam_vmin, width=8).grid(
            row=0, column=6, sticky="w", padx=(6, 2)
        )
        self.ttk.Entry(display_frame, textvariable=self.sam_vmax, width=8).grid(
            row=0, column=7, sticky="w"
        )
        self.ttk.Label(display_frame, text="Format").grid(row=0, column=8, sticky="w", padx=(20, 6))
        self.ttk.Combobox(
            display_frame, textvariable=self.export_format,
            values=("png", "jpeg", "pdf"), state="readonly", width=7,
        ).grid(row=0, column=9, sticky="w")
        self.ttk.Label(display_frame, text="DPI").grid(row=0, column=10, sticky="w", padx=(12, 4))
        self.ttk.Spinbox(display_frame, from_=72, to=1200, textvariable=self.export_dpi, width=7).grid(
            row=0, column=11, sticky="w"
        )

    def _build_action_row(self, controls: object) -> None:
        self.visualize_button = self.ttk.Button(controls, text="Visualize", command=self.visualize)
        self.visualize_button.grid(row=5, column=0, sticky="w", pady=(8, 0))
        self.prev_button = self.ttk.Button(
            controls, text="◀ Prev", width=8, command=lambda: self._step_sample(-1)
        )
        self.prev_button.grid(row=5, column=1, sticky="w", pady=(8, 0))
        self.next_button = self.ttk.Button(
            controls, text="Next ▶", width=8, command=lambda: self._step_sample(1)
        )
        self.next_button.grid(row=5, column=2, sticky="w", padx=(6, 0), pady=(8, 0))
        self.clear_zoom_button = self.ttk.Button(controls, text="Clear zoom", command=self.clear_zoom)
        self.clear_zoom_button.grid(row=5, column=3, sticky="w", padx=(6, 0), pady=(8, 0))
        self.save_button = self.ttk.Button(controls, text="Save figure", command=self.save_figure)
        self.save_button.grid(row=5, column=4, sticky="w", padx=(6, 0), pady=(8, 0))
        self.ttk.Label(controls, textvariable=self.status, foreground="#176b3a").grid(
            row=6, column=0, columnspan=7, sticky="w", pady=(8, 0)
        )

    # -- Navigation ----------------------------------------------------------

    def _make_arrow_handler(self, delta: int) -> Callable[[object], None]:
        def handler(_event: object) -> None:
            focused = self.root.focus_get()
            if focused is not None and focused.winfo_class() in {
                "Entry", "Spinbox", "TCombobox", "Listbox",
            }:
                return  # keep the arrow keys working inside text fields
            self._step_sample(delta)

        return handler

    def _step_sample(self, delta: int) -> None:
        if self._task_runner.is_busy:
            return
        if not self._selected_predictions():
            self.status.set("Scan an output directory and select at least one method first.")
            return
        try:
            index = self._read_int(self.sample_index, "Image index")
        except ValueError:
            index = 0
        target_index = min(max(index + delta, 0), self._max_sample_index())
        self.sample_index.set(target_index)
        self.visualize()

    def _max_sample_index(self) -> int:
        try:
            return max(int(float(self.index_spinbox.cget("to"))), 0)
        except (self.tk.TclError, TypeError, ValueError):
            return 0

    # -- Logic ---------------------------------------------------------------

    def run(self) -> None:
        self.root.mainloop()

    def scan_outputs(self) -> None:
        root = Path(self.output_root.get()).expanduser()
        self.candidates = discover_output_predictions(root)
        names = list(self.candidates)
        self.dataset_combo.configure(values=names)
        if not names:
            self.dataset_name.set("")
            self._clear_method_listbox()
            self.status.set(f"No predictions.h5 files found below {root}.")
            return
        selected = self.dataset_name.get()
        self.dataset_name.set(selected if selected in self.candidates else names[0])
        self._on_dataset_changed()
        self.status.set(f"Found {len(names)} dataset-scale candidate(s) below {root}.")

    def _on_dataset_changed(self, _event: object | None = None) -> None:
        selected = self.dataset_name.get()
        self._clear_method_listbox()
        self._drag_index = None
        self._dragging_method = False
        for method_name, _ in self.candidates.get(selected, []):
            self.method_listbox.insert("end", method_name)
        if self.method_listbox.size() > 0:
            self.method_listbox.select_set(0, "end")
        inferred = infer_target_path(Path(self.dataset_root.get()), selected)
        self.target_path.set(str(inferred))
        self.zoom_bounds = None
        self._update_index_limit()

    def _clear_method_listbox(self) -> None:
        self.method_listbox.delete(0, "end")

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
        text = self.method_listbox.get(self._drag_index)
        self.method_listbox.delete(self._drag_index)
        self.method_listbox.insert(target, text)
        self.method_listbox.selection_clear(0, "end")
        self.method_listbox.selection_set(target)
        self._drag_index = target

    def _on_method_drag_end(self, _event: object) -> None:
        if not self._dragging_method:
            return
        self._dragging_method = False
        self._drag_index = None
        self._on_method_selection_changed()

    def _on_method_selection_changed(self, _event: object | None = None) -> None:
        if self._dragging_method:
            return
        if self._index_limit_job is not None:
            self.root.after_cancel(self._index_limit_job)
        self._index_limit_job = self.root.after(200, self._flush_index_limit_update)

    def _flush_index_limit_update(self) -> None:
        self._index_limit_job = None
        self._update_index_limit()

    def _move_method_up(self) -> None:
        selected = self.method_listbox.curselection()
        if not selected or selected[0] == 0:
            return
        for index in selected:
            text = self.method_listbox.get(index)
            self.method_listbox.delete(index)
            self.method_listbox.insert(index - 1, text)
            self.method_listbox.selection_set(index - 1)

    def _move_method_down(self) -> None:
        selected = list(self.method_listbox.curselection())
        if not selected or selected[-1] == self.method_listbox.size() - 1:
            return
        for index in reversed(selected):
            text = self.method_listbox.get(index)
            self.method_listbox.delete(index)
            self.method_listbox.insert(index + 1, text)
            self.method_listbox.selection_set(index + 1)

    def _selected_predictions(self) -> list[tuple[str, Path]]:
        selected_name = self.dataset_name.get()
        candidates = dict(self.candidates.get(selected_name, []))
        selected_indices = self.method_listbox.curselection()
        ordered_names = [self.method_listbox.get(i) for i in selected_indices]
        return [
            (method_name, candidates[method_name])
            for method_name in ordered_names
            if method_name in candidates
        ]

    def _update_index_limit(self) -> None:
        try:
            target_count = sample_count(Path(self.target_path.get()), "HRHS")
            prediction_counts = [
                sample_count(path, "prediction") for _, path in self._selected_predictions()
            ]
            count = min([target_count, *prediction_counts])
        except (FileNotFoundError, KeyError, OSError, ValueError) as error:
            self.status.set(f"Could not determine the sample count: {error}")
            return
        maximum_index = max(count - 1, 0)
        self.index_spinbox.configure(to=maximum_index)
        try:
            current = int(self.sample_index.get())
        except (self.tk.TclError, TypeError, ValueError):
            return
        if current > maximum_index:
            self.sample_index.set(maximum_index)

    def _browse_target(self) -> None:
        selected = self.filedialog.askopenfilename(
            title="Select test.h5 containing HRHS",
            filetypes=(("HDF5", "*.h5"), ("All files", "*.*")),
        )
        if selected:
            self.target_path.set(selected)
            self._update_index_limit()

    def _read_int(self, variable: object, label: str) -> int:
        try:
            value = int(variable.get())  # type: ignore[attr-defined]
        except (self.tk.TclError, TypeError, ValueError) as error:
            raise ValueError(f"{label} must be an integer.") from error
        return value

    def _gather_config(self, selected: list[tuple[str, Path]]) -> ResidualComparisonConfig:
        rgb_bands = parse_rgb_bands(self.rgb_bands.get())
        stretch_percentiles = parse_stretch_percentiles(self.stretch_low.get(), self.stretch_high.get())
        residual_vmin, residual_vmax = parse_metric_range(
            self.residual_vmin.get(), self.residual_vmax.get(), label="Residual range"
        )
        include_sam = bool(self.include_sam.get())
        if include_sam:
            sam_vmin, sam_vmax = parse_metric_range(
                self.sam_vmin.get(), self.sam_vmax.get(), label="SAM range"
            )
        else:
            sam_vmin, sam_vmax = 0.0, None
        index = self._read_int(self.sample_index, "Image index")
        return ResidualComparisonConfig(
            target=Path(self.target_path.get()),
            prediction=[f"{name}={path}" for name, path in selected],
            sample_index=index,
            rgb_bands=rgb_bands,
            residual_vmin=residual_vmin,
            residual_vmax=residual_vmax,
            contrast_stretch=bool(self.contrast_stretch.get()),
            stretch_percentiles=stretch_percentiles,
            include_sam=include_sam,
            include_gt=bool(self.include_gt.get()),
            sam_vmin=sam_vmin,
            sam_vmax=sam_vmax,
            title=f"{self.dataset_name.get()} | image {index:04d}",
        )

    def visualize(self) -> None:
        selected = self._selected_predictions()
        if not selected:
            self.messagebox.showerror("No method selected", "Select at least one fusion method.")
            return
        try:
            config = self._gather_config(selected)
        except (ValueError, self.tk.TclError) as error:
            self.messagebox.showerror("Cannot visualize", str(error))
            return
        self.status.set("Loading target and predictions ...")
        self._task_runner.run(
            lambda: load_comparison_data(config),
            on_done=lambda comparison: self._on_comparison_loaded(config, comparison),
            on_error=self._on_load_error,
        )

    def _on_load_error(self, message: str) -> None:
        self.status.set(f"Load failed: {message}")
        self.messagebox.showerror("Cannot visualize", message)

    def _on_comparison_loaded(
        self, config: ResidualComparisonConfig, comparison: ComparisonData
    ) -> None:
        self.comparison = comparison
        self.comparison_config = config
        self.zoom_bounds = None
        self._draw_comparison()
        height, width = comparison.target.shape[-2:]
        self.status.set(
            f"Loaded {len(comparison.methods)} method(s), image {config.sample_index:04d} "
            f"({width}x{height}). Drag with the left mouse button on any panel to choose a zoom region."
        )

    def _draw_comparison(self) -> None:
        if self.comparison is None or self.comparison_config is None:
            return
        for selector in self.selectors:
            selector.set_active(False)
        self.selectors = []
        if self.figure is None or self.canvas is None:
            # One canvas/toolbar pair lives for the whole session; zoom redraws
            # only clear and repopulate the figure, so nothing flickers.
            self.figure = plt.figure(layout="constrained")
            self.canvas = self.figure_canvas_type(self.figure, master=self.figure_frame)
            self.toolbar = self.toolbar_type(self.canvas, self.figure_frame, pack_toolbar=False)
            self.toolbar.update()
            self.toolbar.pack(side="bottom", fill="x")
            self.canvas.get_tk_widget().pack(fill="both", expand=True)
            self.figure_frame.bind("<Configure>", self._on_frame_resized)
        assert self.comparison_config is not None
        _figure, axes = create_residual_comparison_figure(
            self.comparison, self.comparison_config, zoom_bounds=self.zoom_bounds, figure=self.figure
        )
        self._ideal_figsize = _ideal_figsize(
            len(self.comparison.methods),
            3 if self.comparison.sam_maps is not None else 2,
        )
        selector_properties = {"edgecolor": "red", "facecolor": "none", "linewidth": 1.3}
        self.selectors = [
            self.rectangle_selector_type(
                axis, self._on_zoom_selected, useblit=True, button=[1],
                minspanx=2, minspany=2, spancoords="data", props=selector_properties,
            )
            for axis in axes.flat
        ]
        self._fit_figure()

    def _on_frame_resized(self, _event: object) -> None:
        self._fit_figure()

    def _fit_figure(self) -> None:
        """Match the figure size to its container so the preview fills the window."""

        canvas = self.canvas
        figure = self.figure
        if canvas is None or figure is None:
            return
        self.figure_frame.update_idletasks()
        width = self.figure_frame.winfo_width()
        height = self.figure_frame.winfo_height()
        if width < 80:
            return
        toolbar_height = self.toolbar.winfo_height() if self.toolbar is not None else 0
        usable_height = max(height - toolbar_height, 80)
        figure.set_size_inches(width / figure.get_dpi(), usable_height / figure.get_dpi())
        canvas.draw_idle()

    def _on_zoom_selected(self, click_event: object, release_event: object) -> None:
        if self.comparison is None or self._task_runner.is_busy:
            return
        try:
            bounds = selection_to_zoom_bounds(
                getattr(click_event, "xdata", None),
                getattr(click_event, "ydata", None),
                getattr(release_event, "xdata", None),
                getattr(release_event, "ydata", None),
                shape=self.comparison.target.shape[-2:],
            )
        except ValueError as error:
            self.status.set(str(error))
            return
        if bounds is None:
            self.status.set("The selected zoom region is too small; drag a larger rectangle.")
            return
        self.zoom_bounds = bounds
        self._draw_comparison()
        self.status.set("Zoom region applied to all RGB and residual panels. Drag again to replace it.")

    def clear_zoom(self) -> None:
        if self.comparison is None or self._task_runner.is_busy:
            return
        self.zoom_bounds = None
        self._draw_comparison()
        self.status.set("Zoom region cleared.")

    def save_figure(self) -> None:
        if self.figure is None:
            self.messagebox.showerror("No figure", "Visualize an image before saving a figure.")
            return
        benchmark_name = self.dataset_name.get() or "comparison"
        default_directory = Path(self.output_root.get()) / benchmark_name
        export_format = self.export_format.get().casefold()
        suffixes = {"png": ".png", "jpeg": ".jpeg", "pdf": ".pdf"}
        suffix = suffixes[export_format]
        try:
            dpi = self._read_int(self.export_dpi, "DPI")
        except ValueError as error:
            self.messagebox.showerror("Invalid DPI", str(error))
            return
        if dpi < 1:
            self.messagebox.showerror("Invalid DPI", "DPI must be positive.")
            return
        try:
            index = int(self.sample_index.get())
        except (self.tk.TclError, TypeError, ValueError):
            index = 0
        default_name = f"residual_comparison_{index:04d}{suffix}"
        selected = self.filedialog.asksaveasfilename(
            title="Save residual comparison",
            initialdir=default_directory,
            initialfile=default_name,
            defaultextension=suffix,
            filetypes=(("PNG", "*.png"), ("JPEG", "*.jpeg *.jpg"), ("PDF", "*.pdf")),
        )
        if not selected:
            return
        output_path = Path(selected).with_suffix(suffix)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Export at the natural CLI figure size so the aspect ratio matches
        # headless renders, then snap back to the on-screen preview size.
        display_size = tuple(self.figure.get_size_inches())
        if self._ideal_figsize is not None:
            self.figure.set_size_inches(*self._ideal_figsize)
        try:
            self.figure.savefig(
                output_path, format=export_format, dpi=dpi, bbox_inches="tight", facecolor="white"
            )
        finally:
            self.figure.set_size_inches(*display_size)
        self.status.set(f"Saved figure: {output_path}")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def launch_gui(*, output_root: Path, dataset_root: Path) -> None:
    """Open the interactive desktop front end."""

    ResidualComparisonGui(output_root=output_root, dataset_root=dataset_root).run()


def main() -> int:
    """Launch the CLI or GUI."""

    parser = argparse.ArgumentParser(description="HSMSFusion residual comparison tool")
    parser.add_argument("--target", type=Path, help="Target HRHS HDF5 file")
    parser.add_argument("--target-key", default="HRHS", help="HDF5 dataset key for target")
    parser.add_argument("--prediction", nargs="+", help="METHOD=PATH pairs")
    parser.add_argument("--prediction-key", default="prediction", help="HDF5 dataset key for predictions")
    parser.add_argument("--sample-index", type=int, default=0, help="Image index to compare")
    parser.add_argument("--rgb-bands", type=int, nargs=3, default=[54, 36, 21], help="RGB band indices")
    parser.add_argument("--output", type=Path, default=Path("residual_comparison.png"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--contrast-stretch", action="store_true", default=True)
    parser.add_argument("--stretch-percentiles", type=float, nargs=2, default=[1.0, 99.0])
    parser.add_argument("--include-sam", action="store_true")
    parser.add_argument("--zoom", type=int, nargs=4, metavar=("ROW", "COL", "H", "W"))
    parser.add_argument("--zoom-corner", default="lower right")
    parser.add_argument("--title", default="")
    parser.add_argument("--hide-row-labels", action="store_true")
    parser.add_argument("--gui", action="store_true", help="Launch interactive GUI instead")

    arguments = parser.parse_args()

    if arguments.gui or arguments.target is None:
        launch_gui(output_root=Path("output"), dataset_root=Path("Dataset"))
        return 0

    config = ResidualComparisonConfig(
        target=arguments.target,
        target_key=arguments.target_key,
        prediction=arguments.prediction or [],
        prediction_key=arguments.prediction_key,
        sample_index=arguments.sample_index,
        rgb_bands=tuple(arguments.rgb_bands),
        residual_vmin=0.0,
        residual_vmax=None,
        cmap=arguments.cmap,
        contrast_stretch=arguments.contrast_stretch,
        stretch_percentiles=tuple(arguments.stretch_percentiles),
        include_sam=arguments.include_sam,
        sam_cmap="magma",
        zoom=arguments.zoom,
        zoom_corner=arguments.zoom_corner,
        title=arguments.title,
        hide_row_labels=arguments.hide_row_labels,
        dpi=arguments.dpi,
        output=arguments.output,
    )
    try:
        render_residual_comparison(config)
    except (FileNotFoundError, KeyError, OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
