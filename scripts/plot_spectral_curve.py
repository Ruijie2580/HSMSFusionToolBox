"""Plot per-pixel spectral vector curve comparisons between fused results and GT.

The toolbox test export uses the following layout::

    output/<dataset>_x<scale>/<method>/predictions.h5  (/prediction, NCHW)
    Dataset/<dataset>/test.h5                           (/HRHS, NCHW)

The spectral vector curve of a pixel is the vector of values taken along the
channel axis at one (row, column) position.  Run the script to launch a GUI
that previews the ground truth as a false-colour image; click a pixel (or
type row/column) to plot its ground-truth curve together with the curve of
every selected fusion method.  The X axis can be switched between band
index and wavelength (nm) when a matching degradation configuration exists.

The curve panel supports two comparison aids:

* Drag a rectangle on the curve axes to add a small 4:3 zoom inset of that
  region.  The inset is placed automatically so it never covers the legend
  and hides as few curve points as possible; it is outlined without
  connector lines and method curves are dashed inside the inset.
* Tick the per-method "Highlight" checkbox to add point markers, thicker
  lines and legend entries directly below the ground truth.
"""

from __future__ import annotations

import csv
import re
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

# Allow ``python scripts/plot_spectral_curve.py`` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np

from scripts._shared import (
    discover_output_predictions,
    infer_target_path,
    read_hsi_sample,
    resolve_benchmark,
    sample_count,
    select_predictions,
    setup_matplotlib_font,
    validate_hsi_dataset,
)

setup_matplotlib_font()

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_ROOT = _REPO_ROOT / "configs"

_X_MODE_BAND = "Band index"
_X_MODE_WAVELENGTH = "Wavelength (nm)"
_X_MODES = (_X_MODE_BAND, _X_MODE_WAVELENGTH)

_GT_COLOR = "#111111"
_GT_STYLE = {"color": _GT_COLOR, "linewidth": 2.2, "zorder": 5}
_METHOD_LINEWIDTH = 1.4
_HIGHLIGHT_LINEWIDTH = 2.0
_HIGHLIGHT_MARKER = "o"
_HIGHLIGHT_MARKERSIZE = 4.5
_ZOOM_LINESTYLE = "--"
_ZOOM_INSET_WIDTH_FRACTION = 0.28  # axes-fraction width; height follows 4:3 visually
_ZOOM_INSET_ASPECT = 4 / 3  # width / height of the inset in display units
_ZOOM_INSET_MARGIN = 0.03  # clearance towards the right and top axes edges
# Extra clearance towards the left and bottom edges so the inset's own small
# tick labels stay inside the axes instead of colliding with the main labels.
_ZOOM_INSET_AXIS_MARGIN = 0.085
_ZOOM_INSET_TICK_FONTSIZE = 7
_INDICATE_EDGE = "#606060"
_SELECTOR_STYLE = {
    "edgecolor": "#1f77b4", "facecolor": "#1f77b4", "alpha": 0.15, "linewidth": 1.2,
}
_FORMAT_SUFFIXES = {"png": "png", "jpg": "jpeg", "jpeg": "jpeg", "pdf": "pdf", "svg": "svg"}
_CSV_X_HEADERS = {_X_MODE_BAND: "BandIndex", _X_MODE_WAVELENGTH: "WavelengthNM"}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpectralCurveData:
    """Spectral vector curves of one pixel for the ground truth and methods."""

    benchmark: str
    target_path: Path
    sample_index: int
    row: int
    column: int
    gt_curve: np.ndarray  # shape (C,), float32
    curves: dict[str, np.ndarray]  # method display name -> (C,) float32
    wavelengths: np.ndarray | None  # shape (C,) in nm, or None

    @property
    def band_count(self) -> int:
        return int(self.gt_curve.shape[0])

    @property
    def methods(self) -> list[str]:
        return list(self.curves)


@dataclass(frozen=True)
class _LoadedSample:
    """Everything the GUI caches after reading one test sample."""

    benchmark: str
    target_path: Path
    sample_index: int
    predictions: list[tuple[str, Path]]  # (display name, predictions.h5)
    gt_cube: np.ndarray  # CHW float32
    sample_count: int
    wavelengths: np.ndarray | None


# ---------------------------------------------------------------------------
# Curve extraction and loading
# ---------------------------------------------------------------------------


def extract_spectral_curve(cube: np.ndarray, *, row: int, column: int) -> np.ndarray:
    """Extract the along-channel vector at ``(row, column)`` from a CHW cube."""

    if cube.ndim != 3:
        raise ValueError(f"Expected a CHW HSI sample, got shape {cube.shape}.")
    bands, height, width = (int(v) for v in cube.shape)
    if not 0 <= row < height or not 0 <= column < width:
        raise IndexError(
            f"pixel (row={row}, column={column}) is outside the image of size "
            f"{height}x{width}."
        )
    curve = np.asarray(cube[:, row, column], dtype=np.float32)
    if not np.isfinite(curve).all():
        raise ValueError(f"Pixel (row={row}, column={column}) contains NaN or Inf.")
    return curve


def _read_prediction_curve(
    path: Path,
    *,
    dataset_key: str,
    sample_index: int,
    row: int,
    column: int,
    expected_shape: tuple[int, int, int],
) -> np.ndarray:
    """Read only the ``(C,)`` spectral vector at one pixel from a predictions file.

    Reading a single column instead of the full NCHW cube keeps pixel picking
    responsive when many methods are compared.  The dataset shape is still
    validated against the ground truth before any value is used.
    """

    path = path.expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Prediction file does not exist: {path}")
    with h5py.File(path, "r") as handle:
        if dataset_key not in handle:
            raise KeyError(f"{path} does not contain dataset {dataset_key!r}.")
        dataset = handle[dataset_key]
        shape = validate_hsi_dataset(dataset, path=path, key=dataset_key)
        if tuple(shape[1:]) != tuple(expected_shape):
            raise ValueError(
                f"Prediction shape {tuple(shape[1:])} does not match the ground "
                f"truth shape {tuple(expected_shape)} for {path}."
            )
        if not 0 <= sample_index < shape[0]:
            raise IndexError(
                f"sample_index={sample_index} is outside [0, {shape[0] - 1}] for {path}."
            )
        curve = np.asarray(dataset[sample_index, :, row, column], dtype=np.float32)
    if not np.isfinite(curve).all():
        raise ValueError(f"{path}:{dataset_key}[{sample_index}] contains NaN or Inf.")
    return curve


def load_spectral_curve_data(
    *,
    output_root: Path,
    dataset_root: Path,
    benchmark: str,
    sample_index: int,
    row: int,
    column: int,
    methods: Sequence[str] | None = None,
    target_path: Path | None = None,
    target_key: str = "HRHS",
    prediction_key: str = "prediction",
    config_root: Path | None = None,
) -> SpectralCurveData:
    """Discover exports, read one sample and return per-method spectral curves."""

    discovered = discover_output_predictions(output_root)
    name, candidates = resolve_benchmark(discovered, benchmark)
    selected = select_predictions(candidates, methods)
    if not selected:
        raise ValueError("Select at least one fusion method to compare.")
    resolved_target = (
        Path(target_path).expanduser() if target_path is not None else infer_target_path(dataset_root, name)
    )
    gt_cube = read_hsi_sample(resolved_target, target_key, sample_index)
    bands, height, width = (int(v) for v in gt_cube.shape)
    gt_curve = extract_spectral_curve(gt_cube, row=row, column=column)
    curves = {
        display_name: _read_prediction_curve(
            prediction_path,
            dataset_key=prediction_key,
            sample_index=sample_index,
            row=row,
            column=column,
            expected_shape=(bands, height, width),
        )
        for display_name, prediction_path in selected
    }
    wavelengths = (
        resolve_wavelength_axis(config_root, name, bands) if config_root is not None else None
    )
    return SpectralCurveData(
        benchmark=name,
        target_path=resolved_target,
        sample_index=sample_index,
        row=row,
        column=column,
        gt_curve=gt_curve,
        curves=curves,
        wavelengths=wavelengths,
    )


def resolve_wavelength_axis(
    config_root: Path | None,
    benchmark: str,
    band_count: int,
) -> np.ndarray | None:
    """Return one wavelength (nm) per band from the degradation configuration.

    Looks up ``configs/degradation/<dataset>_<sensor>_x<scale>.yaml`` for a
    benchmark like ``chikusei_x4`` and linearly spans
    ``wavelength_start_nm``..``wavelength_end_nm`` over ``num_hsi_bands``.
    Returns ``None`` (fall back to the band-index axis) when no matching and
    consistent configuration exists.
    """

    if config_root is None:
        return None
    match = re.fullmatch(r"(?P<dataset>.+)_x(?P<scale>\d+)", benchmark, flags=re.IGNORECASE)
    if match is None:
        return None
    degradation_dir = Path(config_root).expanduser() / "degradation"
    if not degradation_dir.is_dir():
        return None
    dataset = match.group("dataset").casefold()
    scale = match.group("scale")
    config_path = next(
        (
            child
            for child in sorted(degradation_dir.glob("*.yaml"))
            # Match <dataset>_<sensor>_x<scale>.yaml with any sensor token.
            if re.fullmatch(rf"{re.escape(dataset)}_.+_x{re.escape(scale)}\.yaml", child.name.casefold())
        ),
        None,
    )
    if config_path is None:
        return None
    try:
        from omegaconf import OmegaConf

        config = OmegaConf.load(config_path)
        spectral = config["spectral"]
        start = float(spectral["wavelength_start_nm"])
        end = float(spectral["wavelength_end_nm"])
        bands = int(spectral["num_hsi_bands"])
    except Exception:  # malformed or unreadable configuration -> band index axis
        return None
    if bands != band_count or not end > start:
        return None
    return np.linspace(start, end, band_count, dtype=np.float32)


# ---------------------------------------------------------------------------
# False-colour preview helpers
# ---------------------------------------------------------------------------


def _default_rgb_bands(band_count: int) -> tuple[int, int, int]:
    """Pick a red/green/blue band triple spread across the spectral range."""

    if band_count < 1:
        raise ValueError("band_count must be positive.")
    last = band_count - 1
    bands = (int(round(0.7 * last)), int(round(0.5 * last)), int(round(0.3 * last)))
    if len(set(bands)) < 3 and band_count >= 3:
        bands = (last, last // 2, 0)
    return bands


def _parse_rgb_bands(text: str) -> tuple[int, int, int] | None:
    """Parse a ``"54,36,21"`` band triple; ``None`` means automatic selection."""

    stripped = text.strip()
    if not stripped:
        return None
    parts = [part.strip() for part in stripped.split(",")]
    if len(parts) != 3:
        raise ValueError("RGB bands must be empty (auto) or three comma-separated indices, e.g. 54,36,21.")
    return tuple(int(part) for part in parts)  # type: ignore[return-value]


def _rgb_image(
    cube: np.ndarray,
    bands: Sequence[int],
    *,
    contrast_stretch: bool,
    percentiles: Sequence[float] = (2.0, 98.0),
) -> np.ndarray:
    """Extract an RGB image from a CHW HSI cube with optional contrast stretch."""

    if cube.ndim != 3:
        raise ValueError(f"Expected a CHW HSI sample, got shape {cube.shape}.")
    if any(band < 0 or band >= cube.shape[0] for band in bands):
        raise ValueError(f"RGB bands {tuple(bands)} are invalid for {cube.shape[0]} spectral channels.")
    rgb = np.moveaxis(cube[list(bands)], 0, -1).copy()
    if not contrast_stretch:
        return np.clip(rgb, 0.0, 1.0)
    low, high = float(percentiles[0]), float(percentiles[1])
    if not 0.0 <= low < high <= 100.0:
        raise ValueError("Contrast stretch percentiles must satisfy 0 <= LOW < HIGH <= 100.")
    lowers, uppers = np.percentile(rgb, (low, high), axis=(0, 1))
    denominator = np.maximum(uppers - lowers, 1e-8)
    rgb = (rgb - lowers) / denominator
    return np.clip(rgb, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def create_preview_figure(
    cube: np.ndarray,
    *,
    rgb_bands: Sequence[int] | None = None,
    row: int,
    column: int,
    contrast_stretch: bool = True,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """False-colour ground-truth preview with a marker at the picked pixel."""

    band_total, height, width = (int(v) for v in cube.shape)
    bands = tuple(rgb_bands) if rgb_bands is not None else _default_rgb_bands(band_total)
    if not 0 <= row < height or not 0 <= column < width:
        raise IndexError(
            f"pixel (row={row}, column={column}) is outside the image of size {height}x{width}."
        )
    rgb = _rgb_image(cube, bands, contrast_stretch=contrast_stretch)
    figure = Figure(figsize=figsize or (4.4, 4.4), layout="constrained")
    axis = figure.add_subplot()
    axis.imshow(rgb, interpolation="nearest", origin="upper")
    axis.plot(
        column, row,
        linestyle="none", marker="o", markersize=14,
        markerfacecolor="none", markeredgecolor="#ff3b30", markeredgewidth=1.8,
        zorder=6,
    )
    axis.plot(
        column, row,
        linestyle="none", marker="+", markersize=9,
        color="#ff3b30", markeredgewidth=1.4, zorder=7,
    )
    axis.set_title("Ground truth (RGB preview)")
    axis.set_xlabel("Column")
    axis.set_ylabel("Row")
    return figure


def create_curve_figure(
    data: SpectralCurveData,
    *,
    x_mode: str = _X_MODE_BAND,
    figsize: tuple[float, float] | None = None,
    highlight_methods: Sequence[str] = (),
    zoom: Sequence[float] | None = None,
) -> Figure:
    """Draw the ground-truth spectral curve together with every method curve.

    ``highlight_methods`` marks the named methods with point markers and
    thicker lines, and orders their legend entries directly below the ground
    truth.  ``zoom`` is an ``(x0, x1, y0, y1)`` rectangle in display units;
    it adds a dashed zoom inset whose position is chosen automatically so it
    does not cover the legend or the zoomed source rectangle.
    """

    if x_mode not in _X_MODES:
        raise ValueError(f"Unsupported X axis mode: {x_mode!r}. Expected one of {_X_MODES}.")
    if x_mode == _X_MODE_WAVELENGTH and data.wavelengths is None:
        raise ValueError(
            f"Wavelength information is unavailable for {data.benchmark}; "
            f"switch the X axis back to {_X_MODE_BAND!r}."
        )
    band_count = data.band_count
    if x_mode == _X_MODE_WAVELENGTH:
        x_values = np.asarray(data.wavelengths, dtype=np.float64)
        xlabel = _X_MODE_WAVELENGTH
    else:
        x_values = np.arange(band_count, dtype=np.float64)
        xlabel = _X_MODE_BAND
    highlight_set = {name.casefold() for name in highlight_methods}

    figure = Figure(figsize=figsize or (6.8, 4.8), layout="constrained")
    axis = figure.add_subplot()
    axis.plot(x_values, data.gt_curve, label="Ground truth", **_GT_STYLE)
    _draw_spectral_lines(
        axis, data, x_values, highlight_set=highlight_set, dashed=False
    )
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Reflectance")
    axis.set_title(
        f"Spectral curves at row {data.row}, column {data.column} "
        f"({data.benchmark}, sample {data.sample_index})"
    )
    if x_mode == _X_MODE_BAND:
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.grid(alpha=0.3, linestyle=":")
    axis.legend(loc="best", fontsize=9, framealpha=0.9)

    if zoom is not None:
        x0, x1, y0, y1 = _normalize_zoom(zoom, x_values)
        # A renderer is required to measure the legend box for the placement.
        FigureCanvasAgg(figure)
        figure.canvas.draw()
        source_fraction = _data_rect_in_axes_fraction(axis, (x0, y0, x1, y1))
        plotted = [data.gt_curve, *data.curves.values()]
        left, bottom, right, top = _zoom_inset_position(axis, source_fraction, plotted, x_values)
        # inset_axes expects [left, bottom, width, height].
        inset = axis.inset_axes([left, bottom, right - left, top - bottom])
        inset.set_facecolor("white")
        inset.plot(x_values, data.gt_curve, color=_GT_COLOR, linewidth=1.6, zorder=5)
        _draw_spectral_lines(
            inset, data, x_values, highlight_set=highlight_set, dashed=True
        )
        inset.set_xlim(x0, x1)
        inset.set_ylim(y0, y1)
        inset.xaxis.set_major_locator(MaxNLocator(nbins=3))
        inset.yaxis.set_major_locator(MaxNLocator(nbins=3))
        inset.tick_params(labelsize=_ZOOM_INSET_TICK_FONTSIZE)
        inset.grid(alpha=0.25, linestyle=":")
        # Outline the zoomed source region without connector lines.
        axis.add_patch(
            Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                fill=False, edgecolor=_INDICATE_EDGE, linewidth=1.0, zorder=8,
            )
        )
    return figure


def _draw_spectral_lines(
    axis: plt.Axes,
    data: SpectralCurveData,
    x_values: np.ndarray,
    *,
    highlight_set: set[str],
    dashed: bool,
) -> None:
    """Draw every method curve; highlighted ones get markers and thicker lines."""

    colors = plt.get_cmap("tab20").colors
    names = list(data.curves)
    for name in _ordered_method_names(data, highlight_set):
        index = names.index(name)
        highlighted = name.casefold() in highlight_set
        style: dict[str, object] = {
            "color": colors[index % len(colors)],
            "linewidth": _HIGHLIGHT_LINEWIDTH if highlighted else _METHOD_LINEWIDTH,
            "zorder": 6 if highlighted else 3,
            "label": name,
        }
        if dashed:
            style["linestyle"] = _ZOOM_LINESTYLE
        if highlighted:
            style.update(
                marker=_HIGHLIGHT_MARKER,
                markersize=_HIGHLIGHT_MARKERSIZE,
                markerfacecolor="none",
            )
        axis.plot(x_values, data.curves[name], **style)


def _ordered_method_names(data: SpectralCurveData, highlight_set: set[str]) -> list[str]:
    """Highlighted methods first, so their legend entries sit below the GT."""

    names = list(data.curves)
    return [name for name in names if name.casefold() in highlight_set] + [
        name for name in names if name.casefold() not in highlight_set
    ]


def _normalize_zoom(zoom: Sequence[float], x_values: np.ndarray) -> tuple[float, float, float, float]:
    """Sort a zoom rectangle and clip it to the plotted extent; validate it."""

    if len(zoom) != 4:
        raise ValueError("zoom must contain (x0, x1, y0, y1).")
    x0, x1 = sorted((float(zoom[0]), float(zoom[1])))
    y0, y1 = sorted((float(zoom[2]), float(zoom[3])))
    x_low, x_high = float(x_values[0]), float(x_values[-1])
    x0, x1 = max(x0, x_low), min(x1, x_high)
    if not x1 > x0 or not y1 > y0:
        raise ValueError("The zoom region is empty after clipping to the data extent.")
    return x0, x1, y0, y1


def _rects_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float], *, pad: float = 0.012) -> bool:
    """Whether two ``(x0, y0, x1, y1)`` rectangles overlap with a small pad."""

    return (
        a[0] - pad < b[2] + pad
        and b[0] - pad < a[2] + pad
        and a[1] - pad < b[3] + pad
        and b[1] - pad < a[3] + pad
    )


def _bbox_in_axes_fraction(axis: plt.Axes, display_bbox: object) -> tuple[float, float, float, float]:
    """Convert a display-coordinates bbox to axes-fraction coordinates."""

    inverse = axis.transAxes.inverted()
    (x0, y0), (x1, y1) = inverse.transform(
        [[display_bbox.x0, display_bbox.y0], [display_bbox.x1, display_bbox.y1]]
    )
    return (x0, y0, x1, y1)


def _data_rect_in_axes_fraction(
    axis: plt.Axes, rect: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """Convert a data-coordinates rectangle to axes-fraction coordinates."""

    points = axis.transData.transform([[rect[0], rect[1]], [rect[2], rect[3]]])
    inverse = axis.transAxes.inverted()
    (x0, y0), (x1, y1) = inverse.transform(points)
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _inset_fraction_size(axis: plt.Axes) -> tuple[float, float]:
    """Axes-fraction inset size with a visual 4:3 (width/height) aspect."""

    figure = axis.figure
    figure_width, figure_height = (float(v) for v in (figure.get_size_inches() * figure.dpi))
    position = axis.get_position()
    axes_width = position.width * figure_width
    axes_height = position.height * figure_height
    width = _ZOOM_INSET_WIDTH_FRACTION
    visual_height = width * axes_width / _ZOOM_INSET_ASPECT
    return (width, visual_height / axes_height)


def _zoom_inset_position(
    axis: plt.Axes,
    source_rect: tuple[float, float, float, float] | None,
    curves: Sequence[np.ndarray],
    x_values: np.ndarray,
) -> tuple[float, float, float, float]:
    """Pick an axes-fraction rect for the zoom inset that hides little.

    Tries the four corners, the centre-top/centre-bottom and the
    centre-left/centre-right slots, dropping every slot that overlaps the
    legend box or the zoomed source rectangle.  Among the remaining slots the
    one covering the fewest data points of the plotted curves wins.  When
    every slot collides, the slot farthest from the obstacle centres is used
    as the least-bad fallback.
    """

    width, height = _inset_fraction_size(axis)
    margin = _ZOOM_INSET_MARGIN
    edge = _ZOOM_INSET_AXIS_MARGIN
    slots = {
        "upper right": (1.0 - width - margin, 1.0 - height - margin),
        "upper left": (edge, 1.0 - height - margin),
        "lower left": (edge, edge),
        "lower right": (1.0 - width - margin, edge),
        "upper centre": (0.5 - width / 2, 1.0 - height - margin),
        "lower centre": (0.5 - width / 2, edge),
        "centre left": (edge, 0.5 - height / 2),
        "centre right": (1.0 - width - margin, 0.5 - height / 2),
    }
    obstacles: list[tuple[float, float, float, float]] = []
    legend = axis.get_legend()
    if legend is not None:
        obstacles.append(_bbox_in_axes_fraction(axis, legend.get_window_extent()))
    if source_rect is not None:
        obstacles.append(source_rect)
    free_rects = []
    for position in ("upper right", "upper left", "lower left", "lower right", "upper centre", "lower centre"):
        left, bottom = slots[position]
        rect = (left, bottom, left + width, bottom + height)
        if not any(_rects_overlap(rect, obstacle) for obstacle in obstacles):
            free_rects.append(rect)
    if free_rects:
        return min(free_rects, key=lambda rect: _points_covered(axis, rect, curves, x_values))
    if obstacles:
        centre_x = sum((o[0] + o[2]) / 2 for o in obstacles) / len(obstacles)
        centre_y = sum((o[1] + o[3]) / 2 for o in obstacles) / len(obstacles)
    else:
        centre_x, centre_y = 0.5, 0.5
    best_left, best_bottom = max(
        slots.values(),
        key=lambda left_bottom: (left_bottom[0] + width / 2 - centre_x) ** 2
        + (left_bottom[1] + height / 2 - centre_y) ** 2,
    )
    return (best_left, best_bottom, best_left + width, best_bottom + height)


def _points_covered(
    axis: plt.Axes,
    rect: tuple[float, float, float, float],
    curves: Sequence[np.ndarray],
    x_values: np.ndarray,
) -> int:
    """Count curve data points inside an axes-fraction rectangle."""

    corners = axis.transAxes.transform([[rect[0], rect[1]], [rect[2], rect[3]]])
    (dx0, dy0), (dx1, dy1) = axis.transData.inverted().transform(corners)
    dx0, dx1 = min(dx0, dx1), max(dx0, dx1)
    dy0, dy1 = min(dy0, dy1), max(dy0, dy1)
    covered = 0
    for curve in curves:
        y_values = np.asarray(curve, dtype=np.float64)
        covered += int(
            np.sum(
                (x_values >= dx0) & (x_values <= dx1)
                & (y_values >= dy0) & (y_values <= dy1)
            )
        )
    return covered


def _band_to_display_x(band: float, wavelengths: np.ndarray | None, x_mode: str) -> float:
    """Map a (fractional) band index to the X coordinate of the axis mode."""

    if x_mode == _X_MODE_WAVELENGTH:
        if wavelengths is None:
            raise ValueError("Wavelength information is unavailable for this conversion.")
        axis_values = np.asarray(wavelengths, dtype=np.float64)
        return float(np.interp(band, np.arange(len(axis_values)), axis_values))
    return float(band)


def _display_x_to_band(x: float, wavelengths: np.ndarray | None, x_mode: str) -> float:
    """Map an X coordinate of the axis mode back to a (fractional) band index."""

    if x_mode == _X_MODE_WAVELENGTH:
        if wavelengths is None:
            raise ValueError("Wavelength information is unavailable for this conversion.")
        axis_values = np.asarray(wavelengths, dtype=np.float64)
        return float(np.interp(x, axis_values, np.arange(len(axis_values))))
    return float(x)


def _curve_x_values(data: SpectralCurveData, *, x_mode: str) -> np.ndarray:
    if x_mode not in _X_MODES:
        raise ValueError(f"Unsupported X axis mode: {x_mode!r}. Expected one of {_X_MODES}.")
    if x_mode == _X_MODE_WAVELENGTH:
        if data.wavelengths is None:
            raise ValueError(
                f"Wavelength information is unavailable for {data.benchmark}; "
                f"switch the X axis back to {_X_MODE_BAND!r}."
            )
        return np.asarray(data.wavelengths, dtype=np.float64)
    return np.arange(data.band_count, dtype=np.float64)


def write_curve_csv(
    data: SpectralCurveData,
    path: Path,
    *,
    x_mode: str = _X_MODE_BAND,
) -> Path:
    """Write the pixel's spectral vectors as CSV with one column per source."""

    x_values = _curve_x_values(data, x_mode=x_mode)
    header = [_CSV_X_HEADERS[x_mode], "GT", *data.methods]
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index in range(data.band_count):
            if x_mode == _X_MODE_WAVELENGTH:
                x_text = f"{float(x_values[index]):.2f}"
            else:
                x_text = str(index)
            row = [x_text, f"{float(data.gt_curve[index]):.6f}"]
            row.extend(f"{float(curve[index]):.6f}" for curve in data.curves.values())
            writer.writerow(row)
    return path


def render_curve_figure(
    data: SpectralCurveData,
    *,
    output_directory: Path,
    output_format: str,
    dpi: int,
    x_mode: str = _X_MODE_BAND,
) -> Path:
    """Render the curve comparison figure and save it; return its path."""

    if dpi < 1:
        raise ValueError("dpi must be a positive integer.")
    normalized_format = _FORMAT_SUFFIXES.get(output_format.casefold())
    if normalized_format is None:
        raise ValueError(f"Unsupported output format: {output_format!r}.")
    figure = create_curve_figure(data, x_mode=x_mode)
    output_directory = output_directory.expanduser()
    path = output_directory / (
        f"spectral_curve_{data.benchmark}-sample{data.sample_index}"
        f"_r{data.row}c{data.column}.{normalized_format}"
    )
    figure.savefig(path, dpi=dpi)
    return path


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------


class _GuiTaskRunner:
    """Runs background loading work for SpectralCurveGui on daemon threads."""

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
            raise ValueError("A task is already running.")
        self._busy.set()
        self._set_buttons_state("disabled")

        def worker() -> None:
            try:
                result = work()
            except (FileNotFoundError, IndexError, KeyError, OSError, ValueError) as error:
                self._root.after(0, on_error, str(error))
                return
            finally:
                self._busy.clear()
                self._root.after(0, self._set_buttons_state, "normal")
            self._root.after(0, on_done, result)

        threading.Thread(target=worker, daemon=True).start()


class SpectralCurveGui:
    """Tkinter front end for per-pixel spectral curve comparison."""

    _CANVAS_DPI = 100
    _PREVIEW_PANEL = "preview"
    _CURVE_PANEL = "curve"

    def __init__(
        self,
        *,
        output_root: Path,
        dataset_root: Path,
        config_root: Path | None = None,
    ) -> None:
        try:
            import tkinter as tk
            from tkinter import filedialog, messagebox, ttk
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.widgets import RectangleSelector
        except ImportError as error:  # pragma: no cover - platform-owned dependency
            raise RuntimeError("The spectral curve GUI requires Tkinter.") from error

        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.figure_canvas_type = FigureCanvasTkAgg
        self.rectangle_selector_type = RectangleSelector
        self.config_root = config_root if config_root is not None else _DEFAULT_CONFIG_ROOT

        self.root = tk.Tk()
        self.root.title("HSMSFusion Spectral Curve")
        self.root.geometry("1280x860")
        self.root.minsize(1080, 700)

        self.output_root = tk.StringVar(value=str(output_root))
        self.dataset_root = tk.StringVar(value=str(dataset_root))
        self.dataset_name = tk.StringVar()
        self.target_path = tk.StringVar()
        self.sample_index = tk.IntVar(value=0)
        self.row_var = tk.IntVar(value=0)
        self.column_var = tk.IntVar(value=0)
        self.rgb_bands = tk.StringVar()
        self.contrast_stretch = tk.BooleanVar(value=True)
        self.x_mode = tk.StringVar(value=_X_MODE_BAND)
        self.export_format = tk.StringVar(value="png")
        self.export_dpi = tk.IntVar(value=300)
        self.status = tk.StringVar(value="Scan the output directory, then load a sample.")

        self.candidates: dict[str, list[tuple[str, Path]]] = {}
        self.method_variables: dict[str, object] = {}
        self.highlight_variables: dict[str, object] = {}
        self._method_state: dict[str, bool] = {}
        self._highlight_state: dict[str, bool] = {}
        self.zoom_range: tuple[float, float, float, float] | None = None
        self.loaded: _LoadedSample | None = None
        self.current_data: SpectralCurveData | None = None
        self._container: dict[str, object] = {}
        self._placeholder: dict[str, object] = {}
        self._selectors: dict[str, object] = {}
        self._placeholder_text = {
            self._PREVIEW_PANEL: "Load a sample to preview the ground truth.",
            self._CURVE_PANEL: "Click the preview to plot spectral curves.",
        }
        self._canvas: dict[str, object] = {}
        self._figure: dict[str, Figure] = {}
        self._task_runner = _GuiTaskRunner(self.root)

        self._build_layout()
        self._task_runner.register_button(self.load_button)
        self._task_runner.register_button(self.save_figure_button)
        self._task_runner.register_button(self.save_csv_button)
        self.scan_outputs()

    # -- Layout ---------------------------------------------------------------

    def _build_layout(self) -> None:
        controls = self.ttk.Frame(self.root, padding=10)
        controls.pack(fill="x")
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(4, weight=1)

        self._build_path_row(controls)
        self._build_sample_row(controls)
        self._build_method_row(controls)

        body = self.ttk.Frame(self.root)
        body.pack(fill="both", expand=True, padx=10)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)
        preview_frame = self.ttk.LabelFrame(
            body, text="Ground truth preview - click a pixel to plot its spectral curve", padding=6
        )
        preview_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        curve_frame = self.ttk.LabelFrame(body, text="Spectral curve comparison", padding=6)
        curve_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        for panel, frame in ((self._PREVIEW_PANEL, preview_frame), (self._CURVE_PANEL, curve_frame)):
            container = self.ttk.Frame(frame)
            container.pack(fill="both", expand=True)
            self._container[panel] = container
            container.bind("<Configure>", lambda _event, name=panel: self._fit_figure(name))
            self._show_placeholder(panel)

        self._build_export_row()
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

    def _build_sample_row(self, controls: object) -> None:
        self.ttk.Label(controls, text="Dataset-scale").grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.dataset_combo = self.ttk.Combobox(
            controls, textvariable=self.dataset_name, state="readonly", width=25
        )
        self.dataset_combo.grid(row=1, column=1, sticky="w", pady=(10, 0))
        self.dataset_combo.bind("<<ComboboxSelected>>", lambda _event: self._on_dataset_changed())
        self.ttk.Label(controls, text="Target HRHS").grid(
            row=1, column=4, sticky="w", padx=(18, 6), pady=(10, 0)
        )
        self.ttk.Entry(controls, textvariable=self.target_path).grid(
            row=1, column=5, sticky="ew", pady=(10, 0)
        )
        self.ttk.Button(controls, text="Browse", command=self._browse_target).grid(
            row=1, column=6, sticky="w", padx=6, pady=(10, 0)
        )
        self.ttk.Label(controls, text="Sample").grid(row=1, column=7, sticky="w", padx=(12, 4), pady=(10, 0))
        self.sample_spin = self.ttk.Spinbox(
            controls, from_=0, to=0, textvariable=self.sample_index, width=5
        )
        self.sample_spin.grid(row=1, column=8, sticky="w", pady=(10, 0))
        self.load_button = self.ttk.Button(controls, text="Load sample", command=self.load_sample)
        self.load_button.grid(row=1, column=9, sticky="w", padx=(8, 0), pady=(10, 0))

    def _build_method_row(self, controls: object) -> None:
        method_frame = self.ttk.LabelFrame(
            controls, text="Fusion methods (right checkbox = highlight)", padding=6
        )
        method_frame.grid(row=2, column=0, columnspan=7, sticky="ew", pady=(10, 0))
        button_row = self.ttk.Frame(method_frame)
        button_row.pack(fill="x")
        self.ttk.Button(button_row, text="All", width=6, command=lambda: self._set_all_methods(True)).pack(
            side="left"
        )
        self.ttk.Button(button_row, text="None", width=6, command=lambda: self._set_all_methods(False)).pack(
            side="left", padx=(4, 0)
        )
        canvas = self.tk.Canvas(method_frame, height=118, highlightthickness=0)
        scrollbar = self.ttk.Scrollbar(method_frame, orient="vertical", command=canvas.yview)
        self.method_inner = self.ttk.Frame(canvas)
        self.method_inner.bind(
            "<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.method_inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(fill="both", expand=True, pady=(4, 0))
        scrollbar.pack(side="right", fill="y")

        options = self.ttk.LabelFrame(controls, text="Display options", padding=6)
        options.grid(row=2, column=7, columnspan=3, sticky="ew", padx=(10, 0), pady=(10, 0))
        options.columnconfigure(1, weight=1)
        self.ttk.Label(options, text="X axis").grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.x_mode_combo = self.ttk.Combobox(
            options, textvariable=self.x_mode, state="readonly", values=list(_X_MODES), width=16
        )
        self.x_mode_combo.grid(row=0, column=1, sticky="w")
        self.x_mode_combo.bind("<<ComboboxSelected>>", lambda _event: self._on_x_mode_changed())
        self.ttk.Label(options, text="RGB bands").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=(6, 0))
        self.ttk.Entry(options, textvariable=self.rgb_bands, width=12).grid(
            row=1, column=1, sticky="w", pady=(6, 0)
        )
        self.ttk.Checkbutton(
            options, text="Contrast stretch", variable=self.contrast_stretch,
            command=self._on_preview_settings_changed,
        ).grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.ttk.Button(options, text="Clear zoom", command=self.clear_zoom).grid(
            row=2, column=1, sticky="w", pady=(6, 0)
        )

    def _build_export_row(self) -> None:
        export = self.ttk.Frame(self.root, padding=(10, 6))
        export.pack(fill="x")
        self.ttk.Label(export, text="Figure format").pack(side="left")
        self.ttk.Combobox(
            export, textvariable=self.export_format, state="readonly",
            values=["png", "pdf", "svg", "jpg"], width=5,
        ).pack(side="left", padx=(6, 0))
        self.ttk.Label(export, text="DPI").pack(side="left", padx=(12, 0))
        self.ttk.Entry(export, textvariable=self.export_dpi, width=6).pack(side="left", padx=(6, 0))
        self.save_figure_button = self.ttk.Button(export, text="Save figure", command=self._save_figure)
        self.save_figure_button.pack(side="left", padx=(12, 0))
        self.save_csv_button = self.ttk.Button(export, text="Save CSV", command=self._save_csv)
        self.save_csv_button.pack(side="left", padx=(6, 0))

    # -- File dialogs ---------------------------------------------------------

    def _browse_output_root(self) -> None:
        selected = self.filedialog.askdirectory(parent=self.root, title="Select the output root")
        if selected:
            self.output_root.set(selected)
            self.scan_outputs()

    def _browse_dataset_root(self) -> None:
        selected = self.filedialog.askdirectory(parent=self.root, title="Select the dataset root")
        if selected:
            self.dataset_root.set(selected)
            self._on_dataset_changed()

    def _browse_target(self) -> None:
        selected = self.filedialog.askopenfilename(
            parent=self.root,
            title="Select test.h5 containing HRHS",
            filetypes=(("HDF5", "*.h5"), ("All files", "*.*")),
        )
        if selected:
            self.target_path.set(selected)

    # -- Discovery and sample loading ------------------------------------------

    def scan_outputs(self) -> None:
        """Rediscover ``predictions.h5`` exports under the output root."""

        self.candidates = discover_output_predictions(Path(self.output_root.get()))
        benchmarks = list(self.candidates)
        self.dataset_combo.configure(values=benchmarks)
        if not benchmarks:
            self.dataset_name.set("")
            self._on_dataset_changed()
            self.status.set("No predictions.h5 found under the output root. Adjust the path and click Scan.")
            return
        if self.dataset_name.get() not in benchmarks:
            self.dataset_name.set(benchmarks[0])
        self._on_dataset_changed()
        method_total = sum(len(methods) for methods in self.candidates.values())
        self.status.set(
            f"Found {method_total} method export(s) across {len(benchmarks)} dataset-scale folder(s)."
        )

    def _on_dataset_changed(self) -> None:
        benchmark = self.dataset_name.get().strip()
        self.loaded = None
        self.current_data = None
        self.sample_index.set(0)
        self.sample_spin.configure(to=0)
        self._show_placeholder(self._PREVIEW_PANEL)
        self._show_placeholder(self._CURVE_PANEL)
        self.zoom_range = None
        self._rebuild_method_checkboxes(
            [name for name, _path in self.candidates.get(benchmark, [])]
        )
        if not benchmark:
            return
        inferred = infer_target_path(Path(self.dataset_root.get()), benchmark)
        self.target_path.set(str(inferred))
        self.status.set(f"Dataset-scale {benchmark}: click 'Load sample' to read curves.")

    def _set_all_methods(self, selected: bool) -> None:
        for variable in self.method_variables.values():
            variable.set(selected)
        self._on_methods_changed()

    def _selected_methods(self) -> list[str]:
        return [name for name, variable in self.method_variables.items() if variable.get()]

    def _highlighted_methods(self) -> list[str]:
        return [name for name, variable in self.highlight_variables.items() if variable.get()]

    def _rebuild_method_checkboxes(self, method_names: Sequence[str]) -> None:
        # Persist the checkbox states so they survive switching datasets.
        for method, variable in self.method_variables.items():
            self._method_state[method] = bool(variable.get())
        for method, variable in self.highlight_variables.items():
            self._highlight_state[method] = bool(variable.get())
        for child in self.method_inner.winfo_children():
            child.destroy()
        self.method_variables = {}
        self.highlight_variables = {}
        self.ttk.Label(self.method_inner, text="Method").grid(row=0, column=0, sticky="w", padx=4)
        self.ttk.Label(self.method_inner, text="Highlight").grid(row=0, column=1, sticky="w")
        for index, name in enumerate(method_names, start=1):
            selected = self.tk.BooleanVar(value=self._method_state.get(name, True))
            highlighted = self.tk.BooleanVar(value=self._highlight_state.get(name, False))
            self.method_variables[name] = selected
            self.highlight_variables[name] = highlighted
            self.ttk.Checkbutton(
                self.method_inner, text=name, variable=selected,
                command=self._on_methods_changed,
            ).grid(row=index, column=0, sticky="w", padx=4, pady=1)
            self.ttk.Checkbutton(
                self.method_inner, variable=highlighted, command=self._on_highlight_changed
            ).grid(row=index, column=1, sticky="w", padx=4, pady=1)

    def load_sample(self) -> None:
        """Read the ground-truth cube and selected method curves in background."""

        if not self.dataset_name.get().strip():
            self.status.set("Select a dataset-scale output folder first (click Scan).")
            return
        try:
            sample_index = int(self.sample_index.get())
        except (TypeError, ValueError, self.tk.TclError):
            self.status.set("Sample index must be an integer.")
            return
        settings = {
            "output_root": Path(self.output_root.get()),
            "dataset_root": Path(self.dataset_root.get()),
            "benchmark": self.dataset_name.get().strip(),
            "target_override": self.target_path.get().strip(),
            "sample_index": sample_index,
        }
        if not self._selected_methods():
            self.status.set("Select at least one fusion method to compare.")
            return
        self._task_runner.run(
            lambda: self._load_sample_work(settings), self._on_sample_loaded, self._on_load_error
        )

    def _load_sample_work(self, settings: dict[str, object]) -> _LoadedSample:
        discovered = discover_output_predictions(settings["output_root"])
        benchmark, candidates = resolve_benchmark(discovered, settings["benchmark"])
        if not candidates:
            raise ValueError(f"No method exports found for {benchmark}.")
        override = str(settings["target_override"])
        target = Path(override).expanduser() if override else infer_target_path(
            settings["dataset_root"], benchmark
        )
        sample_index = int(settings["sample_index"])
        gt_cube = read_hsi_sample(target, "HRHS", sample_index)
        wavelengths = resolve_wavelength_axis(self.config_root, benchmark, int(gt_cube.shape[0]))
        count = sample_count(target, "HRHS")
        # Cache every discovered method; the curve refresh filters by the
        # current checkbutton state so toggling a method needs no reload.
        return _LoadedSample(
            benchmark=benchmark,
            target_path=target,
            sample_index=sample_index,
            predictions=candidates,
            gt_cube=gt_cube,
            sample_count=count,
            wavelengths=wavelengths,
        )

    def _on_sample_loaded(self, loaded: object) -> None:
        assert isinstance(loaded, _LoadedSample)
        self.loaded = loaded
        self.current_data = None
        self.zoom_range = None
        self.sample_spin.configure(to=max(loaded.sample_count - 1, 0))
        bands, height, width = (int(v) for v in loaded.gt_cube.shape)
        self.row_var.set(min(max(self.row_var.get(), 0), height - 1))
        self.column_var.set(min(max(self.column_var.get(), 0), width - 1))
        scale_note = "" if loaded.wavelengths is not None else "; no wavelength config found"
        self.status.set(
            f"Loaded {loaded.benchmark} sample {loaded.sample_index} "
            f"({bands} bands, {height}x{width}, {len(loaded.predictions)} method(s)"
            f"{scale_note}). Click the preview to pick a pixel."
        )
        self._on_pixel_changed()

    def _on_load_error(self, error: str) -> None:
        self.status.set(f"Loading failed: {error}")

    # -- Preview and curve refresh ----------------------------------------------

    def _current_pixel(self) -> tuple[int, int] | None:
        try:
            row = int(self.row_var.get())
            column = int(self.column_var.get())
        except (TypeError, ValueError, self.tk.TclError):
            return None
        return row, column

    def _on_preview_click(self, event: object) -> None:
        if self.loaded is None:
            return
        if getattr(event, "inaxes", None) is None:
            return
        x_data, y_data = getattr(event, "xdata", None), getattr(event, "ydata", None)
        if x_data is None or y_data is None:
            return
        _, height, width = (int(v) for v in self.loaded.gt_cube.shape)
        column = min(max(int(round(float(x_data))), 0), width - 1)
        row = min(max(int(round(float(y_data))), 0), height - 1)
        self.row_var.set(row)
        self.column_var.set(column)
        self._on_pixel_changed()

    def _on_preview_settings_changed(self) -> None:
        if self.loaded is None:
            return
        self._refresh_preview()

    def _on_x_mode_changed(self) -> None:
        if self.current_data is None:
            return
        self._refresh_curve_figure()

    def _on_highlight_changed(self) -> None:
        if self.current_data is None:
            return
        self._refresh_curve_figure()

    def clear_zoom(self) -> None:
        """Remove the zoom inset from the curve panel."""

        if self.zoom_range is None:
            self.status.set("No zoom inset is active.")
            return
        self.zoom_range = None
        if self.current_data is not None:
            self._refresh_curve_figure()
        self.status.set("Zoom inset cleared.")

    def _on_curve_zoom_selected(self, click_event: object, release_event: object) -> None:
        """Store the dragged rectangle (in band units) and redraw the figure."""

        if self.current_data is None:
            return
        x0 = getattr(click_event, "xdata", None)
        y0 = getattr(click_event, "ydata", None)
        x1 = getattr(release_event, "xdata", None)
        y1 = getattr(release_event, "ydata", None)
        if x0 is None or y0 is None or x1 is None or y1 is None:
            return
        mode = self.x_mode.get()
        wavelengths = self.current_data.wavelengths if mode == _X_MODE_WAVELENGTH else None
        try:
            band0 = _display_x_to_band(float(x0), wavelengths, mode)
            band1 = _display_x_to_band(float(x1), wavelengths, mode)
        except ValueError as error:
            self.status.set(str(error))
            return
        band0, band1 = sorted((band0, band1))
        band0 = min(max(band0, 0.0), float(self.current_data.band_count - 1))
        band1 = min(max(band1, 0.0), float(self.current_data.band_count - 1))
        y0, y1 = sorted((float(y0), float(y1)))
        if band1 - band0 < 0.5 or y1 - y0 <= 0.0:
            self.status.set("The selected zoom region is too small; drag a larger rectangle.")
            return
        self.zoom_range = (band0, band1, y0, y1)
        self._refresh_curve_figure()
        self.status.set(
            f"Zoom inset added. Drag again to replace it, or click 'Clear zoom' to remove it."
        )

    def _zoom_display_range(self) -> tuple[float, float, float, float] | None:
        """The stored zoom rectangle converted to the current axis units."""

        if self.zoom_range is None or self.current_data is None:
            return None
        band0, band1, y0, y1 = self.zoom_range
        mode = self.x_mode.get()
        wavelengths = self.current_data.wavelengths if mode == _X_MODE_WAVELENGTH else None
        x0 = _band_to_display_x(band0, wavelengths, mode)
        x1 = _band_to_display_x(band1, wavelengths, mode)
        return (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))

    def _on_pixel_changed(self) -> None:
        """Refresh the preview marker and the curve figure for the new pixel."""

        if self.loaded is None:
            return
        pixel = self._current_pixel()
        if pixel is None:
            self.status.set("Row and column must be integers.")
            return
        row, column = pixel
        _, height, width = (int(v) for v in self.loaded.gt_cube.shape)
        if not 0 <= row < height or not 0 <= column < width:
            self.status.set(
                f"Pixel out of range: row must be 0..{height - 1} and column 0..{width - 1}."
            )
            return
        try:
            self._refresh_preview()
        except ValueError as error:
            self.status.set(f"Preview update failed: {error}")
            return
        try:
            self._refresh_curves(row, column)
        except (FileNotFoundError, IndexError, KeyError, OSError, ValueError) as error:
            self.status.set(f"Curve update failed: {error}")
            return
        assert self.current_data is not None
        self.status.set(
            f"Pixel (row {row}, column {column}): plotted the ground truth and "
            f"{len(self.current_data.curves)} method curve(s)."
        )

    def _refresh_preview(self) -> None:
        loaded = self.loaded
        if loaded is None:
            return
        pixel = self._current_pixel()
        if pixel is None:
            return
        row, column = pixel
        bands = _parse_rgb_bands(self.rgb_bands.get())
        figure = create_preview_figure(
            loaded.gt_cube,
            rgb_bands=bands,
            row=row,
            column=column,
            contrast_stretch=bool(self.contrast_stretch.get()),
        )
        self._replace_figure(self._PREVIEW_PANEL, figure, on_click=self._on_preview_click)

    def _on_methods_changed(self) -> None:
        if self.loaded is None or self.current_data is None:
            return
        pixel = self._current_pixel()
        if pixel is None:
            return
        try:
            self._refresh_curves(*pixel)
        except (FileNotFoundError, IndexError, KeyError, OSError, ValueError) as error:
            self.status.set(f"Curve update failed: {error}")

    def _refresh_curves(self, row: int, column: int) -> None:
        loaded = self.loaded
        if loaded is None:
            return
        selected = set(self._selected_methods())
        targets = [entry for entry in loaded.predictions if entry[0] in selected]
        if not targets:
            raise ValueError("Select at least one fusion method to compare.")
        gt_curve = extract_spectral_curve(loaded.gt_cube, row=row, column=column)
        bands, height, width = (int(v) for v in loaded.gt_cube.shape)
        curves = {
            name: _read_prediction_curve(
                path,
                dataset_key="prediction",
                sample_index=loaded.sample_index,
                row=row,
                column=column,
                expected_shape=(bands, height, width),
            )
            for name, path in targets
        }
        self.current_data = SpectralCurveData(
            benchmark=loaded.benchmark,
            target_path=loaded.target_path,
            sample_index=loaded.sample_index,
            row=row,
            column=column,
            gt_curve=gt_curve,
            curves=curves,
            wavelengths=loaded.wavelengths,
        )
        self._refresh_curve_figure()

    def _refresh_curve_figure(self) -> None:
        data = self.current_data
        if data is None:
            return
        try:
            zoom = self._zoom_display_range()
        except ValueError as error:  # pragma: no cover - defensive
            self.status.set(f"Zoom inset removed: {error}")
            self.zoom_range = None
            zoom = None
        try:
            figure = create_curve_figure(
                data,
                x_mode=self.x_mode.get(),
                figsize=self._panel_figsize(self._CURVE_PANEL),
                highlight_methods=self._highlighted_methods(),
                zoom=zoom,
            )
        except ValueError as error:
            self.status.set(str(error))
            self.x_mode.set(_X_MODE_BAND)
            figure = create_curve_figure(
                data,
                x_mode=_X_MODE_BAND,
                figsize=self._panel_figsize(self._CURVE_PANEL),
                highlight_methods=self._highlighted_methods(),
                zoom=zoom,
            )
        self._replace_figure(self._CURVE_PANEL, figure, on_select=self._on_curve_zoom_selected)

    # -- Canvas management -------------------------------------------------------

    def _show_placeholder(self, panel: str) -> None:
        selector = self._selectors.pop(panel, None)
        if selector is not None:
            selector.set_active(False)
        canvas = self._canvas.pop(panel, None)
        if canvas is not None:
            canvas.get_tk_widget().destroy()
        figure = self._figure.pop(panel, None)
        if figure is not None:
            figure.clear()
        if panel not in self._placeholder or self._placeholder[panel] is None:
            placeholder = self.ttk.Label(
                self._container[panel], text=self._placeholder_text[panel], anchor="center"
            )
            placeholder.pack(fill="both", expand=True)
            self._placeholder[panel] = placeholder

    def _replace_figure(
        self,
        panel: str,
        figure: Figure,
        on_click: Callable | None = None,
        on_select: Callable | None = None,
    ) -> None:
        placeholder = self._placeholder.pop(panel, None)
        if placeholder is not None:
            placeholder.destroy()
        selector = self._selectors.pop(panel, None)
        if selector is not None:
            selector.set_active(False)
        canvas = self._canvas.pop(panel, None)
        if canvas is not None:
            canvas.get_tk_widget().destroy()
        old_figure = self._figure.pop(panel, None)
        if old_figure is not None:
            old_figure.clear()
        new_canvas = self.figure_canvas_type(figure, master=self._container[panel])
        new_canvas.draw()
        new_canvas.get_tk_widget().pack(fill="both", expand=True)
        if on_click is not None:
            new_canvas.mpl_connect("button_press_event", on_click)
        if on_select is not None:
            self._selectors[panel] = self.rectangle_selector_type(
                figure.axes[0], on_select, useblit=False, button=[1],
                minspanx=0, minspany=0, spancoords="data", props=dict(_SELECTOR_STYLE),
            )
        self._canvas[panel] = new_canvas
        self._figure[panel] = figure

    def _panel_figsize(self, panel: str) -> tuple[float, float] | None:
        """Match a figure to the current container pixels; None before mapping."""

        container = self._container[panel]
        container.update_idletasks()
        width = container.winfo_width()
        height = container.winfo_height()
        if width < 80 or height < 80:
            return None
        return (width / self._CANVAS_DPI, height / self._CANVAS_DPI)

    def _fit_figure(self, panel: str) -> None:
        """Re-fit a figure to its container whenever the window changes size."""

        canvas = self._canvas.get(panel)
        figure = self._figure.get(panel)
        if canvas is None or figure is None:
            return
        container = self._container[panel]
        width = container.winfo_width()
        height = container.winfo_height()
        if width < 80 or height < 80:
            return
        figure.set_size_inches(width / self._CANVAS_DPI, height / self._CANVAS_DPI)
        canvas.draw_idle()

    # -- Export -------------------------------------------------------------------

    def _curve_stem(self) -> str:
        data = self.current_data
        assert data is not None
        return (
            f"spectral_curve_{data.benchmark}-sample{data.sample_index}"
            f"_r{data.row}c{data.column}"
        )

    def _save_figure(self) -> None:
        if self.current_data is None or self._CURVE_PANEL not in self._figure:
            self.status.set("Load a sample and pick a pixel before saving a figure.")
            return
        suffix = _FORMAT_SUFFIXES.get(self.export_format.get().casefold(), "png")
        path_name = self.filedialog.asksaveasfilename(
            parent=self.root,
            title="Save spectral curve figure",
            defaultextension=f".{suffix}",
            initialfile=f"{self._curve_stem()}.{suffix}",
            filetypes=(("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg"), ("All files", "*.*")),
        )
        if not path_name:
            return
        try:
            dpi = int(self.export_dpi.get())
        except (TypeError, ValueError, self.tk.TclError):
            self.status.set("DPI must be a positive integer.")
            return
        try:
            self._figure[self._CURVE_PANEL].savefig(path_name, dpi=dpi)
        except (OSError, ValueError) as error:
            self.status.set(f"Figure export failed: {error}")
            self.messagebox.showerror("Figure export failed", str(error))
            return
        self.status.set(f"Saved figure to {path_name}")

    def _save_csv(self) -> None:
        if self.current_data is None:
            self.status.set("Load a sample and pick a pixel before saving a CSV.")
            return
        path_name = self.filedialog.asksaveasfilename(
            parent=self.root,
            title="Save spectral curve CSV",
            defaultextension=".csv",
            initialfile=f"{self._curve_stem()}.csv",
            filetypes=(("CSV", "*.csv"), ("All files", "*.*")),
        )
        if not path_name:
            return
        try:
            written = write_curve_csv(self.current_data, Path(path_name), x_mode=self.x_mode.get())
        except (OSError, ValueError) as error:
            self.status.set(f"CSV export failed: {error}")
            self.messagebox.showerror("CSV export failed", str(error))
            return
        self.status.set(f"Saved CSV to {written}")

    # -- Entry ----------------------------------------------------------------------

    def run(self) -> None:
        self.root.mainloop()


def launch_gui(
    *,
    output_root: Path,
    dataset_root: Path,
    config_root: Path | None = None,
) -> None:
    """Launch the interactive spectral curve comparison tool."""

    matplotlib.use("TkAgg")
    SpectralCurveGui(output_root=output_root, dataset_root=dataset_root, config_root=config_root).run()


def main() -> int:
    launch_gui(output_root=Path("output"), dataset_root=Path("Dataset"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
