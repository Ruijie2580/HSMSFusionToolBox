"""GUI for browsing and comparing TensorBoard training records.

The viewer searches ``runs/`` (or ``run/``) recursively for TensorBoard event
files.  It supports the run layout produced by this project, including both
``runs/<experiment>/logs/tensorboard`` and timestamped legacy records such as
``runs/<experiment>/<timestamp>/logs/tensorboard``.

Run it from the repository root::

    python scripts/visualize_tensorboard_runs.py

Window layout and interaction flow
----------------------------------
* **Top bar** - experiment directory (Browse/Refresh) and an optional dataset
  filter.
* **Left sidebar**, top to bottom in the order a user works: pick
  experiments (grouped by dataset; selecting a group selects all of its
  runs), choose what to plot (three mutually exclusive plot modes, each
  showing only the curve tags that belong to that mode), adjust the
  appearance (epoch/iter x axis, smoothing, stride sampling, log axes,
  grid, metric normalization), and export the current 4:3 figure.
* **Right side** - the 4:3 plot canvas with the Matplotlib navigation
  toolbar; the canvas grows with the window while keeping its aspect ratio.
* **Bottom** - a status bar reporting the outcome of the last action.

Every control redraws the plot immediately; there is no separate apply step.
"""

from __future__ import annotations

import argparse
import math
import re
from bisect import bisect_left
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

EVENT_FILE_PREFIX = "events.out.tfevents."
MODE_TRAINING_LOSS = "Training Loss"
MODE_VALIDATION_LOSS = "Validation Loss"
MODE_VALIDATION_METRIC = "Validation Metric"
PLOT_MODES = (MODE_TRAINING_LOSS, MODE_VALIDATION_LOSS, MODE_VALIDATION_METRIC)
X_AXIS_CHOICES = ("epoch", "iter")
SERIES_LIST_TITLES = {
    MODE_TRAINING_LOSS: "Training loss curves (multi-select):",
    MODE_VALIDATION_LOSS: "Validation loss curves (multi-select):",
    MODE_VALIDATION_METRIC: "Validation metric (single-select):",
}
EXPORT_FORMATS = ("png", "jpeg", "pdf")
MAX_PLOT_POINTS = 5000
FIGURE_DPI = 100
FIGURE_SIZE_INCHES = (8, 6)
PLOT_SIZE_PIXELS = (800, 600)
MIN_PLOT_PIXELS = (320, 240)
PLOT_ASPECT_RATIO = 4 / 3
SIDEBAR_WIDTH_PIXELS = 360
ALL_DATASETS = "All datasets"
UNKNOWN_DATASET = "Unknown"
# DejaVu Sans is bundled with Matplotlib, so the chart remains sans-serif and
# renders consistently across Windows environments without font substitution.
PLOT_FONT_FAMILY = "DejaVu Sans"
METRIC_HINTS = (
    "psnr",
    "ssim",
    "sam",
    "ergas",
    "rmse",
    "uiqi",
    "metric",
    "accuracy",
    "acc",
)


@dataclass(frozen=True)
class ScalarPoint:
    """One TensorBoard scalar sample."""

    step: int
    wall_time: float
    value: float


@dataclass
class TensorBoardRun:
    """Scalar data loaded from one TensorBoard log directory."""

    name: str
    log_directory: Path
    scalars: dict[str, list[ScalarPoint]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def tags(self) -> tuple[str, ...]:
        return tuple(sorted(self.scalars))


def _event_files(log_directory: Path) -> list[Path]:
    return sorted(
        path
        for path in log_directory.glob(f"{EVENT_FILE_PREFIX}*")
        if path.is_file()
    )


def _run_name(log_directory: Path, runs_root: Path) -> str:
    """Return a readable name, removing the conventional log suffix."""

    relative = log_directory.relative_to(runs_root)
    parts = list(relative.parts)
    if len(parts) >= 2 and parts[-2:] == ["logs", "tensorboard"]:
        parts = parts[:-2]
    return "/".join(parts) or log_directory.name


def dataset_name_from_run_name(run_name: str) -> str:
    """Extract the dataset from ``model_dataset_xN`` run names.

    Timestamped records keep the experiment name before the first ``/``;
    names that do not follow the standard recipe are grouped as ``Unknown``.
    """

    experiment_name = run_name.replace("\\", "/").split("/", maxsplit=1)[0]
    match = re.fullmatch(r"[^_]+_(?P<dataset>.+)_x\d+", experiment_name, re.IGNORECASE)
    return match.group("dataset") if match else UNKNOWN_DATASET


def filter_runs_by_dataset(
    runs: Iterable[TensorBoardRun],
    selected_dataset: str,
) -> list[TensorBoardRun]:
    """Return all runs or only runs belonging to ``selected_dataset``."""

    run_list = list(runs)
    if selected_dataset in {"", ALL_DATASETS}:
        return run_list
    return [
        run
        for run in run_list
        if dataset_name_from_run_name(run.name) == selected_dataset
    ]


def discover_tensorboard_runs(runs_root: Path) -> list[TensorBoardRun]:
    """Find and load every distinct TensorBoard log directory below ``runs_root``."""

    runs_root = runs_root.expanduser().resolve()
    if not runs_root.is_dir():
        raise FileNotFoundError(f"Experiment directory does not exist: {runs_root}")

    log_directories = sorted(
        {event_file.parent for event_file in runs_root.rglob(f"{EVENT_FILE_PREFIX}*")}
    )
    return [
        load_tensorboard_run(
            log_directory,
            name=_run_name(log_directory, runs_root),
        )
        for log_directory in log_directories
    ]


def load_tensorboard_run(log_directory: Path, *, name: str | None = None) -> TensorBoardRun:
    """Load scalar events from all event files in one TensorBoard directory."""

    log_directory = log_directory.expanduser().resolve()
    run = TensorBoardRun(name=name or log_directory.name, log_directory=log_directory)
    event_files = _event_files(log_directory)
    if not event_files:
        run.errors.append(f"No event files found: {log_directory}")
        return run

    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError as exc:  # pragma: no cover - depends on the user's environment
        raise RuntimeError(
            "Missing TensorBoard dependency; install with: pip install tensorboard"
        ) from exc

    for event_file in event_files:
        try:
            accumulator = EventAccumulator(
                str(event_file),
                size_guidance={"scalars": 0},
            )
            accumulator.Reload()
            for tag in accumulator.Tags().get("scalars", []):
                points = run.scalars.setdefault(tag, [])
                points.extend(
                    ScalarPoint(
                        step=int(event.step),
                        wall_time=float(event.wall_time),
                        value=float(event.value),
                    )
                    for event in accumulator.Scalars(tag)
                    if _is_finite(float(event.value))
                )
        except Exception as exc:  # noqa: BLE001 - keep other experiments usable if one file is partial
            run.errors.append(f"{event_file.name}: {exc}")

    for tag, points in run.scalars.items():
        run.scalars[tag] = deduplicate_scalar_points(points)
    return run


def _is_finite(value: float) -> bool:
    return math.isfinite(value)


def is_validation_metric_tag(tag: str) -> bool:
    """Classify common validation metrics while remaining useful for custom tags."""

    lowered = tag.casefold()
    return lowered.startswith("val/") and any(hint in lowered for hint in METRIC_HINTS)


def deduplicate_scalar_points(points: Iterable[ScalarPoint]) -> list[ScalarPoint]:
    """Keep the newest wall-time sample for each TensorBoard step."""

    latest_by_step: dict[int, ScalarPoint] = {}
    for point in points:
        previous = latest_by_step.get(point.step)
        if previous is None or point.wall_time >= previous.wall_time:
            latest_by_step[point.step] = point
    return sorted(latest_by_step.values(), key=lambda point: (point.step, point.wall_time))


def loss_tags(tags: Iterable[str]) -> list[str]:
    """Return train/validation loss tags, excluding metric-like validation tags."""

    return sorted(
        tag
        for tag in set(tags)
        if (tag.casefold().startswith("train/") or tag.casefold().startswith("val/"))
        and tag.casefold() != "epoch"
        and not is_validation_metric_tag(tag)
    )


def train_loss_tags(tags: Iterable[str]) -> list[str]:
    """Return the ``train/*`` loss tags."""

    return [tag for tag in loss_tags(tags) if tag.casefold().startswith("train/")]


def validation_loss_tags(tags: Iterable[str]) -> list[str]:
    """Return the ``val/*`` loss tags, excluding metric-like tags."""

    return [tag for tag in loss_tags(tags) if tag.casefold().startswith("val/")]


def series_tags_for_mode(mode: str, tags: Iterable[str]) -> list[str]:
    """Return the curve tags that belong to one plot mode."""

    if mode not in PLOT_MODES:
        raise ValueError(f"Unknown plot mode: {mode}")
    if mode == MODE_VALIDATION_METRIC:
        return validation_metric_tags(tags)
    if mode == MODE_VALIDATION_LOSS:
        return validation_loss_tags(tags)
    return train_loss_tags(tags)


def default_series_selection(tags: Iterable[str], mode: str) -> list[str]:
    """Pick sensible default curves for one plot mode.

    Loss modes prefer the conventional aggregate (``train/total`` or
    ``val/total``, then ``val/loss``) and otherwise select the first tag of
    the mode; the metric mode prefers PSNR and then SSIM so the headline
    metric is plotted without extra clicks.
    """

    ordered = list(tags)
    if not ordered:
        return []
    if mode == MODE_VALIDATION_METRIC:
        for hint in ("psnr", "ssim"):
            for tag in ordered:
                if hint in tag.casefold():
                    return [tag]
        return ordered[:1]
    prefix = "train/" if mode == MODE_TRAINING_LOSS else "val/"
    preferred = [
        tag
        for tag in ("train/total", "val/total", "val/loss")
        if tag in ordered and tag.casefold().startswith(prefix)
    ]
    if preferred:
        return preferred[:1]
    prefixed = [tag for tag in ordered if tag.casefold().startswith(prefix)]
    return prefixed[:1] or ordered[:1]


def validation_metric_tags(tags: Iterable[str]) -> list[str]:
    """Return validation metric tags, with a fallback for custom ``val/*`` tags."""

    unique_tags = sorted(set(tags))
    metric_tags = [tag for tag in unique_tags if is_validation_metric_tag(tag)]
    if metric_tags:
        return metric_tags
    return [
        tag
        for tag in unique_tags
        if tag.casefold().startswith("val/") and tag.casefold() not in {"val/loss", "val/total"}
    ]


def _interpolate_epoch(step: int, epoch_points: list[ScalarPoint]) -> float:
    """Map a scalar step to the logged epoch, interpolating when necessary."""

    if not epoch_points:
        return float(step)
    steps = [point.step for point in epoch_points]
    index = bisect_left(steps, step)
    if index == 0:
        return epoch_points[0].value
    if index == len(epoch_points):
        return epoch_points[-1].value
    left, right = epoch_points[index - 1], epoch_points[index]
    if right.step == left.step:
        return right.value
    fraction = (step - left.step) / (right.step - left.step)
    return left.value + fraction * (right.value - left.value)


def canonical_x_axis(choice: str) -> str:
    """Map the UI x-axis choice to the internal axis name (``iter`` -> ``step``)."""

    mapping = {"epoch": "epoch", "iter": "step"}
    if choice not in mapping:
        raise ValueError("x axis must be 'epoch' or 'iter'")
    return mapping[choice]


def subsample_series_by_stride(
    x_values: list[float],
    y_values: list[float],
    stride: int,
) -> tuple[list[float], list[float]]:
    """Keep the last sample of every ``stride``-wide x-axis bucket.

    Sampling in x-axis units yields one point per ``stride`` epochs or
    iterations regardless of how densely the runs logged.  A stride of 1 (or
    less) keeps every point, and the final point of the curve is always
    preserved.
    """

    stride = max(1, int(stride))
    if stride == 1 or not x_values:
        return list(x_values), list(y_values)
    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must have equal lengths")
    sampled_x: list[float] = []
    sampled_y: list[float] = []
    current_bucket: int | None = None
    for x_value, y_value in zip(x_values, y_values):
        bucket = math.floor(x_value / stride)
        if bucket == current_bucket and sampled_x:
            sampled_x[-1] = x_value
            sampled_y[-1] = y_value
        else:
            sampled_x.append(x_value)
            sampled_y.append(y_value)
            current_bucket = bucket
    return sampled_x, sampled_y


def series_xy(
    run: TensorBoardRun,
    tag: str,
    *,
    x_axis: str = "epoch",
    smoothing: int = 1,
    sampling_stride: int = 1,
) -> tuple[list[float], list[float]]:
    """Convert a run/tag to plot coordinates with smoothing and stride sampling."""

    points = run.scalars.get(tag, [])
    if not points:
        return [], []
    if x_axis not in {"epoch", "step"}:
        raise ValueError("x_axis must be 'epoch' or 'step'")

    effective_axis = effective_x_axis(run, x_axis)
    epoch_points = run.scalars.get("epoch", [])
    x_values = [
        float(point.step)
        if effective_axis == "step"
        else _interpolate_epoch(point.step, epoch_points)
        for point in points
    ]
    y_values = [point.value for point in points]
    sampled_x, sampled_y = subsample_series_by_stride(
        x_values, smooth_values(y_values, smoothing), sampling_stride
    )
    return downsample_series(sampled_x, sampled_y)


def effective_x_axis(run: TensorBoardRun, requested_axis: str) -> str:
    """Return the actual x-axis source used for one run."""

    if requested_axis not in {"epoch", "step"}:
        raise ValueError("requested_axis must be 'epoch' or 'step'")
    if requested_axis == "epoch" and not run.scalars.get("epoch"):
        return "step"
    return requested_axis


def downsample_series(
    x_values: list[float],
    y_values: list[float],
    max_points: int = MAX_PLOT_POINTS,
) -> tuple[list[float], list[float]]:
    """Uniformly reduce a dense curve while preserving its endpoints."""

    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must have equal lengths")
    if max_points < 2:
        raise ValueError("max_points must be at least 2")
    if len(x_values) <= max_points:
        return list(x_values), list(y_values)
    last_index = len(x_values) - 1
    indices = [round(index * last_index / (max_points - 1)) for index in range(max_points)]
    return [x_values[index] for index in indices], [y_values[index] for index in indices]


def export_figure(figure, target_path: Path, export_format: str, dpi: int) -> Path:
    """Export a Matplotlib figure without changing its configured aspect ratio."""

    export_format = export_format.casefold()
    if export_format not in EXPORT_FORMATS:
        raise ValueError(f"Unsupported export format: {export_format}")
    dpi = max(72, min(1200, int(dpi)))
    figure.savefig(
        target_path,
        format=export_format,
        dpi=dpi,
        bbox_inches=None,
        facecolor="white" if export_format == "jpeg" else None,
    )
    return target_path


def smooth_values(values: list[float], window: int) -> list[float]:
    """Apply a causal moving average without changing the number of points."""

    window = max(1, int(window))
    if window == 1 or len(values) < 2:
        return list(values)
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    cumsum = np.concatenate(([0.0], np.cumsum(arr)))
    indices = np.arange(len(arr))
    starts = np.maximum(0, indices - window + 1)
    counts = np.minimum(indices + 1, window).astype(np.float64)
    smoothed = (cumsum[indices + 1] - cumsum[starts]) / counts
    return smoothed.tolist()


def default_runs_root(project_root: Path | None = None) -> Path:
    """Prefer ``runs`` and fall back to the singular ``run`` spelling."""

    project_root = (project_root or Path(__file__).resolve().parents[1]).expanduser()
    for directory_name in ("runs", "run"):
        candidate = project_root / directory_name
        if candidate.is_dir():
            return candidate
    return project_root / "runs"


class TensorBoardViewer:
    """Tkinter desktop application for interactive TensorBoard comparisons.

    The handlers follow one rule: every ``_on_*`` method funnels its change
    through the ``_reload_runs`` -> ``_populate_run_list`` ->
    ``_populate_series_list`` -> ``_redraw`` pipeline, so selection policy
    lives in exactly one place per stage.
    """

    def __init__(self, root_window, initial_runs_root: Path) -> None:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk

        import matplotlib
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg,
            NavigationToolbar2Tk,
        )
        from matplotlib.figure import Figure

        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.FigureCanvasTkAgg = FigureCanvasTkAgg
        self.NavigationToolbar2Tk = NavigationToolbar2Tk
        matplotlib.rcParams.update(
            {
                "font.family": PLOT_FONT_FAMILY,
                "font.sans-serif": [PLOT_FONT_FAMILY],
                "font.weight": "normal",
                "axes.titleweight": "normal",
                "axes.labelweight": "normal",
                "figure.titleweight": "normal",
                "mathtext.fontset": "dejavusans",
            }
        )
        self.figure = Figure(
            figsize=FIGURE_SIZE_INCHES, dpi=FIGURE_DPI, constrained_layout=True
        )
        self.root_window = root_window

        # One Tk variable per user-facing setting.
        self.runs_root_var = tk.StringVar(value=str(initial_runs_root))
        self.dataset_filter_var = tk.StringVar(value=ALL_DATASETS)
        self.plot_mode_var = tk.StringVar(value=MODE_TRAINING_LOSS)
        self.x_axis_var = tk.StringVar(value="epoch")
        self.smoothing_var = tk.IntVar(value=1)
        self.stride_var = tk.IntVar(value=1)
        self.log_x_var = tk.BooleanVar(value=False)
        self.log_y_var = tk.BooleanVar(value=False)
        self.show_grid_var = tk.BooleanVar(value=True)
        self.normalize_metric_var = tk.BooleanVar(value=False)
        self.export_format_var = tk.StringVar(value=EXPORT_FORMATS[0])
        self.export_dpi_var = tk.IntVar(value=180)

        # Application state.
        self.runs: list[TensorBoardRun] = []
        self._visible_runs: list[TensorBoardRun] = []
        self._runs_by_iid: dict[str, TensorBoardRun] = {}
        self._colors: dict[str, object] = {}
        self._last_valid_smoothing = 1
        self._last_valid_stride = 1
        self._last_processed_selection: tuple[str, ...] = ()
        self._resize_job: str | None = None
        self._plot_pixels: tuple[int, int] | None = None

        self._build_window()
        self._reload_runs()

    # ------------------------------------------------------------------
    # Window construction
    # ------------------------------------------------------------------

    def _build_window(self) -> None:
        tk, ttk = self.tk, self.ttk
        self.root_window.title("TensorBoard Experiment Viewer")
        self.root_window.geometry("1450x900")
        self.root_window.minsize(1080, 680)

        outer = ttk.Frame(self.root_window, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        self._build_top_bar(outer)
        body = ttk.Frame(outer)
        body.pack(fill=tk.BOTH, expand=True)
        self._build_sidebar(body)
        self._build_plot_area(body)

        self.status_var = tk.StringVar(value="Loading experiments...")
        ttk.Label(outer, textvariable=self.status_var, anchor=tk.W).pack(
            fill=tk.X, pady=(6, 0)
        )

    def _build_top_bar(self, parent) -> None:
        tk, ttk = self.tk, self.ttk
        bar = ttk.LabelFrame(parent, text="Experiment directory", padding=8)
        bar.pack(fill=tk.X, pady=(0, 8))

        entry = ttk.Entry(bar, textvariable=self.runs_root_var)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        entry.bind("<Return>", lambda _event: self._on_refresh_clicked())
        ttk.Button(bar, text="Browse...", command=self._on_browse_directory).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Button(bar, text="Refresh", command=self._on_refresh_clicked).pack(
            side=tk.LEFT, padx=(8, 0)
        )

        ttk.Label(bar, text="Dataset").pack(side=tk.LEFT, padx=(16, 5))
        self.dataset_box = ttk.Combobox(
            bar,
            textvariable=self.dataset_filter_var,
            values=(ALL_DATASETS,),
            state="readonly",
            width=18,
        )
        self.dataset_box.pack(side=tk.LEFT)
        self.dataset_box.bind(
            "<<ComboboxSelected>>", lambda _event: self._on_dataset_selected()
        )

    def _build_sidebar(self, parent) -> None:
        tk, ttk = self.tk, self.ttk
        sidebar = ttk.Frame(parent, width=SIDEBAR_WIDTH_PIXELS)
        sidebar.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        sidebar.pack_propagate(False)

        # Step 1: pick the experiments to compare, grouped by dataset.
        # Selecting a dataset group acts as selecting all of its runs.
        experiment_frame = ttk.LabelFrame(
            sidebar, text="Experiments (Ctrl/Shift multi-select)", padding=6
        )
        experiment_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        tree_holder = ttk.Frame(experiment_frame)
        tree_holder.pack(fill=tk.BOTH, expand=True)
        self.run_tree = ttk.Treeview(
            tree_holder, selectmode="extended", show="tree", height=10
        )
        tree_scrollbar = ttk.Scrollbar(
            tree_holder, orient=tk.VERTICAL, command=self.run_tree.yview
        )
        self.run_tree.configure(yscrollcommand=tree_scrollbar.set)
        self.run_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.run_tree.bind(
            "<<TreeviewSelect>>", lambda _event: self._on_run_selection_changed()
        )

        # Step 2: pick what to plot.  The plot mode decides which curve list
        # is shown, so exactly one list is ever active and no selection is
        # silently filtered away at plot time.
        plot_frame = ttk.LabelFrame(sidebar, text="What to plot", padding=6)
        plot_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        for mode in PLOT_MODES:
            ttk.Radiobutton(
                plot_frame,
                text=mode,
                value=mode,
                variable=self.plot_mode_var,
                command=self._on_plot_mode_changed,
            ).pack(anchor=tk.W)
        self.series_label_var = tk.StringVar(
            value=SERIES_LIST_TITLES[MODE_TRAINING_LOSS]
        )
        ttk.Label(plot_frame, textvariable=self.series_label_var).pack(
            anchor=tk.W, pady=(6, 2)
        )
        series_holder, self.series_listbox = self._make_scrolled_listbox(
            plot_frame, height=8, width=38, selectmode=tk.SINGLE
        )
        series_holder.pack(fill=tk.BOTH, expand=True)
        self.series_listbox.bind(
            "<<ListboxSelect>>", lambda _event: self._on_series_selection_changed()
        )

        # Step 3: appearance.
        appearance = ttk.LabelFrame(sidebar, text="Appearance", padding=6)
        appearance.pack(fill=tk.X, pady=(0, 8))
        appearance.columnconfigure(1, weight=1)
        ttk.Label(appearance, text="X axis").grid(row=0, column=0, sticky=tk.W, pady=2)
        x_axis_box = ttk.Combobox(
            appearance,
            textvariable=self.x_axis_var,
            values=X_AXIS_CHOICES,
            state="readonly",
            width=9,
        )
        x_axis_box.grid(row=0, column=1, sticky=tk.EW, padx=(8, 0))
        x_axis_box.bind(
            "<<ComboboxSelected>>", lambda _event: self._on_display_changed()
        )
        ttk.Label(appearance, text="Smoothing").grid(row=1, column=0, sticky=tk.W, pady=2)
        smoothing_spinbox = ttk.Spinbox(
            appearance, from_=1, to=999, textvariable=self.smoothing_var, width=9
        )
        smoothing_spinbox.grid(row=1, column=1, sticky=tk.EW, padx=(8, 0))
        for sequence in ("<Return>", "<FocusOut>", "<<Increment>>", "<<Decrement>>"):
            smoothing_spinbox.bind(sequence, self._on_smoothing_changed)
        ttk.Label(appearance, text="Sampling").grid(row=2, column=0, sticky=tk.W, pady=2)
        stride_spinbox = ttk.Spinbox(
            appearance, from_=1, to=1000000, textvariable=self.stride_var, width=9
        )
        stride_spinbox.grid(row=2, column=1, sticky=tk.EW, padx=(8, 0))
        for sequence in ("<Return>", "<FocusOut>", "<<Increment>>", "<<Decrement>>"):
            stride_spinbox.bind(sequence, self._on_stride_changed)
        ttk.Checkbutton(
            appearance,
            text="Log X axis",
            variable=self.log_x_var,
            command=self._on_display_changed,
        ).grid(row=3, column=0, sticky=tk.W, pady=2)
        ttk.Checkbutton(
            appearance,
            text="Log Y axis",
            variable=self.log_y_var,
            command=self._on_display_changed,
        ).grid(row=3, column=1, sticky=tk.W, padx=(8, 0))
        ttk.Checkbutton(
            appearance,
            text="Show grid",
            variable=self.show_grid_var,
            command=self._on_display_changed,
        ).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=2)
        self.normalize_check = ttk.Checkbutton(
            appearance,
            text="Normalize each run to 0-1",
            variable=self.normalize_metric_var,
            command=self._on_display_changed,
        )
        self.normalize_check.grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=2)

        # Step 4: export.
        export_frame = ttk.LabelFrame(sidebar, text="Export", padding=6)
        export_frame.pack(fill=tk.X)
        ttk.Label(export_frame, text="Format").grid(row=0, column=0, sticky=tk.W)
        ttk.Combobox(
            export_frame,
            textvariable=self.export_format_var,
            values=EXPORT_FORMATS,
            state="readonly",
            width=7,
        ).grid(row=0, column=1, sticky=tk.W, padx=(6, 12))
        ttk.Label(export_frame, text="DPI").grid(row=0, column=2, sticky=tk.E)
        ttk.Spinbox(
            export_frame, from_=72, to=1200, textvariable=self.export_dpi_var, width=6
        ).grid(row=0, column=3, sticky=tk.EW)
        ttk.Button(
            export_frame, text="Save figure...", command=self._on_save_figure
        ).grid(row=1, column=0, columnspan=4, sticky=tk.EW, pady=(6, 0))

    def _build_plot_area(self, parent) -> None:
        tk, ttk = self.tk, self.ttk
        # Keep the displayed plotting surface at a real 4:3 ratio.  Letting
        # the Tk canvas fill an arbitrarily shaped parent stretches the
        # figure, so instead the canvas holder is resized (still 4:3) when
        # the window changes shape.
        self.plot_area = ttk.Frame(parent)
        self.plot_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.plot_holder = ttk.Frame(
            self.plot_area,
            width=PLOT_SIZE_PIXELS[0],
            height=PLOT_SIZE_PIXELS[1],
        )
        self.plot_holder.pack(anchor=tk.CENTER, expand=True)
        self.plot_holder.pack_propagate(False)
        self.canvas = self.FigureCanvasTkAgg(self.figure, master=self.plot_holder)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.toolbar = self.NavigationToolbar2Tk(
            self.canvas, self.plot_area, pack_toolbar=False
        )
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.plot_area.bind("<Configure>", self._on_plot_area_configured)

    def _make_scrolled_listbox(self, parent, *, height: int, width: int, selectmode=None):
        """Return ``(frame, listbox)`` with a vertical scrollbar attached."""

        tk, ttk = self.tk, self.ttk
        frame = ttk.Frame(parent)
        listbox = tk.Listbox(
            frame,
            height=height,
            width=width,
            selectmode=selectmode or tk.EXTENDED,
            exportselection=False,
            activestyle="none",
        )
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=listbox.yview)
        listbox.configure(yscrollcommand=scrollbar.set)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        return frame, listbox

    # ------------------------------------------------------------------
    # State pipeline: scan -> run list -> series list -> figure
    # ------------------------------------------------------------------

    def _reload_runs(self) -> None:
        """Rescan the experiment directory and rebuild everything below it."""

        try:
            self.runs = discover_tensorboard_runs(Path(self.runs_root_var.get()))
        except Exception as exc:  # noqa: BLE001 - report path/parser failures in the GUI
            self.runs = []
            self._colors.clear()
            self._refresh_dataset_options()
            self._populate_run_list()
            self._set_status(f"Scan failed: {exc}")
            self.messagebox.showerror("Failed to load experiment records", str(exc))
            return
        self._colors.clear()  # the run set may have changed; rebuild stable colors
        self._refresh_dataset_options()
        self._populate_run_list()

    def _refresh_dataset_options(self) -> None:
        datasets = sorted(
            {dataset_name_from_run_name(run.name) for run in self.runs},
            key=str.casefold,
        )
        options = [ALL_DATASETS, *datasets]
        self.dataset_box.configure(values=options)
        if self.dataset_filter_var.get() not in options:
            self.dataset_filter_var.set(ALL_DATASETS)

    def _populate_run_list(self) -> None:
        """Fill the experiment tree (grouped by dataset), preserving selection."""

        previous_names = {run.name for run in self._selected_runs()}
        self._visible_runs = filter_runs_by_dataset(
            self.runs, self.dataset_filter_var.get()
        )
        self._runs_by_iid = {}
        tree_children = self.run_tree.get_children()
        if tree_children:
            self.run_tree.delete(*tree_children)

        groups: dict[str, list[TensorBoardRun]] = {}
        for run in self._visible_runs:
            groups.setdefault(dataset_name_from_run_name(run.name), []).append(run)
        for dataset in sorted(
            groups, key=lambda name: (name == UNKNOWN_DATASET, name.casefold())
        ):
            parent = self.run_tree.insert(
                "",
                self.tk.END,
                iid=f"dataset::{dataset}",
                open=True,
                text=f"{dataset} ({len(groups[dataset])})",
            )
            for run in groups[dataset]:
                display = (
                    f"{run.name} [warnings: {len(run.errors)}]" if run.errors else run.name
                )
                self.run_tree.insert(parent, self.tk.END, iid=run.name, text=display)
                self._runs_by_iid[run.name] = run

        restore = [name for name in previous_names if name in self._runs_by_iid]
        if restore:
            self.run_tree.selection_set(restore)
        elif self._visible_runs:
            self.run_tree.selection_set(next(iter(self._runs_by_iid)))
        # selection_set queues a <<TreeviewSelect>> event; record the snapshot
        # here so the queued handler call becomes a no-op instead of redoing
        # this populate and overwriting the status message below.
        self._last_processed_selection = tuple(self.run_tree.selection())

        if not self._visible_runs:
            self._populate_series_list(preserve=False)
            self.figure.clear()
            self.canvas.draw_idle()
            self._set_status(
                f"No TensorBoard records found under {self.runs_root_var.get()}"
            )
            return

        self._populate_series_list(preserve=True)
        redraw_summary = self._redraw()
        details = [
            f"Loaded {len(self._visible_runs)}/{len(self.runs)} experiment(s) in {len(groups)} dataset(s)"
        ]
        if self.dataset_filter_var.get() != ALL_DATASETS:
            details.append(f"dataset={self.dataset_filter_var.get()}")
        warning_count = sum(bool(run.errors) for run in self._visible_runs)
        if warning_count:
            details.append(f"{warning_count} record(s) with parser warnings")
        self._set_status("; ".join(details) + ". " + redraw_summary)

    def _populate_series_list(self, *, preserve: bool) -> None:
        """Fill the curve list for the current plot mode.

        ``preserve=True`` keeps the user's curve selection across run-list
        changes; a plot-mode switch passes ``preserve=False`` so the mode's
        default selection is applied instead.
        """

        tk = self.tk
        mode = self.plot_mode_var.get()
        runs = self._selected_runs() or self._visible_runs
        available_tags = [tag for run in runs for tag in run.tags]
        tags = series_tags_for_mode(mode, available_tags)

        previous: set[str] = set()
        if preserve:
            previous = {
                self.series_listbox.get(index)
                for index in self.series_listbox.curselection()
            }

        selectmode = tk.SINGLE if mode == MODE_VALIDATION_METRIC else tk.EXTENDED
        if str(self.series_listbox.cget("selectmode")) != selectmode:
            self.series_listbox.configure(selectmode=selectmode)
        self.series_listbox.delete(0, tk.END)
        for tag in tags:
            self.series_listbox.insert(tk.END, tag)
        self.series_label_var.set(SERIES_LIST_TITLES[mode])
        self.normalize_check.configure(
            state=tk.NORMAL if mode == MODE_VALIDATION_METRIC else tk.DISABLED
        )

        for index, tag in enumerate(tags):
            if tag in previous:
                self.series_listbox.selection_set(index)
                if selectmode == tk.SINGLE:
                    break
        if not self.series_listbox.curselection():
            for tag in default_series_selection(tags, mode):
                self.series_listbox.selection_set(tags.index(tag))

    # ------------------------------------------------------------------
    # Event handlers (thin; each delegates to the state pipeline)
    # ------------------------------------------------------------------

    def _on_browse_directory(self) -> None:
        selected = self.filedialog.askdirectory(
            title="Select runs or run directory",
            initialdir=self.runs_root_var.get(),
        )
        if selected:
            self.runs_root_var.set(selected)
            self._reload_runs()

    def _on_refresh_clicked(self) -> None:
        self._reload_runs()

    def _on_dataset_selected(self) -> None:
        self._populate_run_list()

    def _on_run_selection_changed(self) -> None:
        selection = tuple(self.run_tree.selection())
        if selection == self._last_processed_selection:
            return  # programmatic selection during populate, or a stale event
        self._last_processed_selection = selection
        self._populate_series_list(preserve=True)
        self._set_status(self._redraw())

    def _on_plot_mode_changed(self) -> None:
        self._populate_series_list(preserve=False)
        self._set_status(self._redraw())

    def _on_series_selection_changed(self) -> None:
        self._set_status(self._redraw())

    def _on_display_changed(self) -> None:
        self._set_status(self._redraw())

    def _on_smoothing_changed(self, _event=None) -> None:
        self._current_smoothing()  # normalise the spinbox content first
        self._set_status(self._redraw())

    def _on_stride_changed(self, _event=None) -> None:
        self._current_stride()  # normalise the spinbox content first
        self._set_status(self._redraw())

    def _on_save_figure(self) -> None:
        export_format = self.export_format_var.get().casefold()
        if export_format not in EXPORT_FORMATS:
            self.messagebox.showerror("Export Error", "Format must be png, jpeg, or pdf.")
            return
        extension = ".jpg" if export_format == "jpeg" else f".{export_format}"
        target = self.filedialog.asksaveasfilename(
            title="Save Figure",
            defaultextension=extension,
            filetypes=[
                (f"{export_format.upper()} file", f"*{extension}"),
                ("All files", "*.*"),
            ],
        )
        if not target:
            return
        target_path = Path(target)
        if target_path.suffix.casefold() != extension:
            target_path = target_path.with_suffix(extension)
        try:
            dpi = self._current_dpi()
            export_figure(self.figure, target_path, export_format, dpi)
        except Exception as exc:  # noqa: BLE001 - report save failures in the GUI
            self.messagebox.showerror("Save Error", str(exc))
            return
        self._set_status(f"Figure saved: {target_path} ({export_format}, {dpi} DPI)")

    def _on_plot_area_configured(self, _event) -> None:
        # Debounce: resizing the figure redraws every curve, so coalesce the
        # bursts of Configure events produced while dragging the window.
        if self._resize_job is not None:
            self.root_window.after_cancel(self._resize_job)
        self._resize_job = self.root_window.after(80, self._apply_plot_size)

    def _apply_plot_size(self) -> None:
        self._resize_job = None
        available_width = self.plot_area.winfo_width() - 12
        available_height = (
            self.plot_area.winfo_height() - 12 - self.toolbar.winfo_reqheight()
        )
        width = max(int(min(available_width, available_height * PLOT_ASPECT_RATIO)), MIN_PLOT_PIXELS[0])
        width -= width % 4  # keep the 4:3 ratio exact despite integer pixels
        height = max(width * 3 // 4, MIN_PLOT_PIXELS[1])
        if (width, height) == self._plot_pixels:
            return
        self._plot_pixels = (width, height)
        self.plot_holder.configure(width=width, height=height)
        self.figure.set_size_inches(width / FIGURE_DPI, height / FIGURE_DPI)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Reading selections and rendering
    # ------------------------------------------------------------------

    def _selected_runs(self) -> list[TensorBoardRun]:
        """Return selected runs; a selected dataset group yields all its runs."""

        runs: list[TensorBoardRun] = []
        seen: set[str] = set()
        for iid in self.run_tree.selection():
            for run in self._runs_for_iid(iid):
                if run.name not in seen:
                    seen.add(run.name)
                    runs.append(run)
        return runs

    def _runs_for_iid(self, iid: str) -> list[TensorBoardRun]:
        if iid in self._runs_by_iid:
            return [self._runs_by_iid[iid]]
        if iid.startswith("dataset::"):
            return [
                self._runs_by_iid[child]
                for child in self.run_tree.get_children(iid)
                if child in self._runs_by_iid
            ]
        return []

    def _selected_series_tags(self) -> list[str]:
        return [
            self.series_listbox.get(index)
            for index in self.series_listbox.curselection()
        ]

    def _current_x_axis(self) -> str:
        return canonical_x_axis(self.x_axis_var.get())

    def _current_smoothing(self) -> int:
        try:
            current = int(self.smoothing_var.get())
        except (self.tk.TclError, TypeError, ValueError):
            current = None
        value = self._last_valid_smoothing if current is None else max(1, min(999, current))
        self._last_valid_smoothing = value
        if current != value:
            self.smoothing_var.set(value)
        return value

    def _current_stride(self) -> int:
        try:
            current = int(self.stride_var.get())
        except (self.tk.TclError, TypeError, ValueError):
            current = None
        value = (
            self._last_valid_stride
            if current is None
            else max(1, min(1000000, current))
        )
        self._last_valid_stride = value
        if current != value:
            self.stride_var.set(value)
        return value

    def _current_dpi(self) -> int:
        try:
            current = int(self.export_dpi_var.get())
        except (self.tk.TclError, TypeError, ValueError):
            current = None
        value = 180 if current is None else max(72, min(1200, current))
        if current != value:
            self.export_dpi_var.set(value)
        return value

    def _metric_series(
        self, run: TensorBoardRun, tag: str, smoothing: int, stride: int
    ) -> tuple[list[float], list[float]]:
        x_values, y_values = series_xy(
            run,
            tag,
            x_axis=self._current_x_axis(),
            smoothing=smoothing,
            sampling_stride=stride,
        )
        if not self.normalize_metric_var.get() or not y_values:
            return x_values, y_values
        low, high = min(y_values), max(y_values)
        if high - low < 1e-12:
            return x_values, [0.5] * len(y_values)
        return x_values, [(value - low) / (high - low) for value in y_values]

    def _set_grid(self, axis) -> None:
        # Style the grid lines directly instead of passing line properties to
        # axis.grid(): Matplotlib warns and re-enables the grid when they are
        # combined with visible=False, which would break the checkbox.
        visible = bool(self.show_grid_var.get())
        axis.grid(visible=visible, which="major", axis="both")
        for grid_line in (*axis.get_xgridlines(), *axis.get_ygridlines()):
            grid_line.set(
                visible=visible,
                color="#B0B0B0",
                linestyle="--",
                linewidth=0.6,
                alpha=0.45,
            )

    def _run_colors(self) -> dict[str, object]:
        if not self._colors:
            import matplotlib.pyplot as plt

            cmap = plt.get_cmap("tab20")
            ordered_runs = sorted(self.runs, key=lambda run: run.name.casefold())
            for index, run in enumerate(ordered_runs):
                self._colors[run.name] = cmap(index % cmap.N)
        return {run.name: self._colors[run.name] for run in self._selected_runs()}

    def _redraw(self) -> str:
        """Redraw the figure and return a one-line summary for the status bar."""

        mode = self.plot_mode_var.get()
        runs = self._selected_runs()
        tags = self._selected_series_tags()
        if not runs:
            return "Select at least one experiment"
        if not tags:
            return f"No {mode.casefold()} series available for the selected experiments"

        self.figure.clear()
        axis = self.figure.add_subplot(1, 1, 1)
        if self.log_x_var.get():
            axis.set_xscale("log")
        if self.log_y_var.get():
            axis.set_yscale("log")
        palette = self._run_colors()
        smoothing = self._current_smoothing()
        stride = self._current_stride()
        x_axis = self._current_x_axis()
        plotted_runs = 0
        plotted_curves = 0

        if mode == MODE_VALIDATION_METRIC:
            tag = tags[0]
            for run in runs:
                x_values, y_values = self._metric_series(run, tag, smoothing, stride)
                if not x_values:
                    continue
                axis.plot(
                    x_values,
                    y_values,
                    color=palette[run.name],
                    linewidth=1.9,
                    marker="o" if len(x_values) <= 30 else None,
                    markersize=3,
                    label=run.name,
                )
                plotted_runs += 1
                plotted_curves += 1
            title = f"Validation metric: {tag}"
            y_label = (
                "Per-run normalized value"
                if self.normalize_metric_var.get()
                else tag
            )
        else:
            for run in runs:
                run_curves = 0
                for tag in tags:
                    x_values, y_values = series_xy(
                        run,
                        tag,
                        x_axis=x_axis,
                        smoothing=smoothing,
                        sampling_stride=stride,
                    )
                    if not x_values:
                        continue
                    linestyle = "-" if tag.casefold().startswith("train/") else "--"
                    axis.plot(
                        x_values,
                        y_values,
                        color=palette[run.name],
                        linestyle=linestyle,
                        linewidth=1.7,
                        alpha=0.95,
                        label=f"{run.name} - {tag}",
                    )
                    run_curves += 1
                plotted_curves += run_curves
                if run_curves:
                    plotted_runs += 1
            title = mode
            y_label = "Loss"

        if not plotted_curves:
            self.canvas.draw_idle()
            return "The selected experiments contain no data for this plot mode"

        fallback_runs = [
            run
            for run in runs
            if x_axis == "epoch" and effective_x_axis(run, "epoch") != "epoch"
        ]
        x_label = "Iter" if x_axis == "step" else "Epoch"
        if fallback_runs:
            x_label = "Epoch (iter fallback for missing epoch logs)"
        axis.set_title(title, fontname=PLOT_FONT_FAMILY, fontweight="normal", fontsize=13)
        axis.set_xlabel(x_label, fontname=PLOT_FONT_FAMILY, fontweight="normal")
        axis.set_ylabel(y_label, fontname=PLOT_FONT_FAMILY, fontweight="normal")
        axis.tick_params(axis="both", labelsize=10)
        for label in (*axis.get_xticklabels(), *axis.get_yticklabels()):
            label.set_fontname(PLOT_FONT_FAMILY)
            label.set_fontweight("normal")
        self._set_grid(axis)
        axis.spines["top"].set_visible(False)
        axis.legend(
            loc="best",
            fontsize=8,
            frameon=False,
            prop={"family": PLOT_FONT_FAMILY, "weight": "normal"},
        )
        if mode == MODE_VALIDATION_METRIC and self.normalize_metric_var.get():
            axis.set_ylim(-0.05, 1.05)
        self.canvas.draw_idle()

        summary = (
            f"{mode}: plotted {plotted_runs} run(s), {plotted_curves} curve(s), "
            f"smoothing={smoothing}, sampling={stride}, x={self.x_axis_var.get()}"
        )
        log_axes = [
            name
            for name, enabled in (
                ("log-x", self.log_x_var.get()),
                ("log-y", self.log_y_var.get()),
            )
            if enabled
        ]
        if log_axes:
            summary += ", " + "+".join(log_axes)
        if fallback_runs:
            summary += f"; {len(fallback_runs)} run(s) use the iter fallback"
        return summary

    def _set_status(self, message: str) -> None:
        self.status_var.set(message)


def launch_gui(runs_root: Path) -> None:
    """Start the desktop viewer."""

    import tkinter as tk

    import matplotlib

    matplotlib.use("TkAgg")
    root_window = tk.Tk()
    TensorBoardViewer(root_window, runs_root)
    root_window.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GUI for browsing and comparing TensorBoard training records")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="runs/run root directory (defaults to <project>/runs/)",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    launch_gui(arguments.root or default_runs_root())


if __name__ == "__main__":
    main()
