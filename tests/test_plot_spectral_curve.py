from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg

from scripts.plot_spectral_curve import (
    _X_MODE_BAND,
    _X_MODE_WAVELENGTH,
    SpectralCurveData,
    _band_to_display_x,
    _bbox_in_axes_fraction,
    _default_rgb_bands,
    _display_x_to_band,
    _normalize_zoom,
    _rects_overlap,
    create_curve_figure,
    create_preview_figure,
    extract_spectral_curve,
    load_spectral_curve_data,
    render_curve_figure,
    resolve_wavelength_axis,
    write_curve_csv,
)


def _write_h5(path, key: str, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset(key, data=values.astype(np.float32))


def _write_wavelength_config(path, *, start: float, end: float, bands: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "spectral:\n"
        "  name: tabulated_srf\n"
        f"  wavelength_start_nm: {start}\n"
        f"  wavelength_end_nm: {end}\n"
        f"  num_hsi_bands: {bands}\n",
        encoding="utf-8",
    )


def _gradient_cube(bands: int = 3, height: int = 4, width: int = 4) -> np.ndarray:
    """CHW cube whose value encodes (band, row, column) for easy assertions."""

    cube = np.zeros((bands, height, width), dtype=np.float32)
    for band in range(bands):
        cube[band] += 10.0 * band
    for row in range(height):
        cube[:, row, :] += row
    for column in range(width):
        cube[:, :, column] += 0.1 * column
    return cube


def test_extract_spectral_curve_returns_along_channel_vector() -> None:
    cube = _gradient_cube(bands=3, height=4, width=4)

    curve = extract_spectral_curve(cube, row=1, column=2)

    assert curve.shape == (3,)
    assert curve.dtype == np.float32
    # value = 10 * band + row + 0.1 * column
    assert np.allclose(curve, [1.2, 11.2, 21.2])


def test_extract_spectral_curve_rejects_out_of_range_pixels_and_shapes() -> None:
    cube = _gradient_cube(bands=3, height=4, width=4)

    with pytest.raises(IndexError, match="outside the image"):
        extract_spectral_curve(cube, row=4, column=0)
    with pytest.raises(IndexError, match="outside the image"):
        extract_spectral_curve(cube, row=0, column=-1)
    with pytest.raises(ValueError, match="Expected a CHW HSI sample"):
        extract_spectral_curve(cube[0], row=0, column=0)


def test_loader_discovers_benchmark_case_insensitively_and_compares_methods(tmp_path) -> None:
    output_root = tmp_path / "output"
    dataset_root = tmp_path / "Dataset"
    target = _gradient_cube(bands=3, height=4, width=4)[np.newaxis, ...]
    alpha = target.copy()
    beta = np.zeros_like(target)
    _write_h5(dataset_root / "Chikusei" / "test.h5", "HRHS", target)
    _write_h5(output_root / "chikusei_x4" / "alpha" / "predictions.h5", "prediction", alpha)
    _write_h5(output_root / "chikusei_x4" / "beta" / "predictions.h5", "prediction", beta)

    data = load_spectral_curve_data(
        output_root=output_root,
        dataset_root=dataset_root,
        benchmark="CHIKUSEI_X4",
        methods=["beta", "alpha"],
        sample_index=0,
        row=1,
        column=2,
    )

    assert data.benchmark == "chikusei_x4"
    assert data.target_path == dataset_root / "Chikusei" / "test.h5"
    assert data.methods == ["beta", "alpha"]
    assert data.gt_curve.shape == (3,)
    assert np.allclose(data.curves["alpha"], data.gt_curve)
    assert np.allclose(data.curves["beta"], 0.0)


def test_loader_reads_only_the_picked_pixel_but_validates_shapes(tmp_path) -> None:
    output_root = tmp_path / "output"
    dataset_root = tmp_path / "Dataset"
    target = np.zeros((2, 3, 4, 4), dtype=np.float32)
    _write_h5(dataset_root / "chikusei" / "test.h5", "HRHS", target)
    _write_h5(output_root / "chikusei_x4" / "good" / "predictions.h5", "prediction", target)
    _write_h5(
        output_root / "chikusei_x4" / "bad" / "predictions.h5",
        "prediction",
        np.zeros((2, 3, 8, 8), dtype=np.float32),
    )

    data = load_spectral_curve_data(
        output_root=output_root,
        dataset_root=dataset_root,
        benchmark="chikusei_x4",
        methods=["good"],
        sample_index=1,
        row=0,
        column=3,
        config_root=tmp_path / "configs",  # no wavelength configuration
    )
    assert data.wavelengths is None
    assert np.allclose(data.curves["good"], 0.0)

    with pytest.raises(ValueError, match="does not match the ground truth shape"):
        load_spectral_curve_data(
            output_root=output_root,
            dataset_root=dataset_root,
            benchmark="chikusei_x4",
            methods=["bad"],
            sample_index=0,
            row=0,
            column=0,
        )

    with pytest.raises(IndexError, match="outside the image"):
        load_spectral_curve_data(
            output_root=output_root,
            dataset_root=dataset_root,
            benchmark="chikusei_x4",
            sample_index=0,
            row=9,
            column=0,
        )


def test_resolve_wavelength_axis_reads_config_and_falls_back(tmp_path) -> None:
    config_root = tmp_path / "configs"
    _write_wavelength_config(
        config_root / "degradation" / "chikusei_wv2_x4.yaml", start=400.0, end=700.0, bands=3
    )

    axis = resolve_wavelength_axis(config_root, "chikusei_x4", band_count=3)

    assert axis is not None
    assert np.allclose(axis, [400.0, 550.0, 700.0])

    # Band count mismatch, missing configuration and non-benchmark names fall back.
    assert resolve_wavelength_axis(config_root, "chikusei_x4", band_count=4) is None
    assert resolve_wavelength_axis(config_root, "paviac_x4", band_count=3) is None
    assert resolve_wavelength_axis(None, "chikusei_x4", band_count=3) is None


def test_write_curve_csv_band_and_wavelength_headers(tmp_path) -> None:
    curve_gt = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    curve_a = np.array([0.2, 0.4, 0.6], dtype=np.float32)
    data = SpectralCurveData(
        benchmark="chikusei_x4",
        target_path=Path("test.h5"),
        sample_index=0,
        row=1,
        column=2,
        gt_curve=curve_gt,
        curves={"alpha": curve_a},
        wavelengths=np.array([400.0, 550.0, 700.0], dtype=np.float32),
    )

    band_csv = write_curve_csv(data, tmp_path / "band.csv", x_mode=_X_MODE_BAND)
    lines = band_csv.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "BandIndex,GT,alpha"
    assert lines[1] == "0,0.100000,0.200000"
    assert lines[3] == "2,0.900000,0.600000"

    wavelength_csv = write_curve_csv(data, tmp_path / "wl.csv", x_mode=_X_MODE_WAVELENGTH)
    lines = wavelength_csv.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "WavelengthNM,GT,alpha"
    assert lines[1] == "400.00,0.100000,0.200000"

    missing = SpectralCurveData(
        benchmark="chikusei_x4",
        target_path=Path("test.h5"),
        sample_index=0,
        row=1,
        column=2,
        gt_curve=curve_gt,
        curves={"alpha": curve_a},
        wavelengths=None,
    )
    with pytest.raises(ValueError, match="Wavelength information is unavailable"):
        write_curve_csv(missing, tmp_path / "bad.csv", x_mode=_X_MODE_WAVELENGTH)


def test_create_curve_figure_draws_gt_and_methods(tmp_path) -> None:
    data = SpectralCurveData(
        benchmark="cave_x8",
        target_path=tmp_path / "test.h5",
        sample_index=2,
        row=3,
        column=4,
        gt_curve=np.linspace(0.0, 1.0, 5, dtype=np.float32),
        curves={
            "alpha": np.linspace(0.1, 0.9, 5, dtype=np.float32),
            "beta": np.linspace(0.2, 0.8, 5, dtype=np.float32),
        },
        wavelengths=np.linspace(400.0, 700.0, 5, dtype=np.float32),
    )

    figure = create_curve_figure(data, x_mode=_X_MODE_WAVELENGTH)
    axis = figure.axes[0]
    assert axis.get_xlabel() == _X_MODE_WAVELENGTH
    labels = [line.get_label() for line in axis.get_lines()]
    assert labels[0] == "Ground truth"
    assert set(labels[1:]) == {"alpha", "beta"}
    canvas = FigureCanvasAgg(figure)
    path = tmp_path / "curve.png"
    figure.savefig(path, dpi=100)
    assert path.is_file() and path.stat().st_size > 0
    assert isinstance(canvas, FigureCanvasAgg)

    band_figure = create_curve_figure(data, x_mode=_X_MODE_BAND)
    assert band_figure.axes[0].get_xlabel() == _X_MODE_BAND

    no_wavelengths = SpectralCurveData(
        benchmark="cave_x8",
        target_path=tmp_path / "test.h5",
        sample_index=2,
        row=3,
        column=4,
        gt_curve=data.gt_curve,
        curves=dict(data.curves),
        wavelengths=None,
    )
    with pytest.raises(ValueError, match="Wavelength information is unavailable"):
        create_curve_figure(no_wavelengths, x_mode=_X_MODE_WAVELENGTH)


def test_create_preview_figure_marks_picked_pixel_and_saves(tmp_path) -> None:
    cube = _gradient_cube(bands=4, height=6, width=6)

    figure = create_preview_figure(cube, row=2, column=3, contrast_stretch=True)
    axis = figure.axes[0]
    assert axis.get_title() == "Ground truth (RGB preview)"
    markers = axis.get_lines()
    assert len(markers) == 2  # circle + crosshair at the picked pixel
    for marker in markers:
        x_data, y_data = marker.get_xdata(), marker.get_ydata()
        assert list(x_data) == [3] and list(y_data) == [2]

    path = tmp_path / "preview.png"
    figure.savefig(path, dpi=80)
    assert path.is_file() and path.stat().st_size > 0

    with pytest.raises(IndexError, match="outside the image"):
        create_preview_figure(cube, row=6, column=0)


def _sample_curve_data(**overrides) -> SpectralCurveData:
    values = dict(
        benchmark="cave_x8",
        target_path=Path("test.h5"),
        sample_index=2,
        row=3,
        column=4,
        gt_curve=np.linspace(0.0, 1.0, 5, dtype=np.float32),
        curves={
            "alpha": np.linspace(0.1, 0.9, 5, dtype=np.float32),
            "beta": np.linspace(0.2, 0.8, 5, dtype=np.float32),
        },
        wavelengths=np.linspace(400.0, 700.0, 5, dtype=np.float32),
    )
    values.update(overrides)
    return SpectralCurveData(**values)


def test_create_curve_figure_zoom_inset_is_dashed_and_avoids_legend(tmp_path) -> None:
    data = _sample_curve_data()

    figure = create_curve_figure(
        data, x_mode=_X_MODE_WAVELENGTH, zoom=(450.0, 500.0, 0.3, 0.6)
    )
    assert len(figure.axes) == 1  # the inset is a child axis of the main axes
    main = figure.axes[0]
    assert len(main.child_axes) == 1
    inset = main.child_axes[0]

    assert inset.get_xlim() == pytest.approx((450.0, 500.0))
    assert inset.get_ylim() == pytest.approx((0.3, 0.6))
    inset_lines = inset.get_lines()
    assert len(inset_lines) == 3  # GT + two methods
    assert inset_lines[0].get_linestyle() == "-"  # GT stays solid
    for line in inset_lines[1:]:
        assert line.get_linestyle() == "--"
    assert len(main.get_lines()) == 3  # no connector lines towards the inset
    assert len(main.patches) == 1  # the zoomed source rectangle outline
    for line in main.get_lines()[1:]:
        assert line.get_linestyle() == "-"  # main axes keeps solid methods

    # The inset must not overlap the legend box (axes-fraction coordinates).
    canvas = FigureCanvasAgg(figure)
    assert isinstance(canvas, FigureCanvasAgg)
    figure.canvas.draw()
    legend_rect = _bbox_in_axes_fraction(main, main.get_legend().get_window_extent())
    main_pos, inset_pos = main.get_position(), inset.get_position()
    inset_rect = (
        (inset_pos.x0 - main_pos.x0) / main_pos.width,
        (inset_pos.y0 - main_pos.y0) / main_pos.height,
        (inset_pos.x1 - main_pos.x0) / main_pos.width,
        (inset_pos.y1 - main_pos.y0) / main_pos.height,
    )
    assert not _rects_overlap(inset_rect, legend_rect)

    # The inset renders at a 4:3 aspect ratio and stays compact.
    figure_width, figure_height = (float(v) for v in (figure.get_size_inches() * figure.dpi))
    visual_width = inset_pos.width * figure_width
    visual_height = inset_pos.height * figure_height
    assert visual_width / visual_height == pytest.approx(4 / 3, rel=0.02)
    assert inset_pos.width / main_pos.width == pytest.approx(0.28, abs=0.01)

    path = tmp_path / "zoom.png"
    figure.savefig(path, dpi=100)
    assert path.is_file() and path.stat().st_size > 0

    with pytest.raises(ValueError, match="zoom region is empty"):
        create_curve_figure(data, zoom=(2.0, 2.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="zoom region is empty"):
        create_curve_figure(data, zoom=(0.0, 2.0, 0.5, 0.5))
    with pytest.raises(ValueError, match="zoom must contain"):
        create_curve_figure(data, zoom=(0.0, 1.0, 0.0))


def test_create_curve_figure_highlight_orders_legend_and_adds_markers() -> None:
    data = _sample_curve_data()

    figure = create_curve_figure(data, highlight_methods=["beta"])
    main = figure.axes[0]
    labels = [line.get_label() for line in main.get_lines()]
    assert labels == ["Ground truth", "beta", "alpha"]

    by_label = {line.get_label(): line for line in main.get_lines()}
    assert by_label["beta"].get_marker() == "o"
    assert by_label["beta"].get_linewidth() > by_label["alpha"].get_linewidth()
    assert by_label["alpha"].get_marker() != "o"

    # Highlight also applies inside the zoom inset.
    zoomed = create_curve_figure(
        data, x_mode=_X_MODE_WAVELENGTH, highlight_methods=["beta"], zoom=(450.0, 550.0, 0.2, 0.7)
    )
    inset_lines = zoomed.axes[0].child_axes[0].get_lines()
    by_label = {line.get_label(): line for line in inset_lines}
    assert by_label["beta"].get_marker() == "o"
    assert by_label["beta"].get_linestyle() == "--"


def test_normalize_zoom_sorts_and_clips() -> None:
    x_values = np.arange(5.0)

    assert _normalize_zoom((3, 1, 0.5, 0.2), x_values) == (1.0, 3.0, 0.2, 0.5)
    assert _normalize_zoom((-2, 8, 0.0, 1.0), x_values) == (0.0, 4.0, 0.0, 1.0)

    with pytest.raises(ValueError, match="zoom region is empty"):
        _normalize_zoom((2, 2, 0.0, 1.0), x_values)
    with pytest.raises(ValueError, match="zoom region is empty"):
        _normalize_zoom((0, 1, 0.0, 0.0), x_values)
    with pytest.raises(ValueError, match="zoom must contain"):
        _normalize_zoom((0, 1, 0), x_values)


def test_band_and_display_x_converters_round_trip() -> None:
    wavelengths = np.array([400.0, 500.0, 600.0], dtype=np.float32)

    assert _band_to_display_x(1.0, wavelengths, _X_MODE_WAVELENGTH) == pytest.approx(500.0)
    assert _band_to_display_x(0.5, wavelengths, _X_MODE_WAVELENGTH) == pytest.approx(450.0)
    assert _band_to_display_x(2.0, wavelengths, _X_MODE_BAND) == 2.0
    assert _display_x_to_band(450.0, wavelengths, _X_MODE_WAVELENGTH) == pytest.approx(0.5)
    assert _display_x_to_band(1.5, wavelengths, _X_MODE_BAND) == 1.5

    with pytest.raises(ValueError, match="Wavelength information is unavailable"):
        _band_to_display_x(1.0, None, _X_MODE_WAVELENGTH)
    with pytest.raises(ValueError, match="Wavelength information is unavailable"):
        _display_x_to_band(450.0, None, _X_MODE_WAVELENGTH)


def test_default_rgb_bands_spread_across_range() -> None:
    assert _default_rgb_bands(3) == (2, 1, 0)
    assert _default_rgb_bands(31) == (21, 15, 9)
    red, green, blue = _default_rgb_bands(128)
    assert red > green > blue


def test_render_curve_figure_uses_expected_filename(tmp_path) -> None:
    data = SpectralCurveData(
        benchmark="chikusei_x4",
        target_path=tmp_path / "test.h5",
        sample_index=0,
        row=1,
        column=2,
        gt_curve=np.zeros(3, dtype=np.float32),
        curves={"alpha": np.ones(3, dtype=np.float32)},
        wavelengths=None,
    )

    path = render_curve_figure(
        data, output_directory=tmp_path, output_format="png", dpi=90
    )

    assert path.name == "spectral_curve_chikusei_x4-sample0_r1c2.png"
    assert path.is_file() and path.stat().st_size > 0

    with pytest.raises(ValueError, match="Unsupported output format"):
        render_curve_figure(data, output_directory=tmp_path, output_format="bmp", dpi=90)
    with pytest.raises(ValueError, match="dpi must be a positive integer"):
        render_curve_figure(data, output_directory=tmp_path, output_format="png", dpi=0)
