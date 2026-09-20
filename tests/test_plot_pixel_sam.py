from __future__ import annotations

import h5py
import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg

from scripts.plot_pixel_sam import (
    _HIGHLIGHT_COLOR,
    _MAX_SAMPLES,
    create_sample_comparison_figure,
    discover_output_predictions,
    load_pixel_sam_data,
    load_sample_pixel_sam_data,
    main,
    pixel_sam_degrees,
    render_sample_comparison_figure,
    select_predictions,
    sorted_pixel_sam_degrees,
)


def _write_h5(path, key: str, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset(key, data=values.astype(np.float32))


def test_pixel_sam_degrees_is_zero_for_identical_cubes() -> None:
    cube = np.linspace(0.0, 1.0, num=3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
    sam = pixel_sam_degrees(cube, cube)

    assert sam.shape == (4, 4)
    assert np.allclose(sam, 0.0, atol=1e-5)


def test_sorted_pixel_sam_is_ascending_and_handles_orthogonal_spectra() -> None:
    target = np.zeros((2, 2, 2), dtype=np.float32)
    target[0] = 1.0
    prediction = np.zeros_like(target)
    prediction[:, 0, 0] = target[:, 0, 0]  # identical
    prediction[:, 0, 1] = np.array([0.0, 1.0], dtype=np.float32)  # 90 degrees
    prediction[:, 1, :] = 0.5

    curve = sorted_pixel_sam_degrees(prediction, target)

    assert curve.shape == (4,)
    assert np.all(np.diff(curve) >= -1e-6)
    assert curve[-1] == pytest.approx(90.0, abs=1e-3)
    assert curve[0] == pytest.approx(0.0, abs=1e-3)


def test_loader_discovers_dataset_and_computes_sorted_curves(tmp_path) -> None:
    output_root = tmp_path / "output"
    dataset_root = tmp_path / "Dataset"
    target = np.zeros((2, 3, 4, 4), dtype=np.float32)
    target[:, 0] = 1.0
    prediction = target.copy()
    bad = target.copy()
    # Sample 1 is orthogonal to the target so every pixel SAM is 90 degrees.
    bad[1, 0] = 0.0
    bad[1, 1] = 1.0
    _write_h5(dataset_root / "Chikusei" / "test.h5", "HRHS", target)
    _write_h5(output_root / "chikusei_x4" / "alpha" / "predictions.h5", "prediction", prediction)
    _write_h5(output_root / "chikusei_x4" / "beta" / "predictions.h5", "prediction", bad)

    data = load_pixel_sam_data(
        output_root=output_root,
        dataset_root=dataset_root,
        benchmark="CHIKUSEI_X4",
        methods=["beta", "alpha"],
    )

    assert data.methods == ["beta", "alpha"]
    assert data.target_path == dataset_root / "Chikusei" / "test.h5"
    assert data.per_image["alpha"].shape == (2, 16)
    assert np.allclose(data.per_image["alpha"], 0.0, atol=1e-4)
    assert np.allclose(data.per_image["beta"][0], 0.0, atol=1e-4)
    assert np.allclose(data.per_image["beta"][1], 90.0, atol=1e-2)


def test_sample_comparison_export_and_highlight(tmp_path) -> None:
    output_root = tmp_path / "output"
    dataset_root = tmp_path / "Dataset"
    target = np.zeros((2, 2, 4, 4), dtype=np.float32)
    target[:, 0] = 1.0
    _write_h5(dataset_root / "chikusei" / "test.h5", "HRHS", target)
    _write_h5(output_root / "chikusei_x4" / "method_a" / "predictions.h5", "prediction", target)
    _write_h5(
        output_root / "chikusei_x4" / "method_b" / "predictions.h5",
        "prediction",
        np.full_like(target, 0.5),
    )

    data = load_sample_pixel_sam_data(
        output_root=output_root,
        dataset_root=dataset_root,
        selections=[("chikusei_x4", 0), ("chikusei_x4", 1)],
    )

    figure_path = render_sample_comparison_figure(
        data,
        output_directory=tmp_path / "figures",
        output_format="pdf",
        dpi=100,
        highlight_methods=["method_b"],
    )

    assert figure_path.name == "pixel_sam_comparison_chikusei_x4-0_chikusei_x4-1.pdf"
    assert figure_path.is_file() and figure_path.stat().st_size > 0


def test_highlighted_method_is_drawn_as_solid_line(tmp_path) -> None:
    output_root = tmp_path / "output"
    dataset_root = tmp_path / "Dataset"
    target = np.zeros((1, 2, 4, 4), dtype=np.float32)
    target[:, 0] = 1.0
    _write_h5(dataset_root / "chikusei" / "test.h5", "HRHS", target)
    _write_h5(output_root / "chikusei_x4" / "method_a" / "predictions.h5", "prediction", target)
    _write_h5(
        output_root / "chikusei_x4" / "method_b" / "predictions.h5",
        "prediction",
        np.full_like(target, 0.5),
    )
    data = load_sample_pixel_sam_data(
        output_root=output_root,
        dataset_root=dataset_root,
        selections=[("chikusei_x4", 0)],
    )

    figure = create_sample_comparison_figure(
        data.samples, methods=data.methods, highlight_methods=["method_b"], log_scale=False
    )
    FigureCanvasAgg(figure)
    figure.canvas.draw()

    lines = figure.axes[0].get_lines()
    styles = {line.get_label(): line.get_linestyle() for line in lines}
    colors = {line.get_label(): line.get_color() for line in lines}
    assert styles["method_b"] == "-"
    assert styles["method_a"] != "-"
    assert colors["method_b"] == _HIGHLIGHT_COLOR


def test_samples_may_come_from_different_datasets_and_share_common_methods(tmp_path) -> None:
    output_root = tmp_path / "output"
    dataset_root = tmp_path / "Dataset"
    chikusei = np.zeros((2, 3, 4, 4), dtype=np.float32)
    chikusei[:, 0] = 1.0
    pavia = np.zeros((2, 2, 4, 4), dtype=np.float32)
    pavia[:, 0] = 1.0
    _write_h5(dataset_root / "chikusei" / "test.h5", "HRHS", chikusei)
    _write_h5(dataset_root / "pavia" / "test.h5", "HRHS", pavia)
    _write_h5(output_root / "chikusei_x4" / "shared" / "predictions.h5", "prediction", chikusei)
    _write_h5(
        output_root / "chikusei_x4" / "extra" / "predictions.h5",
        "prediction",
        np.full_like(chikusei, 0.5),
    )
    _write_h5(output_root / "pavia_x4" / "shared" / "predictions.h5", "prediction", pavia)

    data = load_sample_pixel_sam_data(
        output_root=output_root,
        dataset_root=dataset_root,
        selections=[("chikusei_x4", 1), ("pavia_x4", 0)],
        methods=["shared", "extra"],
    )

    assert data.methods == ["shared"]
    assert [sample.benchmark for sample in data.samples] == ["chikusei_x4", "pavia_x4"]
    assert [sample.image_index for sample in data.samples] == [1, 0]
    assert data.samples[0].pixel_count == 16
    assert data.samples[1].pixel_count == 16

    figure = create_sample_comparison_figure(
        data.samples, methods=data.methods, highlight_methods=["shared"], log_scale=False
    )
    FigureCanvasAgg(figure)
    figure.canvas.draw()

    assert len(figure.axes) == 2
    labels = [axis.get_ylabel() for axis in figure.axes]
    assert labels == ["SAM (°)", "SAM (°)"]


def test_comparison_figure_accepts_at_most_six_samples(tmp_path) -> None:
    output_root = tmp_path / "output"
    dataset_root = tmp_path / "Dataset"
    target = np.zeros((_MAX_SAMPLES + 1, 2, 4, 4), dtype=np.float32)
    _write_h5(dataset_root / "chikusei" / "test.h5", "HRHS", target)
    _write_h5(output_root / "chikusei_x4" / "method" / "predictions.h5", "prediction", target)

    with pytest.raises(ValueError):
        load_sample_pixel_sam_data(
            output_root=output_root,
            dataset_root=dataset_root,
            selections=[("chikusei_x4", index) for index in range(_MAX_SAMPLES + 1)],
        )


def test_six_sample_figure_lays_out_two_rows(tmp_path) -> None:
    output_root = tmp_path / "output"
    dataset_root = tmp_path / "Dataset"
    target = np.zeros((6, 2, 4, 4), dtype=np.float32)
    target[:, 0] = 1.0
    _write_h5(dataset_root / "chikusei" / "test.h5", "HRHS", target)
    _write_h5(output_root / "chikusei_x4" / "method" / "predictions.h5", "prediction", target)

    data = load_sample_pixel_sam_data(
        output_root=output_root,
        dataset_root=dataset_root,
        selections=[("chikusei_x4", index) for index in range(6)],
    )
    figure = create_sample_comparison_figure(data.samples, methods=data.methods)

    FigureCanvasAgg(figure)
    figure.canvas.draw()
    assert len(figure.axes) == 6


def test_method_selection_exposes_available_candidates(tmp_path) -> None:
    output_root = tmp_path / "output"
    prediction = output_root / "chikusei_x4" / "CyberFuse" / "predictions.h5"
    prediction.parent.mkdir(parents=True)
    prediction.touch()
    candidates = discover_output_predictions(output_root)["chikusei_x4"]

    assert select_predictions(candidates, ["cyberfuse"]) == [("CyberFuse", prediction)]


def test_cli_headless_render_writes_figure(tmp_path) -> None:
    output_root = tmp_path / "output"
    dataset_root = tmp_path / "Dataset"
    target = np.zeros((1, 2, 4, 4), dtype=np.float32)
    target[0, 0] = 1.0
    _write_h5(dataset_root / "chikusei" / "test.h5", "HRHS", target)
    _write_h5(output_root / "chikusei_x4" / "method" / "predictions.h5", "prediction", target)

    figure_dir = tmp_path / "figures"
    exit_code = main(
        [
            "--output-root", str(output_root),
            "--dataset-root", str(dataset_root),
            "--selection", "chikusei_x4:0",
            "--output-directory", str(figure_dir),
            "--format", "png",
            "--dpi", "80",
            "--render-only",
        ]
    )

    assert exit_code == 0
    rendered = list(figure_dir.glob("pixel_sam_comparison_*.png"))
    assert len(rendered) == 1
    assert rendered[0].stat().st_size > 0
