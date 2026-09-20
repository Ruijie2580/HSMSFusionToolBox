from __future__ import annotations

import h5py
import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg

from scripts.plot_band_psnr import (
    create_band_psnr_preview_figure,
    create_sample_comparison_figure,
    discover_output_predictions,
    load_band_psnr_data,
    load_sample_band_psnr_data,
    render_sample_comparison_figure,
    select_predictions,
    target_sample_count,
    write_per_band_csv,
)


def _write_h5(path, key: str, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset(key, data=values.astype(np.float32))


def test_band_psnr_loader_discovers_case_insensitive_dataset_and_computes_values(tmp_path) -> None:
    output_root = tmp_path / "output"
    dataset_root = tmp_path / "Dataset"
    target = np.zeros((2, 3, 4, 4), dtype=np.float32)
    alpha = target.copy()
    beta = np.full_like(target, 0.5)
    beta[1] = 0.25
    _write_h5(dataset_root / "Chikusei" / "test.h5", "HRHS", target)
    _write_h5(output_root / "chikusei_x4" / "alpha" / "predictions.h5", "prediction", alpha)
    _write_h5(output_root / "chikusei_x4" / "beta" / "predictions.h5", "prediction", beta)

    data = load_band_psnr_data(
        output_root=output_root,
        dataset_root=dataset_root,
        benchmark="CHIKUSEI_X4",
        methods=["beta", "alpha"],
    )

    assert data.methods == ["beta", "alpha"]
    assert data.target_path == dataset_root / "Chikusei" / "test.h5"
    assert data.per_image["alpha"].shape == (2, 3)
    assert np.allclose(data.per_image["alpha"], 80.0)
    assert np.allclose(data.per_image["beta"][0], 6.0206, atol=1e-3)
    assert np.allclose(data.per_image["beta"][1], 12.0412, atol=1e-3)


def test_sample_comparison_export_and_csv(tmp_path) -> None:
    output_root = tmp_path / "output"
    dataset_root = tmp_path / "Dataset"
    target = np.zeros((2, 2, 4, 4), dtype=np.float32)
    _write_h5(dataset_root / "chikusei" / "test.h5", "HRHS", target)
    _write_h5(output_root / "chikusei_x4" / "method_a" / "predictions.h5", "prediction", target)
    _write_h5(
        output_root / "chikusei_x4" / "method_b" / "predictions.h5",
        "prediction",
        np.full_like(target, 0.5),
    )
    data = load_sample_band_psnr_data(
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
    csv_path = write_per_band_csv(
        data.benchmark_data["chikusei_x4"], tmp_path / "figures" / "values.csv"
    )

    assert figure_path.name == "band_psnr_comparison_chikusei_x4-0_chikusei_x4-1.pdf"
    assert figure_path.is_file() and figure_path.stat().st_size > 0
    assert csv_path.read_text(encoding="utf-8").splitlines()[0] == "Method,ImageIndex,Band,PSNR"


def test_samples_may_come_from_different_datasets_and_share_common_methods(tmp_path) -> None:
    output_root = tmp_path / "output"
    dataset_root = tmp_path / "Dataset"
    chikusei = np.zeros((2, 3, 4, 4), dtype=np.float32)  # 3 bands
    pavia = np.zeros((2, 2, 4, 4), dtype=np.float32)  # 2 bands
    _write_h5(dataset_root / "chikusei" / "test.h5", "HRHS", chikusei)
    _write_h5(dataset_root / "pavia" / "test.h5", "HRHS", pavia)
    _write_h5(output_root / "chikusei_x4" / "shared" / "predictions.h5", "prediction", chikusei)
    _write_h5(
        output_root / "chikusei_x4" / "extra" / "predictions.h5",
        "prediction",
        np.full_like(chikusei, 0.5),
    )
    _write_h5(output_root / "pavia_x4" / "shared" / "predictions.h5", "prediction", pavia)

    data = load_sample_band_psnr_data(
        output_root=output_root,
        dataset_root=dataset_root,
        selections=[("chikusei_x4", 1), ("pavia_x4", 0)],
        methods=["shared", "extra"],
    )

    assert data.methods == ["shared"]
    assert [sample.benchmark for sample in data.samples] == ["chikusei_x4", "pavia_x4"]
    assert [sample.image_index for sample in data.samples] == [1, 0]
    assert data.samples[0].band_count == 3
    assert data.samples[1].band_count == 2

    figure = create_sample_comparison_figure(
        data.samples, methods=data.methods, highlight_methods=["shared"]
    )
    FigureCanvasAgg(figure)
    figure.canvas.draw()

    assert len(figure.axes) == 2
    captions = [axis.texts[0].get_text() for axis in figure.axes]
    assert captions == ["(a)", "(b)"]


def test_comparison_figure_accepts_at_most_three_samples(tmp_path) -> None:
    output_root = tmp_path / "output"
    dataset_root = tmp_path / "Dataset"
    target = np.zeros((4, 2, 4, 4), dtype=np.float32)
    _write_h5(dataset_root / "chikusei" / "test.h5", "HRHS", target)
    _write_h5(output_root / "chikusei_x4" / "method" / "predictions.h5", "prediction", target)

    data = load_sample_band_psnr_data(
        output_root=output_root,
        dataset_root=dataset_root,
        selections=[("chikusei_x4", index) for index in range(4)],
    )

    with pytest.raises(ValueError):
        create_sample_comparison_figure(data.samples, methods=data.methods)


def test_selected_samples_are_the_only_ones_loaded(tmp_path) -> None:
    output_root = tmp_path / "output"
    dataset_root = tmp_path / "Dataset"
    target = np.zeros((3, 2, 4, 4), dtype=np.float32)
    prediction = target.copy()
    prediction[2] = 0.5
    _write_h5(dataset_root / "chikusei" / "test.h5", "HRHS", target)
    _write_h5(output_root / "chikusei_x4" / "method" / "predictions.h5", "prediction", prediction)

    data = load_sample_band_psnr_data(
        output_root=output_root,
        dataset_root=dataset_root,
        selections=[("chikusei_x4", 2)],
    )

    assert data.benchmark_data["chikusei_x4"].image_indices == [2]
    assert data.samples[0].curves["method"].shape == (2,)
    assert np.allclose(data.samples[0].curves["method"], 6.0206, atol=1e-3)

    figure_path = render_sample_comparison_figure(
        data, output_directory=tmp_path / "figures", output_format="png", dpi=100,
    )
    assert figure_path.name == "band_psnr_comparison_chikusei_x4-2.png"


def test_method_selection_exposes_available_candidates(tmp_path) -> None:
    output_root = tmp_path / "output"
    prediction = output_root / "chikusei_x4" / "MCTNet" / "predictions.h5"
    prediction.parent.mkdir(parents=True)
    prediction.touch()
    candidates = discover_output_predictions(output_root)["chikusei_x4"]

    assert select_predictions(candidates, ["mctnet"]) == [("MCTNet", prediction)]


def test_target_sample_count_exposes_selectable_image_count(tmp_path) -> None:
    target_path = tmp_path / "test.h5"
    _write_h5(target_path, "HRHS", np.zeros((4, 2, 3, 3), dtype=np.float32))

    assert target_sample_count(target_path) == 4


def test_preview_figure_shows_active_image_curves(tmp_path) -> None:
    output_root = tmp_path / "output"
    dataset_root = tmp_path / "Dataset"
    target = np.zeros((2, 2, 4, 4), dtype=np.float32)
    _write_h5(dataset_root / "chikusei" / "test.h5", "HRHS", target)
    _write_h5(output_root / "chikusei_x4" / "method_a" / "predictions.h5", "prediction", target)
    _write_h5(
        output_root / "chikusei_x4" / "method_b" / "predictions.h5",
        "prediction",
        np.full_like(target, 0.5),
    )
    data = load_band_psnr_data(
        output_root=output_root, dataset_root=dataset_root, benchmark="chikusei_x4"
    )

    figure = create_band_psnr_preview_figure(
        data, image_indices=[1], highlight_methods=["method_b"]
    )
    FigureCanvasAgg(figure)
    figure.canvas.draw()

    assert len(figure.axes) == 1
    assert len(figure.axes[0].lines) == 2


def test_preview_panels_render_exact_4_3_and_honour_figsize(tmp_path) -> None:
    output_root = tmp_path / "output"
    dataset_root = tmp_path / "Dataset"
    target = np.zeros((2, 2, 4, 4), dtype=np.float32)
    _write_h5(dataset_root / "chikusei" / "test.h5", "HRHS", target)
    _write_h5(output_root / "chikusei_x4" / "method_a" / "predictions.h5", "prediction", target)
    _write_h5(
        output_root / "chikusei_x4" / "method_b" / "predictions.h5",
        "prediction",
        np.full_like(target, 0.5),
    )
    data = load_band_psnr_data(
        output_root=output_root, dataset_root=dataset_root, benchmark="chikusei_x4"
    )

    figure = create_band_psnr_preview_figure(
        data, image_indices=[0],
        highlight_methods=["method_b"], figsize=(10.0, 4.2),
    )
    FigureCanvasAgg(figure)  # a real renderer is required to apply the box aspect
    figure.canvas.draw()

    assert tuple(figure.get_size_inches()) == (10.0, 4.2)
    assert len(figure.axes) == 1
    width, height = figure.axes[0].get_window_extent().size
    assert abs(height / width - 3 / 4) < 0.02
