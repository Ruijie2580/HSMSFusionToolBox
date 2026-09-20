from __future__ import annotations

import csv

import h5py
import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg

from scripts.plot_metric_summary import (
    best_method_for_metric,
    create_boxplot_figure,
    discover_metric_methods,
    format_summary_cell,
    load_metric_summary_data,
    main,
    metric_display_name,
    metric_higher_is_better,
    normalize_metric_header,
    parse_metrics_csv,
    recompute_metrics_csv,
    render_metric_summary,
    summarize_metric,
    write_summary_csv,
)


def _write_h5(path, key: str, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset(key, data=values.astype(np.float32))


def _write_metrics_csv(path, *, samples: int = 4, psnr_offset: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["Image", "PSNR", "SSIM", "*SAM", "*ERGAS", "RMSE", "UIQI"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for index in range(samples):
            writer.writerow(
                [
                    f"image_{index:04d}",
                    f"{40.0 + psnr_offset + index:.4f}",
                    f"{0.90 + index * 0.01:.4f}",
                    f"{1.0 + index * 0.1:.4f}",
                    f"{2.0 + index * 0.05:.4f}",
                    f"{0.05 + index * 0.001:.4f}",
                    f"{0.80 + index * 0.02:.4f}",
                ]
            )
        writer.writerow(["Average", "41.5000", "0.9150", "1.1500", "2.0750", "0.0515", "0.8300"])
        writer.writerow(["Std", "1.1180", "0.0112", "0.1118", "0.0559", "0.0011", "0.0224"])


def test_normalize_metric_header_strips_star_and_maps_registry_names() -> None:
    assert normalize_metric_header("*SAM") == "sam"
    assert normalize_metric_header("*ERGAS") == "ergas"
    assert normalize_metric_header("psnr_global") == "psnr"
    assert normalize_metric_header("ssim_band_mean") == "ssim"
    assert normalize_metric_header("RMSE") == "rmse"
    assert metric_higher_is_better("psnr") is True
    assert metric_higher_is_better("sam") is False
    assert metric_display_name("sam") == "*SAM"


def test_parse_metrics_csv_ignores_average_and_std_rows(tmp_path) -> None:
    path = tmp_path / "metrics.csv"
    _write_metrics_csv(path, samples=3)

    values = parse_metrics_csv(path)

    assert set(values) == {"psnr", "ssim", "sam", "ergas", "rmse", "uiqi"}
    assert values["psnr"].shape == (3,)
    # image rows only: 40, 41, 42 — not the Average 41.5
    assert np.allclose(values["psnr"], [40.0, 41.0, 42.0])
    assert np.allclose(values["sam"], [1.0, 1.1, 1.2])


def test_loader_prefers_metrics_csv_and_aligns_min_sample_count(tmp_path) -> None:
    output_root = tmp_path / "output"
    dataset_root = tmp_path / "Dataset"
    _write_metrics_csv(
        output_root / "chikusei_x4" / "alpha" / "metrics.csv", samples=5, psnr_offset=0.0
    )
    _write_metrics_csv(
        output_root / "chikusei_x4" / "beta" / "metrics.csv", samples=3, psnr_offset=-2.0
    )
    # predictions.h5 is not required when metrics.csv is present.
    target = np.zeros((5, 3, 8, 8), dtype=np.float32)
    _write_h5(dataset_root / "chikusei" / "test.h5", "HRHS", target)

    data = load_metric_summary_data(
        output_root=output_root,
        dataset_root=dataset_root,
        benchmark="chikusei_x4",
        methods=["alpha", "beta"],
        metrics=["psnr", "ssim", "sam"],
    )

    assert data.methods == ["alpha", "beta"]
    assert data.metrics == ["psnr", "ssim", "sam"]
    assert data.sample_count == 3
    assert data.values["alpha"]["psnr"].shape == (3,)
    assert data.values["beta"]["psnr"].shape == (3,)
    assert any("Sample counts differ" in warning for warning in data.warnings)


def test_missing_metrics_csv_is_recomputed_and_written_back(tmp_path) -> None:
    output_root = tmp_path / "output"
    dataset_root = tmp_path / "Dataset"
    target = np.zeros((2, 4, 16, 16), dtype=np.float32)
    target[:, 0] = 0.5
    target[:, 1] = 0.25
    prediction = target.copy()
    _write_h5(dataset_root / "chikusei" / "test.h5", "HRHS", target)
    method_dir = output_root / "chikusei_x4" / "alpha"
    _write_h5(method_dir / "predictions.h5", "prediction", prediction)
    assert not (method_dir / "metrics.csv").is_file()

    data = load_metric_summary_data(
        output_root=output_root,
        dataset_root=dataset_root,
        benchmark="chikusei_x4",
        methods=["alpha"],
        metrics=["psnr", "ssim", "sam", "ergas", "rmse", "uiqi"],
        prefer_csv=True,
        recompute_missing=True,
    )

    assert (method_dir / "metrics.csv").is_file()
    assert data.sample_count == 2
    # Perfect reconstruction → very high PSNR and near-zero RMSE.
    assert float(np.mean(data.values["alpha"]["psnr"])) > 40.0
    assert float(np.mean(data.values["alpha"]["rmse"])) < 1e-2


def test_discover_metric_methods_includes_csv_only_dirs(tmp_path) -> None:
    output_root = tmp_path / "output"
    _write_metrics_csv(output_root / "chikusei_x4" / "alpha" / "metrics.csv", samples=2)
    _write_h5(
        output_root / "chikusei_x4" / "beta" / "predictions.h5",
        "prediction",
        np.zeros((1, 2, 4, 4), dtype=np.float32),
    )

    discovered = discover_metric_methods(output_root)
    names = [name for name, _ in discovered["chikusei_x4"]]
    assert names == ["alpha", "beta"]


def test_boxplot_axis_count_matches_selected_metrics_and_highlight_style(tmp_path) -> None:
    output_root = tmp_path / "output"
    _write_metrics_csv(output_root / "chikusei_x4" / "alpha" / "metrics.csv", samples=4)
    _write_metrics_csv(
        output_root / "chikusei_x4" / "beta" / "metrics.csv", samples=4, psnr_offset=3.0
    )
    _write_metrics_csv(
        output_root / "chikusei_x4" / "gamma" / "metrics.csv", samples=4, psnr_offset=-5.0
    )
    data = load_metric_summary_data(
        output_root=output_root,
        dataset_root=tmp_path / "Dataset",
        benchmark="chikusei_x4",
        metrics=["psnr", "ssim", "sam", "ergas", "rmse", "uiqi"],
    )

    figure = create_boxplot_figure(
        data, highlight_methods=["beta"], show_mean=True, show_median=True
    )
    FigureCanvasAgg(figure)
    figure.canvas.draw()

    assert len(figure.axes) == len(data.metrics)

    boxes = [
        child
        for axis in figure.axes
        for child in axis.get_children()
        if child.__class__.__name__ == "PathPatch"
    ]
    # 3 methods × 6 metrics = 18 box patches.
    assert len(boxes) == 18
    linewidths = sorted(box.get_linewidth() for box in boxes)
    assert linewidths[-1] >= 2.0
    assert linewidths[-1] > linewidths[0]

    highlight_boxes = [box for box in boxes if box.get_linewidth() >= 2.0]
    assert len(highlight_boxes) == len(data.metrics)
    for box in highlight_boxes:
        r, g, b, _a = box.get_facecolor()
        assert r > 0.7 and r > g and r > b
        edge_r = box.get_edgecolor()[0]
        assert edge_r > 0.7


def test_summary_stats_best_method_and_csv_export(tmp_path) -> None:
    output_root = tmp_path / "output"
    _write_metrics_csv(output_root / "chikusei_x4" / "alpha" / "metrics.csv", samples=4)
    _write_metrics_csv(
        output_root / "chikusei_x4" / "beta" / "metrics.csv", samples=4, psnr_offset=5.0
    )
    data = load_metric_summary_data(
        output_root=output_root,
        dataset_root=tmp_path / "Dataset",
        benchmark="chikusei_x4",
        metrics=["psnr", "sam"],
    )

    assert best_method_for_metric(data, "psnr") == "beta"
    # Lower SAM is better; alpha has smaller SAM than beta (same base, so equal
    # unless we give beta a larger SAM — use identical SAM bases so both equal).
    stats = summarize_metric(data.method_values("alpha", "psnr"))
    assert np.isfinite(stats["mean"])
    cell = format_summary_cell(stats)
    assert "±" in cell

    csv_path = write_summary_csv(data, tmp_path / "summary.csv", highlight_best=True)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["Method", "PSNR", "*SAM"]
    assert rows[1][0] == "alpha"
    assert rows[2][0] == "beta"
    assert "*" in rows[2][1]
    assert rows[-1][0] == "Best"
    assert rows[-1][1] == "beta"


def test_render_metric_summary_writes_figure_and_summary(tmp_path) -> None:
    output_root = tmp_path / "output"
    _write_metrics_csv(output_root / "chikusei_x4" / "alpha" / "metrics.csv", samples=3)
    data = load_metric_summary_data(
        output_root=output_root,
        dataset_root=tmp_path / "Dataset",
        benchmark="chikusei_x4",
    )

    written = render_metric_summary(
        data,
        output_directory=tmp_path / "figures",
        output_format="png",
        dpi=80,
        highlight_methods=["alpha"],
    )

    assert written[0].name == "metric_summary_chikusei_x4.png"
    assert written[1].name == "metric_summary_chikusei_x4_summary.csv"
    assert all(path.is_file() and path.stat().st_size > 0 for path in written)


def test_cli_headless_export(tmp_path) -> None:
    output_root = tmp_path / "output"
    _write_metrics_csv(output_root / "chikusei_x4" / "alpha" / "metrics.csv", samples=3)
    out_dir = tmp_path / "cli_figures"

    code = main(
        [
            "--benchmark",
            "chikusei_x4",
            "--output-root",
            str(output_root),
            "--dataset-root",
            str(tmp_path / "Dataset"),
            "--output-dir",
            str(out_dir),
            "--format",
            "png",
            "--dpi",
            "72",
            "--metric",
            "psnr",
            "--metric",
            "ssim",
        ]
    )

    assert code == 0
    assert (out_dir / "metric_summary_chikusei_x4.png").is_file()
    assert (out_dir / "metric_summary_chikusei_x4_summary.csv").is_file()


def test_recompute_helper_requires_predictions(tmp_path) -> None:
    method_dir = tmp_path / "method"
    method_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        recompute_metrics_csv(
            method_dir=method_dir,
            target_path=tmp_path / "missing.h5",
            scale=4,
        )
