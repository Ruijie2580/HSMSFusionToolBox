from __future__ import annotations

from pathlib import Path

import pytest

from scripts.visualize_tensorboard_runs import (
    MODE_TRAINING_LOSS,
    MODE_VALIDATION_LOSS,
    MODE_VALIDATION_METRIC,
    ScalarPoint,
    TensorBoardRun,
    canonical_x_axis,
    dataset_name_from_run_name,
    deduplicate_scalar_points,
    default_series_selection,
    downsample_series,
    effective_x_axis,
    export_figure,
    filter_runs_by_dataset,
    series_tags_for_mode,
    series_xy,
    subsample_series_by_stride,
)


def _point(step: int, value: float, wall_time: float) -> ScalarPoint:
    return ScalarPoint(step=step, wall_time=wall_time, value=value)


@pytest.mark.parametrize(
    ("run_name", "expected_dataset"),
    [
        ("cauwt_chikusei_x4", "chikusei"),
        ("cauwt_cave_x4/20260804-120000", "cave"),
        ("smoke_cli", "Unknown"),
    ],
)
def test_dataset_name_from_run_name_supports_standard_and_timestamped_records(
    run_name: str,
    expected_dataset: str,
) -> None:
    assert dataset_name_from_run_name(run_name) == expected_dataset


def test_filter_runs_by_dataset_keeps_all_runs_or_only_the_selected_dataset() -> None:
    runs = [
        TensorBoardRun("mctnet_chikusei_x4", Path("test")),
        TensorBoardRun("cauwt_cave_x4", Path("test")),
    ]

    assert filter_runs_by_dataset(runs, "All datasets") == runs
    assert filter_runs_by_dataset(runs, "chikusei") == [runs[0]]
    assert filter_runs_by_dataset(runs, "missing") == []


def test_deduplicate_scalar_points_keeps_the_latest_event_for_each_step() -> None:
    points = [_point(1, 10.0, 1.0), _point(1, 11.0, 2.0), _point(2, 12.0, 1.5)]

    deduplicated = deduplicate_scalar_points(points)

    assert deduplicated == [_point(1, 11.0, 2.0), _point(2, 12.0, 1.5)]


def test_missing_epoch_logs_use_step_and_are_explicitly_detectable() -> None:
    run = TensorBoardRun(
        name="without_epoch",
        log_directory=Path("test"),
        scalars={"train/total": [_point(3, 0.5, 1.0)]},
    )

    assert effective_x_axis(run, "epoch") == "step"
    assert series_xy(run, "train/total", x_axis="epoch") == ([3.0], [0.5])


def test_epoch_mapping_is_used_when_epoch_events_are_available() -> None:
    run = TensorBoardRun(
        name="with_epoch",
        log_directory=Path("test"),
        scalars={
            "epoch": [_point(0, 0.0, 1.0), _point(10, 1.0, 2.0)],
            "val/loss": [_point(5, 0.25, 1.5)],
        },
    )

    assert effective_x_axis(run, "epoch") == "epoch"
    assert series_xy(run, "val/loss", x_axis="epoch") == ([0.5], [0.25])


def test_downsample_series_preserves_endpoints_and_validates_shapes() -> None:
    x_values = list(range(101))
    y_values = [value * 2 for value in x_values]

    sampled_x, sampled_y = downsample_series(x_values, y_values, max_points=11)

    assert len(sampled_x) == len(sampled_y) == 11
    assert (sampled_x[0], sampled_y[0]) == (0, 0)
    assert (sampled_x[-1], sampled_y[-1]) == (100, 200)
    with pytest.raises(ValueError, match="equal lengths"):
        downsample_series([0.0], [0.0, 1.0])


def test_series_tags_for_mode_splits_train_val_and_metric_tags() -> None:
    tags = [
        "epoch",
        "hp_metric",
        "train/total",
        "train/reconstruction",
        "val/total",
        "val/loss",
        "val/psnr_global",
    ]

    assert series_tags_for_mode(MODE_TRAINING_LOSS, tags) == [
        "train/reconstruction",
        "train/total",
    ]
    assert series_tags_for_mode(MODE_VALIDATION_LOSS, tags) == ["val/loss", "val/total"]
    assert series_tags_for_mode(MODE_VALIDATION_METRIC, tags) == ["val/psnr_global"]


def test_default_series_selection_prefers_aggregates_and_headline_metric() -> None:
    assert (
        default_series_selection(["train/loss", "train/total"], MODE_TRAINING_LOSS)
        == ["train/total"]
    )
    assert (
        default_series_selection(
            ["train/loss", "train/reconstruction"], MODE_TRAINING_LOSS
        )
        == ["train/loss"]
    )
    assert (
        default_series_selection(
            ["val/ergas", "val/psnr_global", "val/ssim_band_mean"],
            MODE_VALIDATION_METRIC,
        )
        == ["val/psnr_global"]
    )
    assert default_series_selection([], MODE_TRAINING_LOSS) == []


def test_canonical_x_axis_maps_iter_to_step() -> None:
    assert canonical_x_axis("epoch") == "epoch"
    assert canonical_x_axis("iter") == "step"
    with pytest.raises(ValueError, match="epoch.*iter"):
        canonical_x_axis("banana")


def test_subsample_series_by_stride_keeps_the_last_point_of_each_bucket() -> None:
    x_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    y_values = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]

    assert subsample_series_by_stride(x_values, y_values, 1) == (x_values, y_values)

    sampled_x, sampled_y = subsample_series_by_stride(x_values, y_values, 2)
    assert (sampled_x, sampled_y) == ([1.0, 3.0, 5.0], [11.0, 13.0, 15.0])
    assert sampled_x[-1] == x_values[-1]  # the final point is always preserved


def test_series_xy_applies_stride_sampling_in_epoch_units() -> None:
    run = TensorBoardRun(
        name="with_epoch",
        log_directory=Path("test"),
        scalars={
            "epoch": [_point(0, 0.0, 1.0), _point(10, 1.0, 2.0), _point(20, 2.0, 3.0)],
            "train/total": [
                _point(0, 1.0, 1.0),
                _point(5, 0.9, 2.0),
                _point(10, 0.8, 3.0),
                _point(15, 0.7, 4.0),
                _point(20, 0.6, 5.0),
            ],
        },
    )

    x_values, y_values = series_xy(run, "train/total", x_axis="epoch", sampling_stride=2)

    assert (x_values, y_values) == ([1.5, 2.0], [0.7, 0.6])


def test_export_figure_preserves_the_4_to_3_raster_ratio(tmp_path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from PIL import Image

    figure = Figure(figsize=(8, 6), dpi=180)
    axis = figure.add_subplot(111)
    axis.plot([0, 1], [0, 1])
    output_path = tmp_path / "curve.png"

    export_figure(figure, output_path, "png", 180)

    with Image.open(output_path) as image:
        assert image.size == (1440, 1080)
