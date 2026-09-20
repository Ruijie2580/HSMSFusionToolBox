from __future__ import annotations

import matplotlib.pyplot as plt
import h5py
import numpy as np
import pytest

from scripts.plot_residual_comparison import (
    ResidualComparisonConfig,
    _inset_bbox,
    _rgb_image,
    create_residual_comparison_figure,
    discover_output_predictions,
    infer_target_path,
    load_comparison_data,
    parse_metric_range,
    parse_rgb_bands,
    parse_stretch_percentiles,
    render_residual_comparison,
    selection_to_zoom_bounds,
)


def _write_hsi(path, key: str, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset(key, data=array.astype(np.float32))


def test_residual_comparison_script_writes_a_shared_scale_figure(tmp_path) -> None:
    target = np.full((1, 4, 8, 8), 0.5, dtype=np.float32)
    prediction_a = target.copy()
    prediction_b = target.copy()
    prediction_b[:, :, 2:6, 2:6] += 0.1

    target_path = tmp_path / "test.h5"
    prediction_a_path = tmp_path / "method_a.h5"
    prediction_b_path = tmp_path / "method_b.h5"
    output_path = tmp_path / "comparison.png"
    _write_hsi(target_path, "HRHS", target)
    _write_hsi(prediction_a_path, "prediction", prediction_a)
    _write_hsi(prediction_b_path, "prediction", prediction_b)

    config = ResidualComparisonConfig(
        target=target_path,
        prediction=[f"MethodA={prediction_a_path}", f"MethodB={prediction_b_path}"],
        rgb_bands=(3, 2, 1),
        residual_vmax=None,
        include_sam=True,
        sam_vmin=0.0,
        sam_vmax=0.5,
        zoom=(2, 2, 4, 4),
        dpi=100,
        output=output_path,
    )
    written_path = render_residual_comparison(config)

    assert written_path == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 0

    comparison = load_comparison_data(config)
    assert comparison.sam_maps is not None
    assert comparison.sam_maps[0].shape == (8, 8)
    figure, axes = create_residual_comparison_figure(
        comparison,
        config,
        zoom_bounds=selection_to_zoom_bounds(2.0, 2.0, 6.0, 6.0, shape=(8, 8)),
    )
    figure.canvas.draw()
    assert axes.shape == (3, 2)
    for column in range(axes.shape[1]):
        top_position = axes[0, column].get_position()
        bottom_position = axes[1, column].get_position()
        assert top_position.x0 == pytest.approx(bottom_position.x0)
        assert top_position.width == pytest.approx(bottom_position.width)
    figure.clear()


def test_drag_coordinates_convert_to_a_valid_zoom_region() -> None:
    assert selection_to_zoom_bounds(12.2, 20.8, 4.1, 5.3, shape=(32, 32)) == (5, 0, 16, 16)
    assert selection_to_zoom_bounds(1.0, 1.0, 1.5, 1.5, shape=(32, 32)) is None


def test_load_requires_at_least_one_prediction(tmp_path) -> None:
    target_path = tmp_path / "test.h5"
    _write_hsi(target_path, "HRHS", np.full((1, 3, 4, 4), 0.5, dtype=np.float32))

    config = ResidualComparisonConfig(target=target_path)

    with pytest.raises(ValueError, match="at least one"):
        load_comparison_data(config)


def test_rgb_cache_is_used_and_recomputed_when_settings_change(tmp_path) -> None:
    target = np.zeros((1, 3, 8, 8), dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
    target[0, 0] = ramp
    target[0, 1] = ramp.T
    target[0, 2] = ((np.indices((8, 8))[0] + np.indices((8, 8))[1]) % 2).astype(np.float32)
    target_path = tmp_path / "test.h5"
    prediction_path = tmp_path / "method.h5"
    _write_hsi(target_path, "HRHS", target)
    _write_hsi(prediction_path, "prediction", target.copy())

    config = ResidualComparisonConfig(
        target=target_path,
        prediction=[f"Method={prediction_path}"],
        rgb_bands=(2, 1, 0),
        contrast_stretch=False,
        dpi=100,
        output=tmp_path / "comparison.png",
    )
    comparison = load_comparison_data(config)

    assert comparison.rgb_params == ((2, 1, 0), False, (1.0, 99.0))
    assert comparison.rgb_images is not None
    assert len(comparison.rgb_images) == len(comparison.methods)
    assert comparison.rgb_images[0].shape == (8, 8, 3)

    figure, axes = create_residual_comparison_figure(comparison, config, zoom_bounds=None)
    cached_render = np.asarray(axes[0, 0].images[0].get_array())
    assert np.allclose(cached_render, comparison.rgb_images[0])
    plt.close(figure)

    swapped = ResidualComparisonConfig(
        target=target_path,
        prediction=[f"Method={prediction_path}"],
        rgb_bands=(0, 1, 2),
        contrast_stretch=False,
        dpi=100,
        output=tmp_path / "comparison.png",
    )
    figure, axes = create_residual_comparison_figure(comparison, swapped, zoom_bounds=None)
    recomputed = np.asarray(axes[0, 0].images[0].get_array())
    expected = _rgb_image(
        comparison.methods[0][1], (0, 1, 2), contrast_stretch=False, percentiles=(1.0, 99.0)
    )
    assert np.allclose(recomputed, expected)
    assert not np.allclose(recomputed, comparison.rgb_images[0])
    plt.close(figure)


def test_create_figure_can_repopulate_an_existing_figure(tmp_path) -> None:
    target = np.full((1, 3, 8, 8), 0.5, dtype=np.float32)
    target_path = tmp_path / "test.h5"
    prediction_path = tmp_path / "method.h5"
    _write_hsi(target_path, "HRHS", target)
    _write_hsi(prediction_path, "prediction", target)

    config = ResidualComparisonConfig(
        target=target_path,
        prediction=[f"Method={prediction_path}"],
        rgb_bands=(2, 1, 0),
        dpi=100,
        output=tmp_path / "comparison.png",
    )
    comparison = load_comparison_data(config)

    existing = plt.figure(layout="constrained")
    returned, axes = create_residual_comparison_figure(
        comparison, config, zoom_bounds=None, figure=existing
    )

    assert returned is existing
    assert axes.shape == (2, 1)
    assert len(existing.axes) == 3  # two panels plus the residual colorbar
    plt.close(existing)


def test_inset_bbox_rejects_unknown_corner() -> None:
    with pytest.raises(ValueError, match="zoom corner"):
        _inset_bbox("middle center")


def test_parse_helpers_validate_shared_inputs() -> None:
    assert parse_rgb_bands("54, 36, 21") == (54, 36, 21)
    with pytest.raises(ValueError):
        parse_rgb_bands("1, 2")
    assert parse_metric_range("0", "auto", label="Residual range") == (0.0, None)
    with pytest.raises(ValueError):
        parse_metric_range("abc", "1", label="Residual range")
    with pytest.raises(ValueError):
        parse_metric_range("0", "-1", label="Residual range")
    assert parse_stretch_percentiles("1", "99") == (1.0, 99.0)
    with pytest.raises(ValueError):
        parse_stretch_percentiles("99", "1")


def test_gui_helpers_discover_methods_and_infer_the_matching_target(tmp_path) -> None:
    output_root = tmp_path / "output"
    mctnet_prediction = output_root / "chikusei_x4" / "mctnet" / "predictions.h5"
    sdpnet_prediction = output_root / "chikusei_x4" / "sdpnet" / "predictions.h5"
    mctnet_prediction.parent.mkdir(parents=True)
    sdpnet_prediction.parent.mkdir(parents=True)
    mctnet_prediction.touch()
    sdpnet_prediction.touch()
    dataset_root = tmp_path / "Dataset"
    target_path = dataset_root / "Chikusei" / "test.h5"
    target_path.parent.mkdir(parents=True)
    target_path.touch()

    discovered = discover_output_predictions(output_root)

    assert discovered == {
        "chikusei_x4": [
            ("MCTNet", mctnet_prediction),
            ("SDPNet", sdpnet_prediction),
        ]
    }
    assert infer_target_path(dataset_root, "chikusei_x4") == target_path
