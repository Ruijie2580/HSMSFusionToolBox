"""Hydra-backed unified command line interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

import h5py
import hydra
import numpy as np
import torch
from hydra.utils import get_original_cwd
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from hsmsfusion.callbacks import (
    RNGStateCallback,
    EpochMetricReporter,
    EpochRichProgressCallback,
    EpochBoundaryLastCheckpointCallback,
    PredictTimerCallback,
    TrainingVisualizationCallback,
    print_test_report,
    print_training_overview,
)
from hsmsfusion.core.exceptions import ConfigurationError
from hsmsfusion.core.registry import DATASET_REGISTRY, MODEL_REGISTRY
from hsmsfusion.data import discover_datasets
from hsmsfusion.evaluation import write_prediction_file
from hsmsfusion.metrics.evaluator import evaluate_prediction_file
from hsmsfusion.models import discover_models
from hsmsfusion.utils.experiment import (
    as_plain,
    benchmark_output_directory,
    config_hash,
    create_run_directory,
    experiment_directory,
    load_resolved_config,
    save_resolved_config,
    update_manifest,
)


COMMANDS = {"train", "test", "evaluate", "inspect-data", "list-models", "print-config"}


def _mapping(config: Any) -> dict[str, Any]:
    return dict(as_plain(config))


def _training_config(config: DictConfig) -> DictConfig:
    """Return the recipe-owned training section, adapting archived v0 runs."""
    training = config.get("training")
    if training is not None:
        return training

    dataset = _mapping(config.dataset)
    batch_size = int(dataset["batch_size"])
    num_workers = int(dataset.get("num_workers", 0))
    pin_memory = bool(dataset.get("pin_memory", True))
    return OmegaConf.create(
        {
            "trainer": _mapping(config.trainer),
            "optimizer": _mapping(config.optimizer),
            "scheduler": _mapping(config.scheduler),
            "loss": _mapping(config.loss),
            "checkpoint": _mapping(config.checkpoint),
            "dataloader": {
                "materialization_batch_size": int(
                    dataset.get("generation_batch_size", batch_size)
                ),
                "train": {
                    "batch_size": batch_size,
                    "shuffle": True,
                    "pin_memory": pin_memory,
                    "num_workers": num_workers,
                },
                "valid": {
                    "batch_size": batch_size,
                    "shuffle": False,
                    "pin_memory": pin_memory,
                    "num_workers": num_workers,
                },
                "test": {
                    "batch_size": batch_size,
                    "shuffle": False,
                    "pin_memory": pin_memory,
                    "num_workers": num_workers,
                },
            },
        }
    )


def _precision(runtime_config: DictConfig) -> str:
    mode = str(runtime_config.precision)
    if mode == "fp32":
        return "32-true"
    if mode == "fp16-mixed":
        return "16-mixed"
    if mode == "bf16-mixed":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ConfigurationError("bf16-mixed was selected but the current GPU lacks BF16 support.")
        return "bf16-mixed"
    if mode == "auto-mixed":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return "bf16-mixed"
        return "16-mixed"
    raise ConfigurationError(f"Unsupported runtime precision: {mode!r}.")


def _build_model(config: DictConfig):
    discover_models()
    model_class = MODEL_REGISTRY.get(str(config.model.name))
    training_config = _training_config(config)
    operator_access = str(config.benchmark.operator_access)
    if operator_access not in model_class.supported_operator_access:
        raise ConfigurationError(
            f"Model {config.model.name!r} does not support operator_access={operator_access!r}."
        )
    return model_class(
        model_config=_mapping(config.model),
        optimizer_config=_mapping(training_config.optimizer),
        scheduler_config=_mapping(training_config.scheduler),
        loss_config=_mapping(training_config.loss),
        metric_config=_mapping(config.metrics),
        scale=int(config.degradation.spatial.scale),
        operator_config=(
            _mapping(config.degradation)
            if operator_access == "known"
            else None
        ),
    )


def _build_datamodule(config: DictConfig, *, requires_gt: bool):
    discover_datasets()
    data_class = DATASET_REGISTRY.get(str(config.dataset.name))
    training_config = _training_config(config)
    datamodule = data_class(
        _mapping(config.dataset),
        _mapping(config.degradation),
        _mapping(training_config.dataloader),
    )
    datamodule.configure_model_visibility(requires_gt=requires_gt)
    return datamodule


def _logger_list(run_directory: Path) -> list[Any]:
    logs_directory = run_directory / "logs"
    return [
        CSVLogger(save_dir=str(logs_directory), name="csv", version=""),
        TensorBoardLogger(save_dir=str(logs_directory), name="tensorboard", version=""),
    ]


def _checkpoint_callback(config: DictConfig, run_directory: Path) -> ModelCheckpoint:
    checkpoint_config = _training_config(config).checkpoint
    return ModelCheckpoint(
        dirpath=str(run_directory / "checkpoints"),
        filename="best",
        monitor=str(checkpoint_config.monitor),
        mode=str(checkpoint_config.mode),
        save_top_k=1,
        save_last=False,
        save_weights_only=False,
        every_n_epochs=1,
        save_on_train_epoch_end=False,
        auto_insert_metric_name=False,
        enable_version_counter=False,
    )


def _last_checkpoint_callback(run_directory: Path) -> EpochBoundaryLastCheckpointCallback:
    """Save ``last.ckpt`` at every training-epoch boundary."""

    return EpochBoundaryLastCheckpointCallback(
        run_directory / "checkpoints" / "last.ckpt"
    )


def _validation_interval(trainer_config: DictConfig) -> int:
    """Validate the model-owned epoch cadence before constructing Trainer."""

    try:
        interval = int(trainer_config.get("check_val_every_n_epoch", 1))
    except (TypeError, ValueError) as error:
        raise ConfigurationError(
            "training.trainer.check_val_every_n_epoch must be a positive integer."
        ) from error
    if interval < 1:
        raise ConfigurationError(
            "training.trainer.check_val_every_n_epoch must be a positive integer."
        )
    return interval


def _trainer_for_fit(config: DictConfig, run_directory: Path) -> tuple[Trainer, ModelCheckpoint, str]:
    actual_precision = _precision(config.runtime)
    checkpoint_callback = _checkpoint_callback(config, run_directory)
    last_checkpoint_callback = _last_checkpoint_callback(run_directory)
    trainer_config = _training_config(config).trainer
    validation_interval = _validation_interval(trainer_config)
    max_epochs = int(trainer_config.max_epochs)
    if validation_interval > max_epochs:
        raise ConfigurationError(
            "training.trainer.check_val_every_n_epoch cannot exceed max_epochs; "
            "otherwise no validation or best.ckpt would be produced."
        )
    callbacks: list[Any] = [
        TrainingVisualizationCallback(run_directory),
        EpochMetricReporter(),
        checkpoint_callback,
        last_checkpoint_callback,
        RNGStateCallback(),
    ]
    if bool(trainer_config.get("progress_bar", True)):
        callbacks.insert(0, EpochRichProgressCallback())
    trainer = Trainer(
        accelerator=str(config.runtime.accelerator),
        devices=int(config.runtime.devices),
        strategy="auto",
        precision=actual_precision,
        max_epochs=max_epochs,
        check_val_every_n_epoch=validation_interval,
        deterministic=False,
        benchmark=False,
        logger=_logger_list(run_directory),
        callbacks=callbacks,
        default_root_dir=str(run_directory),
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=int(trainer_config.log_every_n_steps),
    )
    return trainer, checkpoint_callback, actual_precision


def _checkpoint_progress(path: Path) -> dict[str, int]:
    """Read trusted local checkpoint progress for the run manifest."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "epoch": int(state.get("epoch", -1)),
        "global_step": int(state.get("global_step", -1)),
    }


def _run_train(config: DictConfig) -> None:
    resume_value = config.get("resume")
    replaced_existing_record = False
    if resume_value:
        run_directory = Path(str(resume_value)).resolve()
        config = load_resolved_config(run_directory)
        checkpoint_path = run_directory / "checkpoints" / "last.ckpt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Cannot resume: missing {checkpoint_path}")
    else:
        run_directory, replaced_existing_record = create_run_directory(config)
        save_resolved_config(config, run_directory)
        checkpoint_path = None

    seed_everything(int(config.seed), workers=False)
    model = _build_model(config)
    datamodule = _build_datamodule(config, requires_gt=model.requires_gt)
    trainer, callback, actual_precision = _trainer_for_fit(config, run_directory)
    training_config = _training_config(config)
    total_parameters_million = sum(parameter.numel() for parameter in model.parameters()) / 1e6
    trainable_parameters_million = (
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        / 1e6
    )
    print_training_overview(
        model_name=str(config.model.name),
        model_class_name=type(model).__name__,
        model_parameters=_mapping(config.model.params),
        total_parameters_million=total_parameters_million,
        trainable_parameters_million=trainable_parameters_million,
        dataset_name=str(config.dataset.name),
        dataset_root=str(config.dataset.root),
        train_dataloader_config=_mapping(training_config.dataloader.train),
        valid_dataloader_config=_mapping(training_config.dataloader.valid),
        materialization_batch_size=int(training_config.dataloader.materialization_batch_size),
        validation_every_n_epochs=_validation_interval(training_config.trainer),
        spectral_config=_mapping(config.degradation.spectral),
        spatial_config=_mapping(config.degradation.spatial),
        operator_access=str(config.benchmark.operator_access),
        run_directory=str(run_directory),
    )
    update_manifest(
        run_directory,
        experiment={
            "name": str(config.experiment.name),
            "config_hash": config_hash(config),
            "seed": int(config.seed),
            "operator_access": str(config.benchmark.operator_access),
        },
        runtime={
            "precision_requested": str(config.runtime.precision),
            "precision_actual": actual_precision,
            "deterministic_algorithms": False,
            "cudnn_benchmark": True,
            "bitwise_reproducible": False,
        },
        checkpoint={
            "monitor": str(training_config.checkpoint.monitor),
            "mode": str(training_config.checkpoint.mode),
            "uses_validation_gt": bool(training_config.checkpoint.uses_validation_gt),
            "validation_every_n_epochs": _validation_interval(training_config.trainer),
        },
        logging={
            "figures_directory": "logs/figures",
            "replaced_existing_record": replaced_existing_record,
        },
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=checkpoint_path)
    last_progress = _checkpoint_progress(run_directory / "checkpoints" / "last.ckpt")
    best_progress = _checkpoint_progress(run_directory / "checkpoints" / "best.ckpt")
    update_manifest(
        run_directory,
        last_checkpoint={
            "path": "checkpoints/last.ckpt",
            **last_progress,
        },
        best_checkpoint={
            "path": "checkpoints/best.ckpt",
            **best_progress,
            "monitor": str(training_config.checkpoint.monitor),
            "mode": str(training_config.checkpoint.mode),
            "value": None if callback.best_model_score is None else float(callback.best_model_score),
            "uses_validation_gt": bool(training_config.checkpoint.uses_validation_gt),
        },
    )


def _load_run_configuration(config: DictConfig) -> tuple[Path, DictConfig]:
    run_directory = (
        Path(str(config.run_dir)).resolve()
        if config.get("run_dir")
        else experiment_directory(config)
    )
    return run_directory, load_resolved_config(run_directory)


def _run_test(config: DictConfig) -> None:
    run_directory, run_config = _load_run_configuration(config)
    # Ensure metrics added after the run was saved are still evaluated.
    if "uiqi" not in list(run_config.metrics.leaderboard):
        run_config.metrics.leaderboard.append("uiqi")
    # Inject inference config from current experiment if missing in saved run.
    if "inference" not in run_config.model:
        current_inference = config.model.get("inference", {})
        if current_inference:
            run_config.model.inference = current_inference
    checkpoint_name = str(config.get("checkpoint_name", "best"))
    if checkpoint_name not in {"best", "last"}:
        raise ConfigurationError("checkpoint_name must be 'best' or 'last'.")
    checkpoint_path = run_directory / "checkpoints" / f"{checkpoint_name}.ckpt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing requested checkpoint: {checkpoint_path}")

    model = _build_model(run_config)
    datamodule = _build_datamodule(run_config, requires_gt=model.requires_gt)
    timer = PredictTimerCallback()
    trainer = Trainer(
        accelerator=str(run_config.runtime.accelerator),
        devices=int(run_config.runtime.devices),
        strategy="auto",
        precision=_precision(run_config.runtime),
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[timer],
        default_root_dir=str(run_directory),
    )
    # Test only needs model weights. Loading explicitly keeps inference compatible
    # with PyTorch's safe checkpoint defaults while resume restores full state.
    checkpoint_state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint_state["state_dict"], strict=True)
    batches = trainer.predict(model, datamodule=datamodule)
    total_wall = sum(timer.batch_times)
    per_sample_times: list[float] = []
    for t, s in zip(timer.batch_times, timer.batch_sizes):
        per_sample_times.append(t / s)
    per_sample_times.sort()
    trimmed = per_sample_times[1:-1] if len(per_sample_times) > 2 else per_sample_times
    trimmed_ms = float(np.mean(trimmed)) * 1000.0 if trimmed else None
    sample_count = sum(timer.batch_sizes)
    prediction = torch.cat(batches, dim=0).numpy().astype(np.float32, copy=False)
    output_directory = benchmark_output_directory(run_config)
    prediction_path = output_directory / "predictions.h5"
    write_prediction_file(prediction_path, prediction)
    result = evaluate_prediction_file(
        prediction_path=prediction_path,
        target_path=Path(str(run_config.dataset.root)) / "test.h5",
        metric_config=_mapping(run_config.metrics),
        scale=int(run_config.degradation.spatial.scale),
        output_directory=output_directory,
    )
    print_test_report(
        result=result,
        model_name=str(run_config.model.name),
        benchmark_name=str(run_config.benchmark.name),
        total_wall_seconds=total_wall,
        trimmed_per_sample_ms=trimmed_ms,
        sample_count=sample_count,
    )
    update_manifest(
        run_directory,
        test={
            "checkpoint": checkpoint_name,
            "output_directory": str(output_directory),
            "prediction_path": str(prediction_path),
            "metrics_path": str(output_directory / "metrics.csv"),
            "metrics_summary": result.summary,
        },
    )


def _run_evaluate(config: DictConfig) -> None:
    run_directory, run_config = _load_run_configuration(config)
    if "uiqi" not in list(run_config.metrics.leaderboard):
        run_config.metrics.leaderboard.append("uiqi")
    if "inference" not in run_config.model:
        current_inference = config.model.get("inference", {})
        if current_inference:
            run_config.model.inference = current_inference
    output_directory = benchmark_output_directory(run_config)
    prediction_path = (
        Path(str(config.prediction_path)).resolve()
        if config.get("prediction_path")
        else output_directory / "predictions.h5"
    )
    result = evaluate_prediction_file(
        prediction_path=prediction_path,
        target_path=Path(str(run_config.dataset.root)) / "test.h5",
        metric_config=_mapping(run_config.metrics),
        scale=int(run_config.degradation.spatial.scale),
        output_directory=output_directory,
    )
    print_test_report(
        result=result,
        model_name=str(run_config.model.name),
        benchmark_name=str(run_config.benchmark.name),
        total_wall_seconds=0.0,
        trimmed_per_sample_ms=None,
        sample_count=len(result.per_image),
    )
    update_manifest(
        run_directory,
        evaluation={
            "output_directory": str(output_directory),
            "metrics_path": str(output_directory / "metrics.csv"),
            "metrics_summary": result.summary,
        },
    )


def _run_inspect_data(config: DictConfig) -> None:
    root = Path(str(config.dataset.root))
    result: dict[str, Any] = {}
    for split in ("train", "valid", "test"):
        path = root / f"{split}.h5"
        with h5py.File(path, "r") as handle:
            dataset = handle["HRHS"]
            result[split] = {
                "shape": list(dataset.shape),
                "dtype": str(dataset.dtype),
                "chunks": list(dataset.chunks) if dataset.chunks else None,
                "compression": dataset.compression,
            }
    print(json.dumps(result, indent=2))


def _run_list_models(config: DictConfig) -> None:
    discover_models()
    for status in MODEL_REGISTRY.statuses():
        detail = f"  {status.reason}" if status.reason else ""
        print(f"{status.name:<24} {status.status}{detail}")


def _run_print_config(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config, resolve=True))


def _dispatch(command: str, config: DictConfig) -> None:
    handlers: dict[str, Callable[[DictConfig], None]] = {
        "train": _run_train,
        "test": _run_test,
        "evaluate": _run_evaluate,
        "inspect-data": _run_inspect_data,
        "list-models": _run_list_models,
        "print-config": _run_print_config,
    }
    handlers[command](config)


def _run_hydra(command: str) -> None:
    config_directory = str(Path(__file__).resolve().parent.parent / "configs")

    @hydra.main(version_base="1.3", config_path=config_directory, config_name="config")
    def app(config: DictConfig) -> None:
        _dispatch(command, config)

    app()


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        commands = ", ".join(sorted(COMMANDS))
        raise SystemExit(f"Usage: hsmsfusion <{commands}> [Hydra overrides]")
    command = sys.argv[1]
    del sys.argv[1]
    _run_hydra(command)


if __name__ == "__main__":
    main()
