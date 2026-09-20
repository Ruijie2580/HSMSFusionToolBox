# HSMSFusionToolBox Architecture

## Extension boundaries

- `models/<name>/` is the complete implementation unit for one paper model.
  Its Lightning module owns `training_step`, losses, and optimizers; its
  network blocks remain local to that folder.
- `MODEL_REGISTRY` discovers model packages automatically. Missing model-only
  imports are reported as `unavailable` by `hsmsfusion list-models`.
- `DATASET_REGISTRY` currently exposes only `chikusei`.
- `SPECTRAL_OPERATOR_REGISTRY` and `SPATIAL_OPERATOR_REGISTRY` are composed by
  `ObservationGenerator`.
- `METRIC_REGISTRY` pins the versioned default benchmark metrics but permits
  additional diagnostics.

## Chikusei v1 protocol

- HDF5 contains NCHW `HRHS`; the DataModule materializes `HRMS` with a
  Gaussian WV2 SRF and `LRHS` with Gaussian blur plus x4 decimation at offset
  `[0, 0]`.
- The concrete SRF and PSF values are configuration fields, never model code.
- `operator_access=known` provides the degradation configuration to the model;
  `hidden` withholds it. They must use separate benchmark names and result
  tables.
- Official runs have no framework-side geometric augmentation because the
  supplied training patches are already augmented.

## Runtime and outputs

- Hydra composes one `experiment=<dataset>/<model>_x<ratio>` recipe (e.g.
  `chikusei/clsnet_x4`) from `configs/experiment/<dataset>/<model>_x<ratio>.yaml`.
  The recipe owns model parameters, optimizer, scheduler, loss, checkpoint
  policy, and per-split dataloader settings. Each run is immutable at
  `runs/<experiment>/<timestamp>-<config_hash>/`.
- Lightning runs on one GPU. Full checkpoints are written at epoch boundaries
  as `last.ckpt`; the monitored best checkpoint is `best.ckpt`.
- `resume` restores the original resolved configuration and continues the same
  run directory. It is not a fine-tuning path.
- The independent CPU float32 evaluator reports global-cube PSNR/RMSE,
  band-mean SSIM, SAM in degree/radian, ERGAS with the configured scale, and
  per-band PSNR. Test-image summaries use population standard deviation.
