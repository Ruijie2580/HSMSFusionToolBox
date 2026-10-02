# HSMSFusionToolBox

A configuration-driven toolbox for supervised and dataset-level unsupervised
hyperspectral--multispectral image fusion experiments. The first benchmark is
Chikusei with the bundled measured WorldView-2 spectral response table and a
fixed Gaussian blur/decimation spatial observation model.

## Installation

Use the project Python environment:

```powershell
conda activate vmamba
pip install --no-build-isolation -e .
```

## Configuration layout

The reusable global groups define only the dataset, observation degradation,
runtime, and evaluation protocol. A selected
`<dataset>/<model>_x<ratio>` experiment recipe (from `configs/experiment/<dataset>/`) owns the model parameters,
epochs, optimizer, scheduler, loss, checkpoint policy, and train/valid/test
dataloader settings.

Current recipes:

- `chikusei/cyberfuse_x4`

The Chikusei reference protocol materializes HRMS and LRHS in CPU memory once
per process, then transfers pinned batches to one GPU.

The Chikusei degradation recipes use `tabulated_srf` with
`hsmsfusion/degradations/srf/WV2_srf.mat`; it interpolates the measured response
curves onto the configured HSI wavelength grid before row normalization. The
other bundled sensor files can be selected by changing `response_file`, sensor,
and expected MSI band count together.

The PaviaC dataset uses the 102-band HRHS splits under `Dataset/PaviaC`;
`configs/degradation/paviac_qb_x4.yaml` maps the 430--840 nm HSI grid to four
QuickBird MSI bands with `hsmsfusion/degradations/srf/QuickBird_srf.mat` and uses
the same fixed Gaussian blur/decimation protocol at scale 4. The current PaviaC
H5 files contain HRHS only, so HRMS and LRHS are materialized from this
configured observation model.

The ICVL dataset uses the 31-band NCHW HRHS splits under `Dataset/ICVL`, whose
standard wavelength grid is 400--700 nm. `configs/degradation/icvl_nikon_x4.yaml`
synthesizes three-band Nikon D700 HRMS observations and Gaussian x4 LRHS
observations.

## Utility scripts

Reusable, non-training tools live in `scripts/`. The first tool renders a
two-row residual comparison for one test image: predicted RGB on the first row
and mean absolute spectral residual on the second. Every method shares one
residual color scale, so the maps are directly comparable.
The script is GUI-only: run it without arguments. It scans
`output/` for candidate dataset-scale folders, provides a checkbox for each
method with `predictions.h5`, infers the matching `Dataset/<dataset>/test.h5`,
and lets you choose an image index. Method labels use the canonical names from
`hsmsfusion list-models`. Drag a rectangle on any panel to apply a square zoom
region to every method; both the source square and the inset have red borders.
The GUI also controls RGB contrast stretching, residual/SAM display ranges,
optional per-pixel SAM (radians) row, PNG/JPEG/PDF export, and DPI. Figures use
Times New Roman throughout.

```powershell
python scripts/plot_residual_comparison.py
```

`plot_band_psnr.py` reads the same exported `predictions.h5` files and the
matching `Dataset/<dataset>/test.h5` reference cube to compute PSNR per
spectral band. It is GUI-only: it scans available dataset-scale folders,
lets you check methods and optional highlighted methods, and presents a
multi-select list of test images with select-all, clear, and invert controls.
Only the selected image curves (plus the optional average curve) are computed
and exported. `Preview PSNR` displays the active selected image's per-band
curve directly in the GUI; when the average option is enabled, the preview
also shows the average curve. The GUI controls output directory, format, DPI,
and CSV export.

```powershell
python scripts/plot_band_psnr.py
```

`plot_pixel_sam.py` draws sorted per-pixel SAM (spectral angle, degrees)
curves for every selected fusion method. Each sample's HxW SAM map is
flattened, sorted in ascending order, and plotted against the pixel index;
up to six samples share a 2x3 panel grid and one figure-level legend. A
highlighted method is drawn as a thick solid red line while the remaining
methods stay dashed. The GUI scans `output/` for dataset-scale folders,
lets you check methods and highlight any of them, and supports multi-dataset
sample comparison (max 6). Y-axis log scale, [0, 1] clipping, format, and
DPI are all controlled from the GUI.

```powershell
python scripts/plot_pixel_sam.py
```

`plot_metric_summary.py`汇总同一 dataset-scale 下各融合方法的测试指标（PSNR / SSIM / SAM / ERGAS / RMSE / UIQI 等，以 `metrics.csv` 实际列为准），绘制一指标一子图的箱线图，并导出 mean ± std 汇总表。优先读取 `metrics.csv`；缺失时自动从 `predictions.h5` 重算并写回。GUI 支持方法多选与拖拽排序、高亮、指标筛选；也可用 CLI：`python scripts/plot_metric_summary.py --benchmark chikusei_x4 --output-dir figures/metrics`。

```powershell
python scripts/plot_metric_summary.py
```

`visualize_tensorboard_runs.py` opens a desktop GUI for browsing the
TensorBoard event files under `runs/` (or `run/`). It supports multi-run
comparison, independent training-loss, validation-loss, and validation-metric
plots, epoch/step selection, curve smoothing, optional per-run metric
normalization, grid control, and PNG/JPEG/PDF export with selectable DPI. Use
the Dataset dropdown to restrict the experiment list and all plots to one
dataset; timestamped run records are grouped by their parent experiment. The
display and exported figures preserve a standard 4:3 layout.

```powershell
python scripts/visualize_tensorboard_runs.py
```

## Loss manager

Each experiment recipe composes its objective from two explicit groups under
`training.loss`; the model does not manually combine standard terms. For
example:

```yaml
training:
  loss:
    reconstruction:
    - name: l1
      alias: reconstruction
      weight: 1.0
    - name: cycle_consistency
      alias: observation_cycle
      weight: 0.1
      params:
        spectral_weight: 1.0
        spatial_weight: 1.0
    regularization:
    - name: sam
      alias: spectral_angle
      weight: 0.05
```

The reconstruction group provides `l1`, `l2`, `char` (`params.epsilon`), and
`cycle_consistency`. The latter re-observes the prediction as HRMS and LRHS
through the fixed SRF and blur/downsampling operators; it requires
`benchmark.operator_access=known`. The regularization group provides `sam`
(`params.epsilon`), differentiable per-band `ssim` (window and Gaussian
parameters are configurable), `spatial_edge`, `spectral_edge`, and spatial
`tv`. Edge losses and TV accept `params.criterion: l1 | mse`; the same option
is available for `cycle_consistency`. Both groups are optional, but at least
one term is required. A term must be placed in its matching group.

The prior `kind`/`weight` and flat `components` loss formats have been removed;
convert old recipes to the two lists above. The train/validation logs include
each raw component, its weighted contribution, and `total`.
The Rich epoch summary prints every configured training loss term and its
weighted contribution.

To add a reusable custom term, implement `ConfigurableLossTerm` and register
it with `@LOSS_REGISTRY.register("your_loss")`, set its `category` to either
`"reconstruction"` or `"regularization"`, and it becomes selectable by the
corresponding YAML list.

## CLI

Inspect available extensions and a resolved configuration:

```powershell
hsmsfusion list-models
hsmsfusion inspect-data experiment=chikusei/clsnet_x4
hsmsfusion print-config experiment=chikusei/clsnet_x4
```

Start a new training run:

```powershell
hsmsfusion train experiment=chikusei/clsnet_x4
hsmsfusion train experiment=chikusei/mctnet_x4
```

Hydra overrides modify the selected recipe for that new run. Examples:

```powershell
hsmsfusion train experiment=chikusei/clsnet_x4 training.optimizer.lr=5e-5
hsmsfusion train experiment=chikusei/clsnet_x4 training.trainer.max_epochs=300
hsmsfusion train experiment=chikusei/clsnet_x4 training.trainer.check_val_every_n_epoch=5
hsmsfusion train experiment=chikusei/clsnet_x4 training.dataloader.train.batch_size=4
hsmsfusion train experiment=chikusei/clsnet_x4 runtime.precision=bf16-mixed
```

Before fitting, the terminal renders a Rich setup card with the selected model,
its total/trainable parameter counts in M, the dataloaders, and parsed
SRF/blur/downsampling configuration. During fitting, Rich renders a compact
dashboard with epoch, colored batch progress, current batch loss, elapsed time,
and ETA. Each completed epoch then renders a color-coded Rich summary card with
the average training loss, training-only time, and full validation metric
names; training completion renders the accumulated training-only time. Set
`training.trainer.progress_bar=false` to suppress only the live progress row.
Set `training.trainer.check_val_every_n_epoch=<N>` to validate every `N`
training epochs; `N` must be a positive integer. `last.ckpt` remains an
epoch-boundary recovery checkpoint, while `best.ckpt` updates only after an
executed validation pass. `N` may not exceed `training.trainer.max_epochs`.

Each experiment has exactly one record at `runs/<experiment>/`. Starting a
non-resume run deletes the existing record with that experiment name—its
checkpoints, logs, predictions, manifest, and resolved configuration—then
creates a fresh record. Resume is intentionally strict: it restores `last.ckpt`,
optimizer, scheduler, epoch, step, RNG state, and the original
`resolved_config.yaml`. Do not pass recipe overrides when resuming.

```powershell
hsmsfusion train resume=runs/clsnet_chikusei_x4
```

Resumed runs do not delete or recreate the existing experiment record.

Training visual artifacts are written under `logs/figures/`: the fixed SRF is
saved as `srf_curve.png`, and each validation epoch saves
`validation/epoch_XXXX.png`. The validation figure contains the complete first
validation batch. Each sample is a compact four-panel group—HRHS, LRHS
(upsampled for viewing), HRMS, and fused HSI—and two sample groups are placed
per row (so a batch of 16 uses eight rows).

Test uses `best.ckpt` by default and writes predictions separately from the
training record. The default Chikusei x4 MCTNet layout is
`output/chikusei_x4/mctnet/`; its contents are `predictions.h5` (NCHW float32)
and the single benchmark report `metrics.csv`. The report contains one row per
image followed by `Average` and population `Std`; its default columns are
`Image, PSNR, SSIM, *SAM, *ERGAS, RMSE`. Use `checkpoint_name=last` to select
the epoch-boundary recovery checkpoint:

```powershell
hsmsfusion test experiment=chikusei/clsnet_x4
hsmsfusion test experiment=chikusei/clsnet_x4 checkpoint_name=last
```

Recompute metrics without running a model, optionally from a separately
provided prediction file:

```powershell
hsmsfusion evaluate experiment=chikusei/clsnet_x4
hsmsfusion evaluate experiment=chikusei/clsnet_x4 prediction_path=D:\results\prediction.h5
```

`predictions.h5` contains exactly one dataset: `/prediction` with float32 NCHW
layout. The evaluator writes no per-band PSNR or JSON sidecar files; all metric
rows and aggregate statistics are contained in `metrics.csv`.
