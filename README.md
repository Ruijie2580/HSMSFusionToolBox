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

- `chikusei/cgsfn_x4`
- `chikusei/clsnet_x4`
- `chikusei/lso_x4`
- `chikusei/mctnet_x4`
- `chikusei/s3o_x4`
- `chikusei/scsrnet_x4`
- `chikusei/sdpnet_x4`
- `chikusei/smgu_x4`
- `chikusei/ssilnet_x4`
- `paviac/cauwt_x4`
- `icvl/cauwt_x4`

The Chikusei reference protocol materializes HRMS and LRHS in CPU memory once
per process, then transfers pinned batches to one GPU.

The Chikusei degradation recipes use `tabulated_srf` with
`hsmsfusion/degradations/srf/WV2_srf.mat`; it interpolates the measured response
curves onto the configured HSI wavelength grid before row normalization. The
other bundled sensor files can be selected by changing `response_file`, sensor,
and expected MSI band count together.

The PaviaC recipe uses the 102-band HRHS splits under `Dataset/PaviaC`, maps the
430--840 nm HSI grid to four QuickBird MSI bands with
`hsmsfusion/degradations/srf/QuickBird_srf.mat`, and uses the same fixed Gaussian
blur/decimation protocol at scale 4. The current PaviaC H5 files contain HRHS
only, so HRMS and LRHS are materialized from this configured observation model.

The ICVL recipe uses the 31-band NCHW HRHS splits under `Dataset/ICVL`, whose
standard wavelength grid is 400--700 nm. It synthesizes three-band Nikon D700
HRMS observations and Gaussian x4 LRHS observations. Because the ICVL test
images are 1024 x 1024, CAUWT uses 256-pixel tiled inference with a 4-pixel
shave margin by default.

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

### MCT-Net note

`MCTNet` is migrated from the official MCT-Net implementation (Wang et al.,
Knowledge-Based Systems, 2023, DOI `10.1016/j.knosys.2023.110362`). It returns
its spatial/spectral/edge auxiliary tensors only when `self.training` is true;
validation, test, prediction, visualization, and HDF5 export receive only the
final HRHS tensor. Its Chikusei recipe decomposes the compound objective into
three independent LossManager terms: final fusion reconstruction plus spatial
and spectral edge regularization. The two edge terms derive their gradients
inside the generic loss implementation and support `params.criterion: l1 | mse`.
Their `weight` values are the only objective coefficients, so every raw value
and weighted contribution is reported in Rich, CSV, and TensorBoard. Its learned
position grids are referenced to the 64x64 Chikusei training patch and
interpolated for the 256x256 test scenes.

### SCSR-Net note

`SCSRNet` is migrated from the official SCSRNet implementation (Zhou et al.,
International Journal of Applied Earth Observation and Geoinformation, 2026,
DOI `10.1016/j.jag.2026.105547`). Its objective upstream is a plain L1 between
the fused cube and the reference, which maps directly onto the recipe's single
`loss.reconstruction: l1` term with no regularization group. Upstream trains
with ADAM at `lr=1e-3`, `weight_decay=1e-4`, and a StepLR that decays by 0.1
every 2000 iterations; the Chikusei recipe instead keeps the shared toolbox
values (`lr=1e-4`, `weight_decay=0`, no scheduler) so that every method in the
benchmark is trained under one protocol. Use Hydra overrides to follow the
paper's schedule instead, e.g.
`training.optimizer.lr=1e-3 training.optimizer.weight_decay=1e-4`.

Upstream targets a three-band RGB MSI; this protocol feeds the full eight-band
WV2 MSI, so the input MSI convolution consumes eight channels. The trunk
predicts a residual over the bicubic-upsampled LRHS, and its block topology
(stage count, per-stage depth, and per-block kernel sizes) is configuration
rather than code. The deploy-time branch folding that upstream applies before
inference is not ported: it is mathematically equivalent to the multi-branch
training-time form that is kept, so the ported network is used unfolded in both
training and test.

### CGSFN note

`CGSFN` is migrated from the official CGSFN implementation
(https://github.com/MSFLabX/CGSFN). Its objective upstream is a plain L1 between
the fused cube and the reference, which maps onto the recipe's single
`loss.reconstruction: l1` term. Upstream trains with ADAM at `lr=0.00015`,
`weight_decay=1e-8`, and a cosine annealing schedule stepped per iteration, and
it applies `xavier_uniform_` to every `Conv2d` and `Linear` before fitting. The
Chikusei recipe keeps the shared toolbox values (`lr=1e-4`, `weight_decay=0`,
no scheduler) and the toolbox's default initialization, so that every method in
the benchmark is trained under one protocol; use Hydra overrides such as
`training.optimizer.lr=1.5e-4 training.optimizer.weight_decay=1e-8` to follow
the paper's schedule instead. Note that the upstream repository carries no
license file, unlike the MIT-licensed SCSRNet, so this port inherits that
ambiguity for any redistribution.

Upstream hardcodes `scale_factor=8` for its CAVE protocol; the port takes
`scale` from `model.params`, so the Chikusei x4 protocol is expressed directly.
Two structural requirements of the architecture are checked in
`CGSFN._check_geometry` and reported as such, rather than surfacing later as a
broadcast or reshape error. The three encoder stages halve the feature map and
the SFA axis attention partitions 8x8 windows, so the HR side must be divisible
by 64; and `NWI` pools rows and columns into `[B, C, 1, W]` and `[B, C, W, 1]`
and adds them, so the feature map must be square. The 64x64 training patches,
the 64x64 inference tiles, and the 256x256 test scenes all satisfy both.

The recipe therefore uses `inference.mode: tiled` rather than `full`: the
cross-modal attention at the first encoder level is quadratic in the
feature-map area, and a whole 256x256 scene peaks at 12.5 GB of VRAM even under
`torch.no_grad()`. The configured `tile_size: 64` with `shave: 16` gives a
32-pixel step, which is exactly the sliding window upstream uses for its own
validation and test (`training_size=64`, `stride=32`, results clipped to
[0, 1] as the toolbox already does); the same scene peaks at 129 MB when tiled.

The port drops dead code without changing forward numerics: the unreachable
`SCAM` module and the `cro` branch that guarded it (and whose attention output
it computed and then discarded), the `Block`, `DualT`, and `ffn` modules that
`Net` never references, and the `MSAA.channel_agg` and `SFA_Improved1.cov_x`
submodules that were constructed but never read. Upstream's `einops` calls are
written out in plain torch, so no new dependency is introduced.

### SSIL-Net note

`SSILNet` is adapted from the official SSIL-Net implementation
(https://github.com/Zhang-bd/SSIL-Net). Its upstream objective is
`L1 + 0.1 * cal_sam`, and `cal_sam` returns the mean spectral angle divided by
pi, so the mapped recipe pairs `loss.reconstruction: l1` at weight 1.0 with
`loss.regularization: sam` at weight `0.1 / pi`. Weighting the toolbox's
radian-valued `sam` term by `1 / pi` is what reproduces upstream's coefficient,
so the recipe records `0.0318310` rather than `0.1`. Upstream trains with ADAM
at `lr=1e-4`, a `MultiStepLR` that halves every 25 epochs from epoch 100, 1000
epochs, and a batch size of 4; the recipe keeps the shared toolbox values
(`lr=1e-4`, `weight_decay=0`, no scheduler, 200 epochs) so that every method in
the benchmark is trained under one protocol. Use Hydra overrides to follow the
paper's schedule instead. Upstream applies no explicit weight initialization of
its own, so the toolbox's default initialization matches it. Note that the
upstream repository carries no license file.

Upstream is a pan-sharpening model: it consumes a panchromatic guide at full
resolution, a multispectral observation at H/4, and the hyperspectral cube at
H/16, and its ladder is two paired factors of 4 for a total ratio of 16. The
toolbox protocol supplies two resolutions, not three, and its scale is 4, so
the ladder's top step collapses and every factor is rewritten to follow the
recipe: `lr_hsi` at H/scale is brought to the MSI resolution by a factor of
`scale` (upstream's `up16` becomes `up scale`), and the MSI-side factors become
1 since the protocol's HRMSI already sits at the output resolution. The
inputs are wired as `hsi = lr_hsi`, `msi = hr_msi`, and
`pan = hr_msi.mean(dim=1)`. Upstream instead averages its 128-band reference
cube for the panchromatic guide, which would hand the network full-resolution
structure of the very cube it has to reconstruct; taking the band-mean of the
measured HRMSI keeps the guide inside the observation model.

This adaptation changes scale factors and input channel counts only. The hidden
width stays 64, and the three `ResSPA` blocks, four `FuseBlock7` blocks, and
three `Multi_Scale` blocks are untouched, so no parameter is added or removed
by the rescaling. Because upstream's single `up4` module is reused at two
different ratios there (4 on the MSI path, 4 on the HSI path) but needs 1 and
`scale` here, it splits into `up_msi_pan` and `up_hsi_msi`; both are
parameterless `nn.Upsample` instances. Note that `SuperResolution(1)` is not a
passthrough — it keeps a full 5x5/3x3/3x3 convolution stack that operates at
the same resolution, so the MSI and residual-decoder branches still compute
features rather than degenerating. `FuseBlock7.scale1` and `.scale2` are kept
even though upstream never reads them, so the parameter count stays comparable.
`Freprocess2`, `SpeFE`, `up2`, and several unused pooling attributes are dropped
as unreachable or parameterless, and upstream's `einops` calls are written out
in plain torch, so no new dependency is introduced.

The recipe uses `inference.mode: full`. The attention in `FuseBlock7` splits
channels across heads and flattens the spatial axes, so its product is a
`[B, heads, c, c]` map rather than a spatial one and the cost is linear in the
pixel count; a whole 256x256 scene peaks at 429 MB of VRAM and fuses in about
56 ms.

### LSO note

`LSO` is migrated from the official LSO implementation
(https://github.com/weili419/LSO, Li et al., ICML 2026). Unlike the three
preceding ports it needs no protocol adaptation: upstream's
`LSOModel.forward(HR_MSI, LR_HSI)` already consumes exactly two observations at
two resolutions and already predicts a residual over the bicubic-upsampled
LRHS, which is the toolbox contract. Its `sf` is a free parameter upstream
rather than a hardcoded constant, so the x4 protocol is expressed directly. The
only structural change is that `Up` hardcodes `in_channels + 3` for the HR-MSI
tensor it concatenates at every rung — the `3` is upstream's RGB band count, so
it becomes `msi_channels`, widening three convolutions to accept the eight-band
WV2 MSI. The remaining edits are surface: `n_bands`, `n_select_bands` and `sf`
become `hsi_channels`, `msi_channels` and `scale`, the argparse namespace
becomes keyword arguments, and `forward` takes `(lr_hsi, hr_msi)` in the
toolbox's order.

Upstream's objective is a plain L1 against the reference, which maps directly
onto the recipe's single `loss.reconstruction: l1` term with an empty
regularization group. Upstream trains with AdamW at `lr=3e-4`,
`weight_decay=1e-4`, and a StepLR that decays by 0.6 every 300 epochs for 1000
epochs; the Chikusei recipe keeps the shared toolbox values (`lr=1e-4`,
`weight_decay=0`, no scheduler, 200 epochs) so that every method is trained
under one protocol. Use Hydra overrides to follow the paper's schedule
instead, e.g. `training.optimizer.name=adamw training.optimizer.lr=3e-4
training.optimizer.weight_decay=1e-4`. Upstream initializes the two learned
latent operator tensors explicitly: the spectral mixing matrix `weights`, shaped
`[width, 2 * num_basis]`, and the latent tokens, shaped
`[head, num_token, width / head]`, are both `(1 / width) * torch.rand(...)`.
That initialization is kept verbatim; the mode list is a non-learned buffer, and
everything else keeps the PyTorch defaults upstream uses.

The EDSR upsampler and tail, the pretrained-weight URL table, and upstream's
tolerant `load_state_dict` override are dropped: the encoder is always built
with `no_upsampling=True`, so no tail exists to build or to mismatch. Also
dropped are the dead `bilinear` argument and the unreachable
`ConvTranspose2d` branch of `Up`, and the `numpy` dependency. The mean-shift
modules are *kept* despite being unreachable — `EDSR.forward` has both call
sites commented out upstream, so they carry frozen weights and no gradient —
because the released checkpoints contain them and upstream's own
`load_state_dict` raises on a checkpoint key the model does not have.

This is the only one of the four ports whose upstream publishes weights, and
the only one whose parameter layout could therefore be checked rather than
asserted. All four released checkpoints (CAVE and Harvard, x4 and x8) strict-load
into the ported network: every one of their 69 tensors matches on key and shape,
and the four `sub_mean`/`add_mean` tensors above are the only entries that would
otherwise differ. Those checkpoints are 3-band-MSI, 31-band-HSI CAVE and Harvard
models, so they still cannot be used for the 8-band-MSI, 128-band-HSI Chikusei
benchmark, which is trained from scratch like every other method here. Upstream
is also MIT licensed, unlike CGSFN and SSIL-Net. The port constructs its
submodules in the same order upstream does, including the mean-shift
convolutions, so a seeded run draws from the global RNG stream in the same
sequence and reproduces upstream's initialization rather than merely its
architecture.

The recipe uses `inference.mode: full`. Each `NeuralSpectralBlock2d` attends
within a `patch_size` window rather than across the whole feature map, so the
cost stays linear in the pixel count and a whole 256x256 scene peaks at 1184 MB
of VRAM, fusing in about 32 ms per scene as measured by `hsmsfusion test`.

### S3O note

`S3O` is migrated from the official Selective Spatial-Spectral Operator
implementation
(https://github.com/GuobaPei/Selective-Spatial-Spectral-Operator, Pei et al.,
Findings of CVPR 2026), which is MIT licensed. Upstream ships three
dataset-specific wrappers that differ only in the pyramid width, the number of
encoder blocks and the head width; the Chikusei wrapper is kept and those
differences become arguments. It follows the same shape as LSO: the LRHS is
bicubically upsampled, concatenated with the HRMSI, pushed through a
multi-scale spatial pyramid and a token MLP, then through the SSM backbone, and
the depthwise-separable head predicts a residual over that upsampled LRHS. The
concatenation order is upstream's (HSI first, MSI second). Upstream's Chikusei
MSI is a three-band RGB cube from its own HDF5 files while this protocol's is
the eight-band WV2 observation, so the pyramid's first convolution widens from
34 to 136 input channels; nothing else about the structure changes.

The one real change is that upstream bakes the spatial size into the network:
`Block` takes `H` and `W` at construction and uses them to reshape the token
sequence, both in the row/column spatial scan and in the top-k token selection.
A model trained on 64x64 patches therefore cannot be applied to a larger scene,
and upstream works around that by rebuilding the model at the evaluation
resolution and loading the same weights into it, in both `evaluate()` and
`test_chikusei.py`. That workaround is only sound because no parameter depends
on the spatial size, so this port derives `H` and `W` from the input instead and
threads an explicit `hw` tuple through `S3OBackbone` and `S3OEncoderBlock` in
place of upstream's `int(N ** 0.5)`. For square inputs that is the same two
numbers, so the forward pass is unchanged; for non-square ones it is the
difference between working and silently mis-reshaping. It also lets a single
instance serve both the 64x64 training patches and the 256x256 test scenes,
which is what the toolbox's model lifecycle requires.

Upstream's selective enhancement is a curriculum, not a constant: the fraction
of tokens routed through the extra `enhance` head is held at zero for the first
10% of epochs, ramped linearly to 0.10 over the next 20%, and held there. Its
training script applies it with `--sel_warmup 0.10 --sel_ramp 0.20 --sel_max
0.10`; here it lives in `model.progressive_enhancement` and is applied from
`on_train_epoch_start`, so it travels with the recipe. `sel_ratio` stays at
0.10 outside training, which is the value upstream's freshly built test-time
model carries. Upstream's objective is a plain L1 against the reference, which
maps onto the recipe's single `loss.reconstruction: l1` term with an empty
regularization group. Its optimizer is AdamW split into five parameter groups
(the SSM parameters at `1e-3`, the enhancement and projection weights at `5e-4`,
everything else at `1e-4`, and biases and norms with no weight decay) with a
MultiStepLR that decays by 0.9 at epochs 50, 100 and 150; the Chikusei recipe
keeps the shared toolbox values (`adam`, `lr=1e-4`, no scheduler, 200 epochs) so
that every method is trained under one protocol. Upstream also evaluates
Chikusei at x8 with `batch_size 32`; this recipe uses the x4 protocol and batch
16 shared by the other Chikusei methods. Use Hydra overrides to follow the
paper's schedule instead, e.g. `training.optimizer.lr=6e-4
training.optimizer.weight_decay=1e-4`.

Dropped as dead or unusable: a depthwise-separable downsample the backbone never
references, the coordinate grid every wrapper builds and passes to a head that
ignores it (upstream marks it "unused, kept for compatibility", and `spec_dim`
carries the same note), the two other dataset wrappers, and two locals named
against the tensors concatenated beneath them. `timm` is also dropped: upstream
imports `DropPath` and `trunc_normal_` from `timm.layers`, an API that does not
exist in the timm version available here, so `DropPath` is inlined and
`trunc_normal_` comes from `torch.nn.init`, as in `CLSNet` and `DPFormer`.
Upstream runs `self.apply(self._init_weights)` at both the backbone and the
outer model, so its linear layers are initialized twice; that is kept, since it
costs nothing and leaves the RNG stream in the same shape. Like `DHMFNet` and
`SDPNet`, this model needs CUDA, because mamba_ssm's `selective_scan_fn` is a
CUDA kernel and the `selective_scan_ref` alternative it ships is a separate
numerical path that is deliberately not wired in.

The recipe uses `inference.mode: full`. The scans are over sequences of the
pixel count, so a whole 256x256 scene peaks at 452 MB of VRAM and fuses at
roughly 17 ms per scene at the model level, or 29 ms per scene as measured by
`hsmsfusion test`; a training batch of 16 at 64x64 peaks at 1555 MB. The
Chikusei recipe has 2.2 M parameters, and one epoch takes about 4 s.

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
