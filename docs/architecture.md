# 架构设计：核心契约与注册机制

本文档描述 HSMSFusionToolBox 的架构性不变量与扩展机制。入口指南见根目录
`AGENTS.md`。

## 硬契约（不可破坏）

- **张量布局**：一律 NCHW float；通道维 = 光谱波段。
- **Batch 键**：`lr_hsi`、`hr_msi`（观测）与 `gt_hsi`（目标，仅在暴露时存在）。
  模型侧统一通过 `core/protocols.py` 的 `observation_view(batch)` /
  `require_target(batch)` 读取。
- **HDF5 键**：划分文件 `<root>/{train,valid,test}.h5` 只含一个数据集 `HRHS`
  （NCHW float32，已归一化到 [0,1]，无 NaN/Inf——硬性校验）。预测文件只含一个
  数据集 `/prediction`，float32 NCHW，已裁剪到 [0,1]，原子写入。
- **预测后处理**：所有模型输出必须经过 `clip_prediction`（裁剪到 [0,1]；
  NaN/Inf 在裁剪*之前*就抛 `ContractError`）。网络 forward 绝不允许产生
  非有限值。
- **观测模型**：`ObservationGenerator(hr_hsi) -> {"hr_msi", "lr_hsi"}`；
  `lr_hsi` 的空间尺寸必须恰为 `(H//scale, W//scale)`。空间抽取 offset 锁定为
  `[0,0]`（基准契约，不可配置）。退化是确定性的；观测按划分一次性物化在 CPU
  内存中（`generate_in_batches`），无随机裁剪、无数据增强。
- **指标**：`Evaluator` 刻意采用 CPU + float32（保证数字可比的参考实现）。
  使用总体标准差（`ddof=0`）。`metrics.csv` 表头为
  `Image, PSNR, SSIM, *SAM, *ERGAS, RMSE, UIQI`（`*` = 越低越好）；行包括
  `image_0000...`、`Average`、`Std`。
- **模型 forward**：`(lr_hsi, hr_msi) -> hr_hsi_estimate`，NCHW 进出。多数
  移植模型预测的是 bicubic 上采样 LRHS 之上的残差；测试锁定了这一点
  （将输出层清零后输出必须塌缩为 bicubic 基线）。

## 基准协议

- 官方运行**不做**框架侧的几何增强：随数据集发布的训练 patch 已离线增强，
  框架内的观测物化保持确定性，任何在线增强都会偏离已发布的基准分布。
- `benchmark.operator_access` 的 `known` / `hidden` 是两个不同的协议档位：
  前者把完整退化配置交给模型，后者扣留。两档**必须**使用不同的 benchmark
  名称与独立的结果表，不得混入同一排名。
- 具体的 SRF/PSF 数值是配置字段（`configs/degradation/*.yaml`），永远不进
  模型代码。

## 注册表与发现机制

`core/registry.py` 中有六个全局注册表：`MODEL_REGISTRY`、`DATASET_REGISTRY`、
`SPECTRAL_OPERATOR_REGISTRY`、`SPATIAL_OPERATOR_REGISTRY`、`METRIC_REGISTRY`、
`LOSS_REGISTRY`（注意：`LOSS_REGISTRY` 未从包级 `__init__` re-export）。

- 注册用装饰器：`@X_REGISTRY.register("name")`。名字大小写敏感、全局唯一；
  重名抛 `ConfigurationError`。
- `SPECTRAL_OPERATOR_REGISTRY` 与 `SPATIAL_OPERATOR_REGISTRY` 各取一项，由
  `ObservationGenerator` 组合成完整观测模型。
- `METRIC_REGISTRY` 锁定基准协议的默认指标集，同时允许配方追加诊断指标
  （如 `test`/`evaluate` 自动补 `uiqi`）。
- 注册靠 **import 副作用** 完成：`discover_models()` / `discover_datasets()`
  会 import 每个子包；损失/指标/算子模块由各自包的 `__init__`（或显式 import
  列表，如 `degradations/observation.py`）导入。新增模块后若忘记导入，注册会
  静默缺失，配置解析时报 "Unknown ..." 并列出可用名。
- 缺失的可选依赖必须通过 `mark_unavailable` 上报（而不是跳过注册），以便
  `hsmsfusion list-models` 能显示原因。
