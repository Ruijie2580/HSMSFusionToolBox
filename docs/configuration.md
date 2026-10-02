# 架构设计：配置体系与运行记录

入口指南见根目录 `AGENTS.md`；核心契约见 `docs/architecture.md`。

## 配置

- 根 `configs/config.yaml`：defaults（`experiment`、`runtime`）、`paths`、
  `seed`、仅运行时键（`resume`、`run_dir`、`checkpoint_name`、
  `prediction_path`）、`metrics`（leaderboard + epsilon）、`benchmark`。
- 配方 `configs/experiment/<dataset>/<model>_x<ratio>.yaml` 必须以
  `# @package _global_` 开头，并声明 `defaults: [/dataset: ...,
  /degradation: ..., _self_]`。它拥有 `experiment.name`、`model.*`、
  `training.{trainer,optimizer,scheduler,loss,dataloader,checkpoint}`；非
  Chikusei 数据集的配方还必须覆盖 `benchmark.name`——默认值是 Chikusei 专用
  的，不覆盖会静默出错。
- `experiment.name` 必须是 `runs_dir` 的单层直接子目录名（防路径穿越）；
  它决定运行目录。
- 跨组一致性**不做**自动校验：`dataset.hsi_channels`、
  `degradation.spectral.num_hsi_bands/num_msi_bands` 与
  `model.params.{hsi_channels,msi_channels,scale}` 必须手工保持一致；不一致
  会以运行时形状错误的形式暴露。
- 损失配置格式：`training.loss.reconstruction` / `regularization` 两个列表，
  每项 `{name, alias, weight, params?}`。损失项的 `category` 必须与所在组
  匹配。旧的扁平 `kind`/`components` 格式会被直接报错拒绝。
- `runtime.precision`：`fp32 | fp16-mixed | bf16-mixed | auto-mixed`
  （bf16 会检查 GPU 是否支持）。

## 运行记录（危险区）

- **非 resume 的 `train` 会先通过 `shutil.rmtree` 删除整个已存在的
  `runs/<experiment.name>/`**（checkpoints、日志、manifest 全部销毁）再重建。
  启动实验名已存在的训练命令前，务必提醒用户。
- `resume=<run_dir>` 是严格续训：加载归档的 `resolved_config.yaml`，忽略当前
  override，并要求存在 `checkpoints/last.ckpt`。它不是微调入口。
- `test`/`evaluate` **不信任** 当前配置：它们从运行目录归档的
  `resolved_config.yaml` 重建模型与数据（当前配置只打两个补丁：追加 `uiqi`
  指标与兼容旧版 `model.inference`）。想改超参必须重新训练。
- `checkpoints/best.ckpt` 只在执行过验证后更新（监控指标来自
  `training.checkpoint`）；`checkpoints/last.ckpt` 在每个 epoch 边界强制保存，
  且额外内嵌 `hsmsfusion_rng_state`——绝不要用 `weights_only=True` 加载它。
- `check_val_every_n_epoch` 必须是 `<= max_epochs` 的正整数，否则永远不会产出
  `best.ckpt`。
- `config_hash`（运行身份）排除 `hydra/resume/run_dir/checkpoint_name/
  prediction_path`；新增仅运行时的 CLI 字段必须加入该排除表。
