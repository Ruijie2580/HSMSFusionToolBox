# 架构设计：数据管线与测试约定

入口指南见根目录 `AGENTS.md`；核心契约见 `docs/architecture.md`。

## 数据管线细节

- DataModule 构造签名：`(dataset_config, degradation_config,
  dataloader_config)`；`cache_device` 必须为 `cpu`。
- `num_workers` 被硬校验为 `0`（Windows；数据本就在内存中）。
- `configure_model_visibility(requires_gt=...)` 控制训练 loader 是否暴露
  `gt_hsi`；验证/测试恒暴露。
- 8 个 datamodule 共享 `hsmsfusion/data/base.py` 的 `BaseFusionDataModule`
  （HRHS 加载校验、观测物化、dataloader 构建全在基类）；各子类只声明
  `dataset_label` 与 `default_hsi_channels` 两个类属性。Harvard 的 HDF5 为
  float64 存储，由共享加载器（`np.asarray(..., dtype=float32)`）在加载时转
  float32；所有数据集文件均为 NCHW 布局，非 NCHW 输入会在 `validate_nchw`
  直接报错。
- 数据集 YAML 中的 `already_augmented: true` 是协议元数据：它声明随数据集
  发布的训练 patch 已完成离线几何增强，因此框架有意**不做**任何在线增强
  （观测物化确定性，见 `docs/architecture.md` 的基准协议一节）。代码不消费
  该键；曾经的 `transforms: []` 占位键已随这一决策定案而移除。

## 测试约定

- 在项目环境中从仓库根运行 `pytest`。测试为扁平的 `tests/test_*.py`，按
  子系统一文件组织。
- `tests/conftest.py` 提供唯一 fixture `require_recipe(relative)`：依赖未发布
  实验配方的测试必须带上它（在公开 clone 上 skip 而非 fail）。
- 测试依赖私有 CLI helper（`_build_model`、`_build_datamodule`、
  `_training_config`、`_validation_interval`、`_load_run_configuration`、
  `_last_checkpoint_callback`）——重构时把它们当作稳定接口。
- 模型移植测试锁定：上游默认宽度/深度、几何校验（形状非法抛 `ValueError`）、
  bicubic 残差不变量、state_dict 往返。S3O 测试需要 CUDA + `mamba_ssm`
  （否则 skip）；其余可在 CPU 运行，但包含真实的 Lightning fit（较慢）。
- 配方测试用 Hydra `initialize_config_dir` + `compose(overrides=["experiment=..."])`
  组合配置；各数据集波长网格在其中有断言（如 Chikusei 363–1018 nm、
  PaviaC 430–840 nm、CAVE/ICVL 400–700 nm）。
