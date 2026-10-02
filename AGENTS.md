# AGENTS.md — HSMSFusionToolBox

面向在本仓库工作的 AI 编码代理的指南。

## 项目简介

配置驱动的高光谱-多光谱（HS–MS）图像融合实验工具箱。技术栈：PyTorch +
PyTorch Lightning + Hydra + h5py。包含 8 个数据集、约 20 个移植的论文模型、
一套统一的基准协议（固定 SRF + 高斯模糊/抽取观测模型）。面向用户的协议细节
见 `README.md`；根目录 `ARCHITECTURE.md` 已瘦身为指向 README 与 docs/ 的
指路牌。

## 深入文档（docs/）

架构设计细节已拆分到 `docs/` 目录，开始改动前先读对应篇章：

- [docs/architecture.md](docs/architecture.md) — 硬契约（张量/batch/HDF5/
  观测模型/指标）与注册表发现机制
- [docs/models.md](docs/models.md) — 模型包结构、Lightning 基类契约、
  推理配置、上游移植惯例
- [docs/configuration.md](docs/configuration.md) — Hydra 配置体系与运行记录
  （含会删除旧训练记录的危险行为）
- [docs/data-and-testing.md](docs/data-and-testing.md) — 数据管线细节与
  测试约定

## 环境与命令

```powershell
conda activate vmamba                  # 项目 Python 环境（torch、lightning、hydra 等）
pip install --no-build-isolation -e .  # 可编辑安装；控制台命令：hsmsfusion
pytest                                 # 全量测试（testpaths=tests，addopts=-ra）
ruff check .                           # lint（见 pyproject [tool.ruff]）
```

PATH 上的默认 `python`（miniconda base）缺少 `lightning`/`hydra`，连测试收集
都会失败——务必在项目环境内运行。

CLI（`hsmsfusion <命令> <hydra overrides>`；命令有 `train`、`test`、
`evaluate`、`inspect-data`、`list-models`、`print-config`）：

```powershell
hsmsfusion train experiment=chikusei/clsnet_x4 training.optimizer.lr=5e-5
hsmsfusion train resume=runs/clsnet_chikusei_x4     # 严格续训，不接受 override
hsmsfusion test experiment=chikusei/clsnet_x4 checkpoint_name=best
hsmsfusion evaluate experiment=chikusei/clsnet_x4 prediction_path=D:\results\prediction.h5
```

子命令之后的所有参数都是 Hydra override。Hydra 以 `chdir: false` 运行：所有
相对路径（`Dataset/...`、`runs`、`output`）都相对于启动命令时所在的仓库根目录。

## 仓库结构

- `hsmsfusion/cli.py` — 唯一入口；train/test/evaluate 生命周期编排。
- `hsmsfusion/core/` — `registry.py`（通用 `Registry[T]` + 6 个全局注册表）、
  `protocols.py`（张量/batch 契约）、`exceptions.py`。
- `hsmsfusion/models/<Name>/` — 每个论文模型一个包（详见
  [docs/models.md](docs/models.md)）。
- `hsmsfusion/data/base.py` — `BaseFusionDataModule`（共享的 HRHS 加载校验、
  观测物化、dataloader 构建）+ `InMemoryFusionDataset`；
  `hsmsfusion/data/datasets/<name>/datamodule.py` — 每个数据集一个薄子类，
  只声明类属性（`dataset_label`、`default_hsi_channels`）。
- `hsmsfusion/degradations/` — `observation.py`（`ObservationGenerator`、
  `build_observation_generator`），加 `spectral/`（`gaussian_srf`、
  `tabulated_srf`）与 `spatial/`（`gaussian_decimation`）；实测 SRF 表在
  `srf/*.mat`。
- `hsmsfusion/losses/` — `LossManager` + `ConfigurableLossTerm`；
  `reconstruction/`（`l1`、`l2`、`char`、`cycle_consistency`）与
  `regularization/`（`sam`、`ssim`、`tv`、`spatial_edge`、`spectral_edge`）。
- `hsmsfusion/metrics/` — 函数式指标注册表（`psnr_global`、`rmse_global`、
  `ssim_band_mean`、`sam_rad`、`sam_deg`、`ergas`、`uiqi`）+ CPU float32
  的 `Evaluator`。
- `hsmsfusion/evaluation/export.py` — `write_prediction_file`（predictions.h5）。
- `hsmsfusion/callbacks.py`、`hsmsfusion/visualization.py` — Rich 进度/报告，
  SRF 与验证图像输出到 `<run>/logs/figures/`。
- `hsmsfusion/utils/experiment.py` — 运行目录、manifest、配置哈希工具。
- `configs/` — Hydra 配置组：`dataset/`、`degradation/`、`runtime/`、
  `experiment/<dataset>/<model>_x<ratio>.yaml` 配方；根 `config.yaml`。
- `scripts/` — 独立分析/GUI 工具（`python scripts/xxx.py`，各自自举
  `sys.path`）；公共助手在 `scripts/_shared.py`。
- `tests/` — 扁平 pytest 套件，每个子系统一个文件。
- `runs/<experiment>/` — 训练记录（checkpoints、tensorboard 日志、
  `resolved_config.yaml`、`manifest.yaml`）；`output/<dataset>_x<scale>/<model>/`
  — 测试预测 + `metrics.csv`。

## 常见任务

- **新增模型**：创建 `hsmsfusion/models/<Name>/`（`__init__.py`、
  `network.py`、以 `@MODEL_REGISTRY.register("<Name>")` 注册的
  `lightning_module.py`），添加
  `configs/experiment/<dataset>/<name>_x4.yaml`（`model.name` 必须与注册名
  完全一致），添加 `tests/test_<name>.py`。用 `hsmsfusion list-models` 与
  `hsmsfusion print-config experiment=...` 验证。
- **新增数据集**：`hsmsfusion/data/datasets/<name>/datamodule.py`（继承
  `hsmsfusion.data.base.BaseFusionDataModule`，设置 `dataset_label` 与
  `default_hsi_channels`）、`configs/dataset/<name>.yaml`（`name` = 注册名）、
  `configs/degradation/<name>_<sensor>_x<scale>.yaml`
  （`<dataset>_<sensor>_x<scale>` 文件名约定同时被
  `scripts/plot_spectral_curve.py` 解析），然后是配方。
- **新增退化算子**：在 `degradations/spectral|spatial/` 注册一个构造签名为
  `__init__(self, config: Mapping)` 的 `nn.Module`，把 import 加到
  `degradations/observation.py`（及子包 `__all__`）；空间算子必须暴露 `scale`
  属性且输出尺寸恰为 `H//scale`。
- **新增损失**：继承 `ConfigurableLossTerm`，设置 `category`
  （`"reconstruction"`/`"regularization"`），注册，并确保子包 `__init__`
  导入了它。
- **新增 CLI 命令**：在 `cli.py` 中扩展 `COMMANDS`、`_dispatch`，并添加
  `_run_*` 处理函数。

## 风格与工作流备注

- Windows 优先的仓库：Git Bash shell；脚本必须在 `num_workers=0` 与非 UTF-8
  终端下工作（callbacks 会把 Rich 面板/进度条降级为 ASCII）。
- Matplotlib 一律使用 Agg 后端（在 `visualization.py` 与脚本的 import 时设置）；
  图件使用 Times New Roman 字体。
- `configs/config.yaml` 固定 `hydra.job.chdir: false`——不要依赖 Hydra 的
  每次运行输出目录；使用 `paths.runs_dir` / `paths.output_dir`。
- `scripts/` 绘图工具读取同一套 `output/` + `Dataset/` 契约；修改导出格式
  （metrics.csv 表头、predictions.h5 键）时，须在同一改动中同步
  `scripts/_shared.py` 与受影响的绘图脚本。
- 生成产物 `runs/`、`output/`、`*.ckpt`、`Dataset/` 已被 gitignore；
  `figures/` 除 `figures/metrics_smoke/` 外会提交。
- 有意仅保留在本地（gitignore，公开 clone 中不存在）：除
  `chikusei/cyberfuse_x4` 之外的所有实验配方（只有它是忠实复现，其余均为
  冒烟测试超参）、`KNLNet`、`SatFusion`、`SpecSolver`、`SSILNet` 四个模型包，
  以及 `tests/test_satfusion.py` / `tests/test_ssilnet.py`。这正是测试用
  `require_recipe` fixture 把关的原因——不要去"修复"由此产生的 skip。
