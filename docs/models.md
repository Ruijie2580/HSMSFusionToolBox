# 架构设计：模型契约与移植惯例

入口指南见根目录 `AGENTS.md`；核心契约见 `docs/architecture.md`。

## 模型包结构

`hsmsfusion/models/<Name>/` 是完整的实现单元：`__init__.py`（re-export
Lightning 模块）、`network.py`（纯 `nn.Module`）、`lightning_module.py`
（被注册的类）。**不要** 改 `hsmsfusion/models/__init__.py` 来做注册——
发现机制会自动 import 每个子包。

## Lightning 基类契约

所有模型继承 `BaseFusionLightningModule`（`models/base.py`），构造函数为
固定的 keyword-only 签名：

```python
(model_config, optimizer_config, scheduler_config, loss_config,
 metric_config, scale, operator_config=None)
```

子类必须设置类属性 `requires_gt: bool` 与
`supported_operator_access: set[str]`（目前全部为 `{"known", "hidden"}`），
并在 `__init__` 中构建 `self.network = XxxNet(**self.model_config["params"])`。
仅当 `benchmark.operator_access == "known"` 时 `operator_config` 携带完整退化
配置，否则为 `None`。

基类已实现 `training_step`/`validation_step`/`on_validation_epoch_end`/
`predict_step`（含分块推理）/`configure_optimizers`（adam/adamw；
none/cosine/multistep 调度器）。**仅当** 目标函数无法表达为已注册损失项的
组合时才覆盖 `compute_objective`。`validation_step` 的输出必须是包含
`prediction` 的 mapping（可视化回调依赖它）。

## 推理配置

`model.inference` 配置：`{mode: full|tiled, tile_size, shave}`；`tiled` 但未给
`tile_size` 时会静默回退为 `full`。分块推理假设 `H、W` 可被 `scale` 整除。
CGSFN 必须使用 tiled 模式（注意力复杂度为面积的二次方）。

## 上游移植惯例

移植上游代码时的项目惯例：把硬编码值（scale、训练计划、波段数）提升为构造
参数/配置字段，删除死代码，并把每一处与上游的偏差记录到 `network.py` 的模块
docstring（范本是 `CGSFN/network.py`）。移植测试会锁定上游默认超参与
state_dict 兼容性。

`KNLNet` 是用于检验框架契约的最小参考模型——新增模型从它复制。
