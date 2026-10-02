# HSMSFusionToolBox Architecture

HSMSFusionToolBox 是一个配置驱动的高光谱–多光谱（HS–MS）图像融合实验
工具箱（PyTorch + Lightning + Hydra），覆盖 8 个数据集、约 20 个移植的
论文模型，以及一套统一的基准协议（固定 SRF + 高斯模糊/抽取观测模型）。

面向用户的协议与使用细节见 [README.md](README.md)。架构设计文档拆分在
`docs/` 下，改动前先读对应篇章：

- [docs/architecture.md](docs/architecture.md) — 硬契约（张量/batch/HDF5/
  观测模型/指标）、基准协议规则与注册表发现机制
- [docs/models.md](docs/models.md) — 模型包结构、Lightning 基类契约、
  推理配置、上游移植惯例
- [docs/configuration.md](docs/configuration.md) — Hydra 配置体系与运行记录
- [docs/data-and-testing.md](docs/data-and-testing.md) — 数据管线细节与
  测试约定

在本仓库工作的 AI 编码代理另见 [AGENTS.md](AGENTS.md)。
