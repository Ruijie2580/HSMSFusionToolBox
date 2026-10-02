"""In-memory XiongAn DataModule with fixed observation materialization."""

from hsmsfusion.core.registry import DATASET_REGISTRY
from hsmsfusion.data.base import BaseFusionDataModule


@DATASET_REGISTRY.register("xiongan")
class XiongAnDataModule(BaseFusionDataModule):
    """Load XiongAn HRHS once and materialize fixed HRMS/LRHS observations."""

    dataset_label = "XiongAn"
    default_hsi_channels = 128
