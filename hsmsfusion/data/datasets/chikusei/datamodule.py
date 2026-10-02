"""In-memory Chikusei DataModule with fixed observation materialization."""

from hsmsfusion.core.registry import DATASET_REGISTRY
from hsmsfusion.data.base import BaseFusionDataModule


@DATASET_REGISTRY.register("chikusei")
class ChikuseiDataModule(BaseFusionDataModule):
    """Load Chikusei HRHS once and materialize fixed HRMS/LRHS observations."""

    dataset_label = "Chikusei"
    default_hsi_channels = 128
