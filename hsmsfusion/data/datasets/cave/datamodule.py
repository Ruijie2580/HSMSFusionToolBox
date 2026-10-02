"""In-memory CAVE DataModule with fixed observation materialization."""

from hsmsfusion.core.registry import DATASET_REGISTRY
from hsmsfusion.data.base import BaseFusionDataModule


@DATASET_REGISTRY.register("cave")
class CAVEDataModule(BaseFusionDataModule):
    """Load CAVE HRHS once and materialize fixed HRMS/LRHS observations."""

    dataset_label = "CAVE"
    default_hsi_channels = 31
