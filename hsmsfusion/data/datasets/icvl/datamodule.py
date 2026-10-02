"""In-memory ICVL DataModule with fixed observation materialization."""

from hsmsfusion.core.registry import DATASET_REGISTRY
from hsmsfusion.data.base import BaseFusionDataModule


@DATASET_REGISTRY.register("icvl")
class ICVLDataModule(BaseFusionDataModule):
    """Load ICVL HRHS once and materialize fixed HRMS/LRHS observations."""

    dataset_label = "ICVL"
    default_hsi_channels = 31
