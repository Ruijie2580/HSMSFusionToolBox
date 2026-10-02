"""In-memory Pavia Center DataModule with fixed observation materialization."""

from hsmsfusion.core.registry import DATASET_REGISTRY
from hsmsfusion.data.base import BaseFusionDataModule


@DATASET_REGISTRY.register("paviac")
class PaviaCDataModule(BaseFusionDataModule):
    """Load PaviaC HRHS once and materialize fixed HRMS/LRHS observations."""

    dataset_label = "PaviaC"
    default_hsi_channels = 102
