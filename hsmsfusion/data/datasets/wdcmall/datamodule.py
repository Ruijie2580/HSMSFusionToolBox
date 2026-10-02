"""In-memory Washington DC Mall DataModule with fixed observation materialization."""

from hsmsfusion.core.registry import DATASET_REGISTRY
from hsmsfusion.data.base import BaseFusionDataModule


@DATASET_REGISTRY.register("wdcmall")
class WDCMallDataModule(BaseFusionDataModule):
    """Load Washington DC Mall HRHS once and materialize fixed HRMS/LRHS observations."""

    dataset_label = "WDCMall"
    default_hsi_channels = 191
