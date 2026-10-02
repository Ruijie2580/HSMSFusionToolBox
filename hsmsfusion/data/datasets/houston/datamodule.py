"""In-memory Houston 2013 DataModule with fixed observation materialization."""

from hsmsfusion.core.registry import DATASET_REGISTRY
from hsmsfusion.data.base import BaseFusionDataModule


@DATASET_REGISTRY.register("houston")
class HoustonDataModule(BaseFusionDataModule):
    """Load Houston 2013 HRHS once and materialize fixed HRMS/LRHS observations."""

    dataset_label = "Houston"
    default_hsi_channels = 144
