"""In-memory Harvard DataModule with fixed observation materialization."""

from hsmsfusion.core.registry import DATASET_REGISTRY
from hsmsfusion.data.base import BaseFusionDataModule


@DATASET_REGISTRY.register("harvard")
class HarvardDataModule(BaseFusionDataModule):
    """Load Harvard HRHS once and materialize fixed HRMS/LRHS observations.

    Harvard HDF5 stores HRHS in NCHW layout with float64 dtype; the shared
    loader casts to float32 on load.
    """

    dataset_label = "Harvard"
    default_hsi_channels = 31
