"""Registry-backed, configuration-driven loss composition."""

# Import packages for registration side effects before LossManager resolves them.
from hsmsfusion.losses import reconstruction as reconstruction
from hsmsfusion.losses import regularization as regularization
from hsmsfusion.losses.manager import LossManager
from hsmsfusion.losses.term import ConfigurableLossTerm

__all__ = ["ConfigurableLossTerm", "LossManager", "reconstruction", "regularization"]
