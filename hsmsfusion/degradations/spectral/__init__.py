"""Spectral observation operators."""

from hsmsfusion.degradations.spectral.gaussian_srf import GaussianSRF
from hsmsfusion.degradations.spectral.tabulated_srf import TabulatedSRF

__all__ = ["GaussianSRF", "TabulatedSRF"]
