"""Regularization loss terms for HS--MS fusion."""

from hsmsfusion.losses.regularization.terms import (
    SAMRegularizationLoss,
    SSIMRegularizationLoss,
    SpatialEdgeRegularizationLoss,
    SpectralEdgeRegularizationLoss,
    TotalVariationRegularizationLoss,
)

__all__ = [
    "SAMRegularizationLoss",
    "SSIMRegularizationLoss",
    "SpatialEdgeRegularizationLoss",
    "SpectralEdgeRegularizationLoss",
    "TotalVariationRegularizationLoss",
]
