"""Regularization loss terms for HS--MS fusion."""

from hsmsfusion.losses.regularization.terms import (
    SAMRegularizationLoss,
    SpatialEdgeRegularizationLoss,
    SpectralEdgeRegularizationLoss,
    SSIMRegularizationLoss,
    TotalVariationRegularizationLoss,
)

__all__ = [
    "SAMRegularizationLoss",
    "SSIMRegularizationLoss",
    "SpatialEdgeRegularizationLoss",
    "SpectralEdgeRegularizationLoss",
    "TotalVariationRegularizationLoss",
]
