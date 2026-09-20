"""MCT-Net backbone adapted to the toolbox NCHW fusion contract.

Source repository: https://github.com/wxy11-27/MCT-Net
Paper: MCT-Net: Multi-hierarchical cross transformer for hyperspectral and
multispectral image fusion, Knowledge-Based Systems 264 (2023), 110362.
DOI: https://doi.org/10.1016/j.knosys.2023.110362

The original implementation is tied to five datasets, five MSI bands, and a
fixed 128x128 training crop.  This port preserves its two-level MCIM/FARM
topology while making channel counts and the learned-position reference size
explicit configuration parameters.
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn

from hsmsfusion.core.exceptions import ConfigurationError, ContractError


class ChannelAttention(nn.Module):
    """Channel attention used by the original MCT-Net encoder stages."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden_channels = max(1, channels // 16)
        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.maximum_pool = nn.AdaptiveMaxPool2d(1)
        self.reduce = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.expand = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        average = self.expand(self.relu(self.reduce(self.average_pool(value))))
        maximum = self.expand(self.relu(self.reduce(self.maximum_pool(value))))
        return self.sigmoid(average + maximum)


class SpatialAttention(nn.Module):
    """Spatial attention from average/max channel summaries."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size not in {3, 7}:
            raise ConfigurationError("MCTNet spatial_attention_kernel must be either 3 or 7.")
        self.convolution = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        average = torch.mean(value, dim=1, keepdim=True)
        maximum = torch.amax(value, dim=1, keepdim=True)
        return self.sigmoid(self.convolution(torch.cat((average, maximum), dim=1)))


class StripDecoderBlock(nn.Module):
    """Four-direction strip-convolution decoder from the original FARM."""

    def __init__(self, input_channels: int) -> None:
        super().__init__()
        if input_channels % 8 != 0:
            raise ConfigurationError("MCTNet strip decoder input channels must be divisible by 8.")
        reduced_channels = input_channels // 4
        strip_channels = input_channels // 8
        self.reduce = nn.Conv2d(input_channels, reduced_channels, kernel_size=1)
        self.normalization = nn.BatchNorm2d(reduced_channels)
        self.relu = nn.ReLU(inplace=True)
        self.horizontal = nn.Conv2d(strip_channels * 2, strip_channels, kernel_size=(1, 9), padding=(0, 4))
        self.vertical = nn.Conv2d(strip_channels * 2, strip_channels, kernel_size=(9, 1), padding=(4, 0))
        self.diagonal_a = nn.Conv2d(strip_channels * 2, strip_channels, kernel_size=(9, 1), padding=(4, 0))
        self.diagonal_b = nn.Conv2d(strip_channels * 2, strip_channels, kernel_size=(1, 9), padding=(0, 4))
        self._initialize_weights()

    @staticmethod
    def _horizontal_transform(value: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = value.shape
        padded = functional.pad(value, (0, width))
        flattened = padded.reshape(batch, channels, -1)[..., :-width]
        return flattened.reshape(batch, channels, height, 2 * width - 1)

    @staticmethod
    def _inverse_horizontal_transform(value: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = value.shape
        flattened = value.reshape(batch, channels, -1).contiguous()
        padded = functional.pad(flattened, (0, height))
        restored = padded.reshape(batch, channels, height, 2 * height)
        return restored[..., :height]

    @classmethod
    def _vertical_transform(cls, value: torch.Tensor) -> torch.Tensor:
        return cls._horizontal_transform(value.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)

    @classmethod
    def _inverse_vertical_transform(cls, value: torch.Tensor) -> torch.Tensor:
        return cls._inverse_horizontal_transform(value.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.relu(self.normalization(self.reduce(value)))
        horizontal = self.horizontal(value)
        vertical = self.vertical(value)
        diagonal_a = self._inverse_horizontal_transform(self.diagonal_a(self._horizontal_transform(value)))
        diagonal_b = self._inverse_vertical_transform(self.diagonal_b(self._vertical_transform(value)))
        return torch.cat((horizontal, vertical, diagonal_a, diagonal_b), dim=1)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)


class CrossAttention(nn.Module):
    """Original MCT attention: self-derived query/key with cross-modality values."""

    def __init__(self, dimension: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.head_dimension = dimension // heads
        self.scale = self.head_dimension**-0.5
        self.to_qkv = nn.Linear(dimension, dimension * 3, bias=False)
        self.attention_dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(dimension, dimension)
        self.projection_dropout = nn.Dropout(dropout)

    def _qkv(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, tokens, dimension = value.shape
        qkv = self.to_qkv(value).reshape(
            batch,
            tokens,
            3,
            self.heads,
            dimension // self.heads,
        )
        query, key, values = qkv.permute(2, 0, 3, 1, 4)
        return query, key, values

    def _attend(self, query: torch.Tensor, key: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        attention = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attention = self.attention_dropout(attention.softmax(dim=-1))
        return torch.matmul(attention, values).transpose(1, 2).flatten(2)

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        first_query, first_key, first_values = self._qkv(first)
        second_query, second_key, second_values = self._qkv(second)
        first_output = self.projection_dropout(
            self.projection(self._attend(first_query, first_key, second_values))
        )
        second_output = self.projection_dropout(
            self.projection(self._attend(second_query, second_key, first_values))
        )
        return first_output, second_output


class CrossTransformerBlock(nn.Module):
    """Pre-normalized cross-attention and independent feed-forward residuals."""

    def __init__(self, dimension: int, heads: int, mlp_dimension: int, dropout: float, attention_dropout: float) -> None:
        super().__init__()
        self.attention_normalization_first = nn.LayerNorm(dimension)
        self.attention_normalization_second = nn.LayerNorm(dimension)
        self.attention = CrossAttention(dimension, heads=heads, dropout=attention_dropout)
        self.feed_forward_normalization_first = nn.LayerNorm(dimension)
        self.feed_forward_normalization_second = nn.LayerNorm(dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(dimension, mlp_dimension),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_dimension, dimension),
            nn.Dropout(dropout),
        )

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        first_update, second_update = self.attention(
            self.attention_normalization_first(first),
            self.attention_normalization_second(second),
        )
        first = first + first_update
        second = second + second_update
        first = first + self.feed_forward(self.feed_forward_normalization_first(first))
        second = second + self.feed_forward(self.feed_forward_normalization_second(second))
        return first, second


class CrossTransformer(nn.Module):
    """Cross transformer with a learned reference-grid position encoding.

    The source code used a fixed square token grid.  Interpolating the learned
    position map retains the exact reference grid during training while allowing
    whole-image validation and test inference at larger spatial sizes.
    """

    def __init__(
        self,
        *,
        channels: int,
        dimension: int,
        depth: int,
        heads: int,
        mlp_dimension: int,
        base_map_size: int,
        dropout: float,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        if dimension % heads != 0:
            raise ConfigurationError("MCTNet transformer dimensions must be divisible by transformer_heads.")
        self.input_projection = nn.Linear(channels, dimension)
        self.output_projection = nn.Linear(dimension, channels)
        self.position_embedding = nn.Parameter(torch.empty(1, dimension, base_map_size, base_map_size))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.blocks = nn.ModuleList(
            CrossTransformerBlock(
                dimension,
                heads=heads,
                mlp_dimension=mlp_dimension,
                dropout=dropout,
                attention_dropout=attention_dropout,
            )
            for _ in range(depth)
        )

    def _tokens(self, value: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        batch, _, height, width = value.shape
        token_value = value.permute(0, 2, 3, 1).reshape(batch, height * width, -1)
        projected = self.input_projection(token_value)
        position = functional.interpolate(
            self.position_embedding,
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        ).flatten(2).transpose(1, 2)
        return projected + position.to(dtype=projected.dtype), (height, width)

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        if first.shape != second.shape:
            raise ContractError(
                "MCTNet cross-transformer inputs must have the same NCHW shape, got "
                f"{tuple(first.shape)} and {tuple(second.shape)}."
            )
        first_tokens, (height, width) = self._tokens(first)
        second_tokens, _ = self._tokens(second)
        for block in self.blocks:
            first_tokens, second_tokens = block(first_tokens, second_tokens)
        first_feature = self.output_projection(first_tokens).transpose(1, 2).reshape(
            first.shape[0],
            first.shape[1],
            height,
            width,
        )
        second_feature = self.output_projection(second_tokens).transpose(1, 2).reshape(
            second.shape[0],
            second.shape[1],
            height,
            width,
        )
        return first_feature + second_feature


class MCTNet(nn.Module):
    """Multi-hierarchical cross transformer for supervised HS--MS fusion."""

    def __init__(
        self,
        *,
        hsi_channels: int,
        msi_channels: int,
        scale: int = 4,
        base_hr_size: int = 64,
        transformer_depth: int = 5,
        transformer_heads: int = 8,
        transformer1_dim: int = 128,
        transformer2_dim: int = 64,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hsi_channels = int(hsi_channels)
        self.msi_channels = int(msi_channels)
        self.scale = int(scale)
        self._validate_configuration(
            base_hr_size=base_hr_size,
            transformer_depth=transformer_depth,
            transformer_heads=transformer_heads,
            transformer1_dim=transformer1_dim,
            transformer2_dim=transformer2_dim,
        )

        channels = self.hsi_channels
        self.msi_encoder = nn.Sequential(
            nn.Conv2d(self.msi_channels, 48, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.hsi_encoder = nn.Sequential(
            nn.Conv2d(channels, 156, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(156, 156, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(156, channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.coarse_transformer = CrossTransformer(
            channels=channels * 2,
            dimension=transformer1_dim,
            depth=transformer_depth,
            heads=transformer_heads,
            mlp_dimension=channels,
            base_map_size=base_hr_size // 16,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )
        self.fine_transformer = CrossTransformer(
            channels=channels,
            dimension=transformer2_dim,
            depth=transformer_depth,
            heads=transformer_heads,
            mlp_dimension=channels,
            base_map_size=base_hr_size // 4,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )
        self.coarse_decoder = StripDecoderBlock(channels * 6)
        self.coarse_projection = nn.Sequential(
            nn.Conv2d(channels * 3, channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fine_decoder = StripDecoderBlock(channels * 4)
        self.fine_projection = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.channel_attention_coarse = ChannelAttention(channels * 2)
        self.channel_attention_fine = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()
        self.reconstruction = nn.Sequential(
            nn.Conv2d(channels * 2 + self.msi_channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.spatial_refinement = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.spectral_refinement = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, lr_hsi: torch.Tensor, hr_msi: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(lr_hsi, hr_msi)
        fine_msi = self._attend(self.msi_encoder(hr_msi), self.channel_attention_fine)
        coarse_msi = self._attend(self.hsi_encoder(fine_msi), self.channel_attention_coarse)
        coarse_hsi = self._attend(self.hsi_encoder(lr_hsi), self.channel_attention_coarse)
        upsampled_hsi = functional.interpolate(
            lr_hsi,
            scale_factor=self.scale,
            mode="bilinear",
            align_corners=False,
        )
        upsampled_hsi = self._attend(upsampled_hsi, self.channel_attention_fine)

        coarse_cross = self.coarse_transformer(coarse_msi, coarse_hsi)
        coarse_features = self.coarse_decoder(torch.cat((coarse_msi, coarse_hsi, coarse_cross), dim=1))
        coarse_features = functional.interpolate(
            coarse_features,
            scale_factor=4,
            mode="bilinear",
            align_corners=False,
        )
        coarse_features = self.coarse_projection(coarse_features)

        fine_cross = self.fine_transformer(fine_msi, lr_hsi)
        fine_features = self.fine_decoder(
            torch.cat((fine_msi, lr_hsi, fine_cross, coarse_features), dim=1)
        )
        fine_features = functional.interpolate(
            fine_features,
            scale_factor=4,
            mode="bilinear",
            align_corners=False,
        )
        fine_features = self.fine_projection(fine_features)

        base = self.reconstruction(torch.cat((fine_features, hr_msi, upsampled_hsi), dim=1))
        spatial = base + self.spatial_refinement(base)
        return spatial + self.spectral_refinement(spatial)

    def _attend(self, value: torch.Tensor, channel_attention: ChannelAttention) -> torch.Tensor:
        return value * channel_attention(value) * self.spatial_attention(value)

    def _validate_configuration(
        self,
        *,
        base_hr_size: int,
        transformer_depth: int,
        transformer_heads: int,
        transformer1_dim: int,
        transformer2_dim: int,
    ) -> None:
        if self.scale != 4:
            raise ConfigurationError("MCTNet supports only scale=4 because its MCIM has two fixed 2x encoders.")
        if self.hsi_channels <= 0 or self.hsi_channels % 4 != 0:
            raise ConfigurationError("MCTNet hsi_channels must be a positive multiple of 4.")
        if self.msi_channels <= 0:
            raise ConfigurationError("MCTNet msi_channels must be positive.")
        if base_hr_size < 16 or base_hr_size % 16 != 0:
            raise ConfigurationError("MCTNet base_hr_size must be at least 16 and divisible by 16.")
        if transformer_depth <= 0 or transformer_heads <= 0:
            raise ConfigurationError("MCTNet transformer_depth and transformer_heads must be positive.")
        if transformer1_dim % transformer_heads != 0 or transformer2_dim % transformer_heads != 0:
            raise ConfigurationError("MCTNet transformer dimensions must be divisible by transformer_heads.")

    def _validate_inputs(self, lr_hsi: torch.Tensor, hr_msi: torch.Tensor) -> None:
        if lr_hsi.ndim != 4 or hr_msi.ndim != 4:
            raise ContractError("MCTNet expects lr_hsi and hr_msi in NCHW format.")
        if lr_hsi.shape[0] != hr_msi.shape[0]:
            raise ContractError("MCTNet lr_hsi and hr_msi must have the same batch size.")
        if lr_hsi.shape[1] != self.hsi_channels or hr_msi.shape[1] != self.msi_channels:
            raise ContractError(
                f"MCTNet expects {self.hsi_channels} HSI and {self.msi_channels} MSI channels, got "
                f"{lr_hsi.shape[1]} and {hr_msi.shape[1]}."
            )
        expected_size = (lr_hsi.shape[-2] * self.scale, lr_hsi.shape[-1] * self.scale)
        if hr_msi.shape[-2:] != expected_size:
            raise ContractError(
                "MCTNet requires HRMS dimensions to equal LRHS dimensions times scale=4; "
                f"expected {expected_size}, got {tuple(hr_msi.shape[-2:])}."
            )
        if hr_msi.shape[-2] % 16 != 0 or hr_msi.shape[-1] % 16 != 0:
            raise ContractError("MCTNet HRMS height and width must be divisible by 16.")
