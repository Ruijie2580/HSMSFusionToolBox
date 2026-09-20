"""S3O adapted to the toolbox NCHW fusion contract.

Source repository: https://github.com/GuobaPei/Selective-Spatial-Spectral-Operator
Paper: S3O: Selective Spatial-Spectral Operator for Cross-Scale Fusion
(Findings of CVPR 2026).

Upstream ships three dataset-specific wrappers, ``S3OCaveModel``,
``S3OHarvardModel`` and ``S3OChikuseiModel``, which differ only in
``guide_dim`` (128 for CAVE, 96 otherwise), in whether the encoder block holds a
single ``S3OEncoderBlock`` or a ``ModuleList`` of them, and in the width of the
reconstruction head.  This port keeps the Chikusei variant, the one whose
benchmark the toolbox shares, and exposes the differences as arguments.

All three upstream wrappers run as ``model(COORD, up_LR, RGB)`` and predict a
residual over the bicubic-upsampled LR-HSI::

    up_lr = bicubic(lr_hsi, scale)                    [B, h,     H, W]
    fx    = spa1(cat([up_lr, hr_msi], dim=1))         [B, g,     H, W]
    tok   = preprocess(fx)                            [B, H * W, n_hidden]
    tok   = blocks(tok)                               [B, H * W, n_hidden]
    return recon(tok) + up_lr                         [B, h,     H, W]

``blocks`` is the S3O backbone: a stack of stages, each a stride-2 depthwise
separable downsample followed by ``depths[i]`` S3O blocks, then a channel
adapter and a sub-pixel upsampler that restore the original resolution.  Each
S3O block runs a row/column spatial SSM and a channel SSM and multiplies the two
outputs, then routes the ``sel_ratio`` fraction of tokens with the highest
spatial-spectral non-separability score through an extra ``enhance`` MLP.

HSMS adaptation notes
---------------------
The upstream Chikusei protocol is x8 and its MSI is the three-band ``RGB``
dataset of its own HDF5 files.  The toolbox protocol is x4 and its MSI is the
eight-band WV2 observation, and the toolbox h5 carries only the reference cube,
so the observation loader is not ported at all: the network consumes ``lr_hsi``
and ``hr_msi`` as they arrive.  ``scale`` and the band counts become arguments,
and the concatenation order stays upstream's (HSI first, MSI second).

The one structural change is that upstream freezes the spatial size in the
network.  ``Block`` takes ``H`` and ``W`` at construction and uses them to
reshape the token sequence, so a checkpoint trained on 64x64 patches cannot be
applied to a larger scene: upstream works around this by rebuilding the model at
the evaluation resolution and loading the trained weights into it, in both
``evaluate()`` and ``test_chikusei.py``.  That workaround is only sound because
no parameter depends on the spatial size, so this port instead derives ``H`` and
``W`` from the tensor each ``Block`` receives, and threads them through
``S3OBackbone.forward`` and ``S3OEncoderBlock.forward`` as an explicit ``hw``
tuple in place of upstream's ``int(N ** 0.5)`` -- the same two numbers, but
correct for non-square feature maps too.  The two are numerically identical --
``H * W`` is exactly the token count either way -- and it lets one model
instance serve both the 64x64 training patches and the 256x256 test scenes,
which the toolbox's single-instance lifecycle requires.

One upstream behaviour is preserved rather than tidied: both ``S3OBackbone`` and
the outer model run ``self.apply(self._init_weights)``, so every ``nn.Linear``
under the backbone is drawn twice from ``trunc_normal_`` (the second draw is the
one that survives; the convolution and normalization branches are idempotent or
untouched).  It costs nothing to keep and it keeps the RNG stream in the same
shape as upstream under a fixed seed.

Like ``DHMFNet`` and ``SDPNet``, this model needs CUDA: mamba_ssm's
``selective_scan_fn`` is a CUDA kernel, and the ``selective_scan_ref`` fallback
that package ships is a separate numerical path, so it is deliberately not
wired in.  ``S3O.forward`` raises early with that explanation rather than
letting the kernel's own device assertion surface from inside a scan.

Removed relative to upstream, with forward numerics unchanged:
  - ``DSDownsample``, a depthwise separable downsample that the backbone never
    references (stages use ``ConvBN``).
  - the coordinate grid every wrapper builds in ``forward`` and hands to
    ``ImplicitHead``, which ignores it.  The computation, the ``coord_dim`` and
    ``n_layers`` arguments, and the trailing ``coord_hw`` parameter of
    ``ImplicitHead.forward`` all go with it.  Upstream's own comments mark the
    coordinates as "unused, kept for compatibility".
  - ``spec_dim``, threaded from the training script through every wrapper into
    ``S3OEncoderBlock`` and marked "unused, kept for compatibility" there.
  - the ``timm`` dependency.  Upstream imports ``DropPath`` and
    ``trunc_normal_`` from ``timm.layers``, an API that only exists in
    timm >= 0.9 and is not a declared dependency of the toolbox;
    ``DropPath`` is inlined here and ``trunc_normal_`` comes from
    ``torch.nn.init``, as in ``CLSNet`` and ``DPFormer``.
  - the unused ``hsi_dim`` / ``msi_dim`` locals of each wrapper, which were
    named the wrong way round relative to the tensors concatenated beneath them.

``sel_ratio`` is threaded down to every ``Block`` and defaults to upstream's
0.10, the value a freshly constructed model uses at test time.  The
progressive-enhancement schedule that varies it during training -- upstream's
``--sel_warmup 0.10 --sel_ramp 0.20 --sel_max 0.10``, applied by its
``set_enhancement_ratio`` -- lives in the Lightning wrapper instead, since it is
a training-time schedule rather than a network property.  ``S3O`` still exposes
``set_enhancement_ratio`` for it to call.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from torch import Tensor


def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:
    """Stochastic depth, inlined from ``timm.layers.drop``."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    """Per-sample stochastic depth with the module interface."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training)


def default_conv(in_channels: int, out_channels: int, kernel_size: int, bias: bool = True) -> nn.Conv2d:
    """Default convolution with symmetric padding."""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias
    )


class Upsampler(nn.Sequential):
    """Sub-pixel upsampler for power-of-two scaling factors."""

    def __init__(self, conv, scale: int, n_feats: int) -> None:
        layers = []
        if scale & (scale - 1) == 0:  # scale is a power of two
            for _ in range(int(math.log2(scale))):
                layers.append(conv(n_feats, 4 * n_feats, 3))
                layers.append(nn.PixelShuffle(2))
        elif scale == 3:
            layers.append(conv(n_feats, 9 * n_feats, 3))
            layers.append(nn.PixelShuffle(3))
        else:
            raise NotImplementedError
        super().__init__(*layers)


class ConvBN(nn.Sequential):
    """Convolution followed by an optional batch normalization."""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        with_bn: bool = True,
    ) -> None:
        if with_bn:
            super().__init__(
                nn.Conv2d(
                    in_planes, out_planes, kernel_size, stride, padding, dilation, groups, bias=False
                ),
                nn.BatchNorm2d(out_planes),
            )
        else:
            super().__init__(
                nn.Conv2d(
                    in_planes, out_planes, kernel_size, stride, padding, dilation, groups, bias=True
                )
            )


class Block(nn.Module):
    """The S3O block: spatial SSM times channel SSM, plus selective enhancement.

    ``H`` and ``W`` are taken from the input tensor rather than stored at
    construction; see the module docstring.
    """

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        N_state_s: int = 32,
        N_state_c: int = 32,
        A_margin: float = 1e-3,
        sel_ratio: float = 0.10,
        sel_groups: int = 4,
        sel_kernel: int = 5,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.sel_ratio = sel_ratio
        self.sel_groups = sel_groups
        self.sel_kernel = sel_kernel

        # Pre-norm and projections
        self.norm1 = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, dim, bias=True)

        # Spatial SSM parameters, with a margin keeping the real part negative
        self.logA_s = nn.Parameter(torch.zeros(dim, N_state_s))
        self.B_s = nn.Parameter(0.05 * torch.randn(dim, N_state_s))
        self.C_s = nn.Parameter(0.05 * torch.randn(dim, N_state_s))
        self.D_s = nn.Parameter(torch.zeros(dim))
        self.log_dt_s = nn.Parameter(torch.zeros(dim))
        self.A_margin = A_margin

        # Channel SSM parameters
        self.logA_c = nn.Parameter(torch.zeros(dim, N_state_c))
        self.B_c = nn.Parameter(0.05 * torch.randn(dim, N_state_c))
        self.C_c = nn.Parameter(0.05 * torch.randn(dim, N_state_c))
        self.D_c = nn.Parameter(torch.zeros(dim))
        self.log_dt_c = nn.Parameter(torch.zeros(dim))

        # Output head with gating
        self.norm2 = nn.LayerNorm(dim)
        self.gate_proj = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(dim, dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        # Row/column fusion gating, initialized to 0.5 after the sigmoid
        self.mix_row = nn.Parameter(torch.zeros(dim))

        # Enhancement head for the selected difficult tokens
        self.enhance = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

    def _A_param(self, logA: Tensor) -> Tensor:
        """Keep ``A`` strictly negative for stability."""
        return -F.softplus(logA) - self.A_margin

    def spatial_ssm(self, x_seq: Tensor, height: int, width: int) -> Tensor:
        """Spatial SSM with bidirectional row and column scanning."""
        batch, _, channels = x_seq.shape
        x = x_seq.transpose(1, 2).reshape(batch, channels, height, width)

        a = self._A_param(self.logA_s).contiguous()

        # Row-wise scan: B * H sequences of length W
        u_row = x.permute(0, 2, 1, 3).reshape(batch * height, channels, width).contiguous()
        dt_row = self.log_dt_s.view(1, channels, 1).expand(batch * height, channels, width).contiguous()
        y_row = selective_scan_fn(
            u_row.float(), dt_row.float(), a.float(),
            self.B_s.float(), self.C_s.float(), self.D_s.float(),
            None, None, True,
        )
        if not torch.is_tensor(y_row):
            y_row = y_row[0]
        y_row = y_row.view(batch, height, channels, width).permute(0, 2, 1, 3)

        # Column-wise scan: B * W sequences of length H
        u_col = x.permute(0, 3, 1, 2).reshape(batch * width, channels, height).contiguous()
        dt_col = self.log_dt_s.view(1, channels, 1).expand(batch * width, channels, height).contiguous()
        y_col = selective_scan_fn(
            u_col.float(), dt_col.float(), a.float(),
            self.B_s.float(), self.C_s.float(), self.D_s.float(),
            None, None, True,
        )
        if not torch.is_tensor(y_col):
            y_col = y_col[0]
        y_col = y_col.view(batch, width, channels, height).permute(0, 2, 3, 1)

        # Adaptive fusion with learnable channel-wise gating
        w = torch.sigmoid(self.mix_row).view(1, channels, 1, 1)
        y = w * y_row + (1.0 - w) * y_col

        return y.permute(0, 2, 3, 1).reshape(batch, height * width, channels).contiguous()

    def channel_ssm(self, x_seq: Tensor) -> Tensor:
        """Channel SSM for cross-band spectral dependencies."""
        batch, length, channels = x_seq.shape
        u = x_seq.transpose(1, 2).contiguous()
        dt_param = self.log_dt_c.view(1, channels, 1).expand(batch, channels, length).contiguous()
        a = self._A_param(self.logA_c).contiguous()
        y = selective_scan_fn(
            u.float(), dt_param.float(), a.float(),
            self.B_c.float(), self.C_c.float(), self.D_c.float(),
            None, None, True,
        )
        if not torch.is_tensor(y):
            y = y[0]
        return y.transpose(1, 2).contiguous()

    @torch.no_grad()
    def _select_topk_indices(self, u_s: Tensor, u_c: Tensor, height: int, width: int) -> Tensor:
        """Select the tokens with the highest spatial-spectral non-separability."""
        batch, length, channels = u_s.shape
        ksize = int(self.sel_kernel)
        groups = max(1, min(int(self.sel_groups), channels))

        # Adjust the group count to divide the channel count
        while channels % groups != 0 and groups > 1:
            groups -= 1
        channels_per_group = channels // groups

        u_s_spatial = u_s.transpose(1, 2).reshape(batch, channels, height, width)
        u_c_spatial = u_c.transpose(1, 2).reshape(batch, channels, height, width)

        # Group-wise averaging
        u_s_grouped = u_s_spatial.view(batch, groups, channels_per_group, height, width).mean(dim=2)
        u_c_grouped = u_c_spatial.view(batch, groups, channels_per_group, height, width).mean(dim=2)

        # Unfold the neighbourhood into (B, G * K^2, L) and then (B, L, K^2, G)
        kernel_area = ksize * ksize
        u_s_unf = F.unfold(u_s_grouped, kernel_size=ksize, padding=ksize // 2)
        u_c_unf = F.unfold(u_c_grouped, kernel_size=ksize, padding=ksize // 2)
        u_s_patch = u_s_unf.transpose(1, 2).view(batch, length, groups, kernel_area).transpose(2, 3)
        u_c_patch = u_c_unf.transpose(1, 2).view(batch, length, groups, kernel_area).transpose(2, 3)

        # Center within the neighbourhood
        u_s_patch = u_s_patch - u_s_patch.mean(dim=2, keepdim=True)
        u_c_patch = u_c_patch - u_c_patch.mean(dim=2, keepdim=True)

        # Symmetric whitening, so the correlation is scale invariant
        tau = 1e-6
        var_s = (u_s_patch**2).mean(dim=2) + tau
        var_c = (u_c_patch**2).mean(dim=2) + tau
        u_s_white = u_s_patch * var_s.rsqrt().unsqueeze(2)
        u_c_white = u_c_patch * var_c.rsqrt().unsqueeze(2)

        # Cross-correlation matrix R: (B, L, G, G)
        correlation = torch.einsum("blkg,blkh->blgh", u_s_white, u_c_white) / float(kernel_area)

        # Off-diagonal energy as the difficulty measure
        eps = 1e-6
        energy_all = (correlation**2).sum(dim=(-1, -2)) + eps
        energy_diag = (correlation.diagonal(dim1=-2, dim2=-1) ** 2).sum(-1)
        difficulty = 1.0 - (energy_diag / energy_all)

        k = max(1, int(self.sel_ratio * length))
        return torch.topk(difficulty, k=k, dim=1, largest=True).indices

    def forward(self, x: Tensor) -> Tensor:
        _, _, height, width = x.shape
        x_seq = x.flatten(2).transpose(1, 2)
        shortcut = x_seq

        x_proj = self.in_proj(self.norm1(x_seq))

        # Dual-branch SSM
        u_s = self.spatial_ssm(x_proj, height, width)
        u_c = self.channel_ssm(x_proj)
        f_x = u_s * u_c

        # Progressive enhancement for the difficult tokens
        if self.sel_ratio > 0:
            idx = self._select_topk_indices(u_s, u_c, height, width)
            batch = f_x.shape[0]
            rows = torch.arange(batch, device=f_x.device)[:, None]
            delta = torch.zeros_like(f_x)
            delta[rows, idx, :] = self.enhance(f_x[rows, idx, :])
            f_x = f_x + delta

        # Output head with gating
        f_x = self.norm2(f_x)
        gated = self.act(self.gate_proj(f_x)) * f_x
        h_x = self.out_proj(gated)
        return (shortcut + self.drop_path(h_x)).transpose(1, 2).reshape(
            shortcut.shape[0], -1, height, width
        )


class S3OBackbone(nn.Module):
    """Multi-stage SSM backbone with hierarchical feature extraction."""

    def __init__(
        self,
        dim: int,
        depths: list[int] | None = None,
        drop_path_rate: float = 0.0,
        sel_ratio: float = 0.10,
    ) -> None:
        super().__init__()
        depths = [1, 1, 1] if depths is None else list(depths)
        self.dim = dim

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList()
        cur = 0
        current_dim = dim
        max_out_channels = dim

        for i_layer in range(len(depths)):
            if i_layer > 0:
                # Downsample and increase channels for deeper stages
                down_sampler = ConvBN(current_dim, current_dim * 2, kernel_size=3, stride=2, padding=1)
                current_dim = current_dim * 2
            else:
                down_sampler = nn.Identity()

            blocks = [
                Block(current_dim, dpr[cur + block_idx], sel_ratio=sel_ratio)
                for block_idx in range(depths[i_layer])
            ]
            self.stages.append(nn.Sequential(down_sampler, *blocks))
            cur += depths[i_layer]
            max_out_channels = max(max_out_channels, current_dim)

        # Channel adapter and upsampler restoring the original resolution
        self.channel_adapter = nn.Conv2d(max_out_channels, self.dim, kernel_size=1)
        total_scale = 2 ** (len(depths) - 1)
        self.upsampler = (
            Upsampler(default_conv, total_scale, n_feats=self.dim) if total_scale > 1 else nn.Identity()
        )
        self.norm = nn.LayerNorm(self.dim)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: Tensor, hw: tuple[int, int]) -> Tensor:
        """Run the stages over a token sequence and return a token sequence.

        Upstream recovers ``H`` and ``W`` from ``x.shape[1]`` by taking its square
        root, which only holds for square inputs and cannot know the intended
        layout for a non-square one.  ``hw`` is passed explicitly instead; it is
        the same two numbers the caller derived from the feature map.
        """
        height, width = hw
        batch, length, channels = x.shape
        if length != height * width:
            raise ValueError(
                f"S3OBackbone got {length} tokens for a {height}x{width} feature map."
            )
        if channels != self.dim:
            raise ValueError(
                f"S3OBackbone expects {self.dim} channels, got {channels}."
            )

        feat = x.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        for stage in self.stages:
            feat = stage(feat)

        if feat.shape[1] != self.dim:
            feat = self.channel_adapter(feat)
        feat = self.upsampler(feat)

        if feat.shape[-2:] != (height, width):
            raise ValueError(
                f"S3OBackbone must restore the input resolution, got "
                f"{tuple(feat.shape[-2:])} from {height}x{width}. The spatial size has "
                f"to be divisible by {2 ** len(self.stages)} for the downsampling "
                f"and sub-pixel upsampling to round-trip."
            )
        return self.norm(feat.permute(0, 2, 3, 1).reshape(batch, length, self.dim))


ACTIVATION = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU(0.1),
    "softplus": nn.Softplus,
    "ELU": nn.ELU,
    "silu": nn.SiLU,
}


class ChannelAttention(nn.Module):
    """Channel attention from max and average pooling."""

    def __init__(self, in_planes: int) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid = max(1, in_planes // 4)
        self.fc1 = nn.Conv2d(in_planes, mid, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(mid, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpaPyBlock(nn.Module):
    """Spatial pyramid block combining 1x, 1/2x and 1/4x features with gating."""

    def __init__(self, inchannels: int, outchannels: int, bias: bool = True) -> None:
        super().__init__()
        self.scale1 = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, 3, 1, 1, bias=False), nn.BatchNorm2d(inchannels), nn.GELU()
        )
        self.scale1_2 = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, 3, 2, 1, bias=False), nn.BatchNorm2d(inchannels), nn.GELU()
        )
        self.scale1_4 = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, 3, 4, 1, bias=False), nn.BatchNorm2d(inchannels), nn.GELU()
        )

        self.channel = ChannelAttention(inchannels * 3)
        self.out = nn.Sequential(
            nn.Conv2d(inchannels * 3, outchannels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.GELU(),
        )

        # Learnable weights for the multi-scale fusion
        self.gamma_low2 = nn.Parameter(torch.tensor(0.1))
        self.gamma_low4 = nn.Parameter(torch.tensor(0.05))

        self.skip_proj = (
            nn.Conv2d(inchannels, outchannels, kernel_size=1, bias=bias)
            if inchannels != outchannels
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        y1 = self.scale1(x)
        y2 = F.interpolate(self.scale1_2(x), scale_factor=2.0, mode="bilinear", align_corners=False)
        y3 = F.interpolate(self.scale1_4(x), scale_factor=4.0, mode="bilinear", align_corners=False)

        y = torch.cat((y1, self.gamma_low2 * y2, self.gamma_low4 * y3), dim=1)
        y = self.channel(y) * y
        return self.out(y) + self.skip_proj(x)


class MLP(nn.Module):
    """Multi-layer perceptron with optional residual connections."""

    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        n_output: int,
        n_layers: int = 1,
        act: str = "gelu",
        res: bool = True,
    ) -> None:
        super().__init__()
        if act not in ACTIVATION:
            raise NotImplementedError
        act_cls = ACTIVATION[act]
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act_cls())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList(
            [nn.Sequential(nn.Linear(n_hidden, n_hidden), act_cls()) for _ in range(n_layers)]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            x = self.linears[i](x) + x if self.res else self.linears[i](x)
        return self.linear_post(x)


class ImplicitHead(nn.Module):
    """Lightweight depthwise-separable reconstruction head."""

    def __init__(self, in_channels: int, out_dim: int = 1, hidden: int = 128, act: str = "gelu") -> None:
        super().__init__()
        if act not in ACTIVATION:
            raise NotImplementedError
        act_cls = ACTIVATION[act]
        self.proj = nn.Conv2d(in_channels, hidden, kernel_size=1, bias=True)
        self.act = act_cls()
        self.dw = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=True)
        self.pw = nn.Conv2d(hidden, out_dim, kernel_size=1, bias=True)

    def forward(self, feat_hw: Tensor) -> Tensor:
        x = self.act(self.proj(feat_hw.permute(0, 3, 1, 2)))
        x = self.act(self.dw(x))
        return self.pw(x)


class S3OEncoderBlock(nn.Module):
    """SSM-based encoder block with LayerScale residual connections."""

    def __init__(
        self,
        hidden_dim: int,
        act: str = "gelu",
        mlp_ratio: int = 1,
        last_layer: bool = True,
        out_dim: int = 1,
        depths: list[int] | None = None,
        drop_path_rate: float = 0.1,
        sel_ratio: float = 0.10,
    ) -> None:
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)

        self.ssm = S3OBackbone(
            dim=hidden_dim, depths=depths, drop_path_rate=drop_path_rate, sel_ratio=sel_ratio
        )

        self.ln_2 = nn.LayerNorm(hidden_dim)
        # The SSM architecture does not need a large expansion ratio
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)

        # LayerScale: learnable scaling factors for the residual branches
        self.gamma1 = nn.Parameter(torch.ones(hidden_dim) * 1e-6)
        self.gamma2 = nn.Parameter(torch.ones(hidden_dim) * 1e-6)

        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx: Tensor, hw: tuple[int, int]) -> Tensor:
        """``fx`` is a token sequence ``(B, N, C)``, as is the return value."""
        res_ssm = self.ssm(self.ln_1(fx), hw)
        fx = fx + self.gamma1 * res_ssm

        res_mlp = self.mlp(self.ln_2(fx))
        fx = fx + self.gamma2 * res_mlp

        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx


class S3O(nn.Module):
    """Selective Spatial-Spectral Operator for HSI-MSI fusion.

    This is upstream's ``S3OChikuseiModel`` with its fixed spatial size, its
    unused coordinate input and its ``spec_dim`` argument removed.
    """

    def __init__(
        self,
        hsi_channels: int,
        msi_channels: int,
        scale: int = 4,
        n_hidden: int = 64,
        guide_dim: int = 96,
        act: str = "gelu",
        mlp_ratio: int = 1,
        depths: list[int] | None = None,
        drop_path_rate: float = 0.1,
        sel_ratio: float = 0.10,
    ) -> None:
        super().__init__()
        if scale < 1:
            raise ValueError(f"scale must be a positive integer, got {scale}.")

        self.hsi_channels = int(hsi_channels)
        self.msi_channels = int(msi_channels)
        self.scale = int(scale)
        self.n_hidden = int(n_hidden)

        # Multi-scale spatial pyramid feature extraction
        self.spa1 = SpaPyBlock(self.hsi_channels + self.msi_channels, guide_dim)

        # Feature preprocessing
        self.preprocess = MLP(guide_dim, self.n_hidden * 2, self.n_hidden, n_layers=0, res=False, act=act)

        # SSM-based encoder
        self.blocks = S3OEncoderBlock(
            hidden_dim=self.n_hidden,
            act=act,
            mlp_ratio=mlp_ratio,
            out_dim=self.n_hidden,
            depths=depths,
            drop_path_rate=drop_path_rate,
            sel_ratio=sel_ratio,
            last_layer=False,
        )

        # Lightweight reconstruction head
        self.recon = ImplicitHead(
            in_channels=self.n_hidden,
            out_dim=self.hsi_channels,
            hidden=int(self.n_hidden * 1.5),
            act=act,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def set_enhancement_ratio(self, ratio: float) -> None:
        """Set the selective-enhancement ratio on every S3O block."""
        for module in self.modules():
            if hasattr(module, "sel_ratio"):
                module.sel_ratio = float(ratio)

    def _check_geometry(self, lr_hsi: Tensor, hr_msi: Tensor) -> None:
        if lr_hsi.shape[1] != self.hsi_channels:
            raise ValueError(
                f"LR HSI should carry {self.hsi_channels} bands, got {lr_hsi.shape[1]}."
            )
        if hr_msi.shape[1] != self.msi_channels:
            raise ValueError(
                f"HR MSI should carry {self.msi_channels} bands, got {hr_msi.shape[1]}."
            )
        _, _, hr_height, hr_width = hr_msi.shape
        if hr_height % 4 or hr_width % 4:
            raise ValueError(
                f"S3O needs the HR side divisible by 4: the spatial pyramid has a "
                f"stride-4 branch that is bilinearly upsampled back by 4, and the "
                f"backbone has two stride-2 stages. Got {hr_height}x{hr_width}."
            )
        expected_lr = (hr_height // self.scale, hr_width // self.scale)
        _, _, lr_height, lr_width = lr_hsi.shape
        if (lr_height, lr_width) != expected_lr:
            raise ValueError(
                f"LR HSI should be {expected_lr[0]}x{expected_lr[1]} for a "
                f"{hr_height}x{hr_width} HR input at scale {self.scale}, got "
                f"{lr_height}x{lr_width}."
            )

    def forward(self, lr_hsi: Tensor, hr_msi: Tensor) -> Tensor:
        self._check_geometry(lr_hsi, hr_msi)
        if not lr_hsi.is_cuda:
            raise RuntimeError(
                "S3O runs its state-space scans with mamba_ssm's selective_scan_fn, "
                "which is a CUDA kernel; a CPU fallback would change the numerics "
                f"relative to upstream. Move the model and inputs to CUDA (got "
                f"{lr_hsi.device})."
            )

        up_lr = F.interpolate(lr_hsi, scale_factor=self.scale, mode="bicubic", align_corners=False)

        # Upstream concatenates HSI first and MSI second
        fx = self.spa1(torch.cat((up_lr, hr_msi), dim=1))
        batch, channels, height, width = fx.shape

        tokens = self.preprocess(fx.permute(0, 2, 3, 1).reshape(batch, height * width, channels))
        tokens = self.blocks(tokens, (height, width))

        return self.recon(tokens.reshape(batch, height, width, -1)) + up_lr
