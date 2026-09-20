import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch import einsum
from pytorch_wavelets import DWTInverse
from pytorch_wavelets import DWTForward


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class DynamicPosBias(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads),
        )

    def forward(self, biases):
        pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos

    def flops(self, N):
        flops = N * 2 * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.num_heads
        return flops


class multiheadAttention(nn.Module):
    r"""Multi-head self attention module with dynamic position bias.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim=180,
        num_heads=6,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        position_bias=True,
        window_size=(8, 8),
    ):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.position_bias = position_bias
        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)
        self.window_size = window_size

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_groups*B, N, C)
            mask: (0/-inf) mask with shape of (num_groups, Gh*Gw, Gh*Gw) or None
            H: height of each group
            W: width of each group
        """
        _, h, w, _ = x.shape
        H, W = self.window_size[0], self.window_size[1]
        x = rearrange(x, "b (h b0) (w b1) c -> (b h w) (b0 b1) c", b0=H, b1=W)
        B_, N, C = x.shape
        assert N == H * W
        x = x.reshape(B_, N, C)
        group_size = (H, W)
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (B_, self.num_heads, N, N), N = H*W

        if self.position_bias:
            # generate mother-set
            position_bias_h = torch.arange(
                1 - group_size[0], group_size[0], device=attn.device
            )
            position_bias_w = torch.arange(
                1 - group_size[1], group_size[1], device=attn.device
            )
            biases = torch.stack(
                torch.meshgrid([position_bias_h, position_bias_w])
            )  # 2, 2Gh-1, 2W2-1
            biases = (
                biases.flatten(1).transpose(0, 1).contiguous().float()
            )  # (2h-1)*(2w-1) 2

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(group_size[0], device=attn.device)
            coords_w = torch.arange(group_size[1], device=attn.device)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Gh, Gw
            coords_flatten = torch.flatten(coords, 1)  # 2, Gh*Gw
            relative_coords = (
                coords_flatten[:, :, None] - coords_flatten[:, None, :]
            )  # 2, Gh*Gw, Gh*Gw
            relative_coords = relative_coords.permute(
                1, 2, 0
            ).contiguous()  # Gh*Gw, Gh*Gw, 2
            relative_coords[:, :, 0] += group_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += group_size[1] - 1
            relative_coords[:, :, 0] *= 2 * group_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Gh*Gw, Gh*Gw

            pos = self.pos(biases)  # 2Gh-1 * 2Gw-1, heads
            # select position bias
            relative_position_bias = pos[relative_position_index.view(-1)].view(
                group_size[0] * group_size[1], group_size[0] * group_size[1], -1
            )  # Gh*Gw,Gh*Gw,nH
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1
            ).contiguous()  # nH, Gh*Gw, Gh*Gw
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nP = mask.shape[0]
            attn = attn.view(B_ // nP, nP, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(
                0
            )  # (B, nP, nHead, N, N)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = rearrange(
            x, "(b h w) (b0 b1) c -> b (h b0) (w b1) c", b0=H, b1=W, h=h // H, w=w // W
        )

        return x


class HSAB(nn.Module):
    def __init__(
        self,
        dim,
        window_size=(8, 8),
        dim_head=64,
        heads=8,
        num_blocks=2,
        depth=1,
        SE_scale=0.01,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        self.depth = depth
        self.SE_scale = SE_scale
        self.shift_size = window_size[0] // 2
        for _ in range(num_blocks):
            self.blocks.append(
                nn.ModuleList(
                    [
                        nn.LayerNorm(dim),
                        multiheadAttention(
                            dim=dim, window_size=window_size, num_heads=heads
                        ),
                        CALayer(dim, reduction=3),
                        nn.LayerNorm(dim),
                        FeedForward(dim=dim),
                    ]
                )
            )

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for norm1, attn, SE, norm2, ff in self.blocks:
            shortcut = x
            x = norm1(x)
            SE_x = SE(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

            attn_x = attn(x)

            x = attn_x + self.SE_scale * SE_x + shortcut
            x = x + ff(norm2(x))
        out = x.permute(0, 3, 1, 2)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


class HST(nn.Module):
    def __init__(self, in_dim=28, out_dim=28, dim=28, num_blocks=[1, 1, 1]):
        super(HST, self).__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.scales = len(num_blocks)

        self.D = DWTForward(J=1, wave="haar", mode="zero")
        self.Dt = DWTInverse(wave="haar", mode="zero")

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_scale = dim
        for i in range(self.scales - 1):
            self.encoder_layers.append(
                nn.ModuleList(
                    [
                        HSAB(
                            dim=dim_scale,
                            num_blocks=num_blocks[i],
                            dim_head=dim,
                            heads=dim_scale // dim,
                            depth=i,
                        ),
                        nn.Conv2d(dim_scale * 4, dim_scale * 2, 3, 1, 1, bias=False),
                    ]
                )
            )
            dim_scale *= 2

        # Bottleneck
        self.bottleneck = HSAB(
            dim=dim_scale,
            dim_head=dim,
            heads=dim_scale // dim,
            num_blocks=num_blocks[-1],
            depth=self.scales - 1,
        )

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(self.scales - 1):
            self.decoder_layers.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(
                            dim_scale,
                            dim_scale * 2,
                            stride=1,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                        nn.Conv2d(dim_scale * 4, dim_scale * 2, 1, 1, bias=False),
                        HSAB(
                            dim=dim_scale // 2,
                            num_blocks=num_blocks[self.scales - 2 - i],
                            dim_head=dim,
                            heads=(dim_scale // 2) // dim,
                            depth=self.scales - 2 - i,
                        ),
                    ]
                )
            )
            dim_scale //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        #### activation function
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """

        b, c, h_inp, w_inp = x.shape
        hb, wb = 16, 16
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode="reflect")

        # Embedding
        fea = self.embedding(x)
        x = x[:, : self.out_dim, :, :]

        # Encoder
        fea_encoder = []
        for HSAB, FeaDownSample in self.encoder_layers:
            fea = HSAB(fea)
            fea, HF = self.D(fea)
            HF = rearrange(HF[0], "b c n w h -> b (c n) w h")
            fea = torch.cat([fea, HF], dim=1)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

        # Bottleneck
        fea = self.bottleneck(fea)

        # Decoder
        for i, (FeaUpSample, Fution, HSAB) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.scales - 2 - i]], dim=1))
            channel = fea.shape[1]
            HF = fea[:, channel // 4 :, :, :]
            LF = fea[:, : channel // 4, :, :]
            HF = rearrange(HF, "b (c n) w h -> b c n w h", n=3)
            fea = self.Dt((LF, [HF]))
            fea = HSAB(fea)

        # Mapping
        out = self.mapping(fea) + x
        return out[:, :, :h_inp, :w_inp]


class HyPaNet3(nn.Module):
    def __init__(self, in_nc=29, out_nc=8, channel=64):
        super(HyPaNet3, self).__init__()
        self.fution = nn.Conv2d(in_nc, channel, 1, 1, 0, bias=True)
        self.down_sample = nn.Conv2d(channel, channel, 3, 2, 1, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, out_nc, 1, padding=0, bias=True),
            nn.Softplus(),
        )
        self.relu = nn.ReLU(inplace=True)
        self.out_nc = out_nc

    def forward(self, x):
        x = self.down_sample(self.relu(self.fution(x)))
        x = self.avg_pool(x)
        x = self.mlp(x) + 1e-6
        return (
            x[:, : self.out_nc // 3, :, :],
            x[:, self.out_nc // 3 : 2 * self.out_nc // 3, :, :],
            x[:, 2 * self.out_nc // 3 :, :, :],
        )


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class SAM(nn.Module):
    def __init__(self, ms_dim, in_c, bias=True):
        super(SAM, self).__init__()
        self.conv1 = nn.Conv2d(
            in_c, in_c, kernel_size=1, padding=0, stride=1, bias=bias
        )
        self.conv2 = nn.Conv2d(
            in_c, in_c, kernel_size=1, padding=0, stride=1, bias=bias
        )
        self.conv3 = nn.Conv2d(
            ms_dim, in_c, kernel_size=1, padding=0, stride=1, bias=bias
        )
        self.conv4 = nn.Conv2d(
            in_c * 3, 2 * in_c + 1, kernel_size=1, padding=0, stride=1, bias=bias
        )
        self.ca = nn.Sequential(CALayer(in_c * 2 + 1, reduction=8))

    def forward(self, z, LR_up, HR):
        z = self.conv1(z)
        LR_up = self.conv2(LR_up)
        HR = self.conv3(HR)
        r_value = self.ca(self.conv4(torch.cat([z, LR_up, HR], dim=1)))
        gamma = r_value[:, -1, :, :]
        mu, eta = r_value[:, :-1, :, :].chunk(2, dim=1)
        return mu, eta, gamma.unsqueeze(1)


class init(nn.Module):
    def __init__(self, in_dim=31 + 3, out_dim=31, dim=64):
        super().__init__()
        self.map0 = nn.Conv2d(
            in_dim, dim, kernel_size=3, padding=1, stride=1, bias=False
        )

        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1),
            nn.GELU(),
            CALayer(dim, reduction=8),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1),
        )

        self.map1 = nn.Conv2d(
            dim, out_dim, kernel_size=3, padding=1, stride=1, bias=False
        )

    def forward(self, z):
        z = self.map0(z)
        z = z + self.conv(z)
        out = self.map1(z)
        return out


class CAUWT(nn.Module):
    def __init__(self, channel=31, ms_channel=3, scale=4, num_iterations=3):
        super(CAUWT, self).__init__()
        self.num_iterations = num_iterations
        self.scale = scale
        self.denoisers = nn.ModuleList([])
        self.sams = nn.ModuleList([])
        for _ in range(num_iterations):
            self.denoisers.append(
                HST(in_dim=channel + 1, out_dim=channel, dim=32, num_blocks=[1, 1, 1]),
            )
            self.sams.append(SAM(ms_dim=ms_channel, in_c=channel))
        self.init = init(in_dim=channel + ms_channel, out_dim=channel, dim=64)

        self.conv_downsample = torch.nn.Conv2d(
            in_channels=channel,
            out_channels=channel,
            kernel_size=13,
            stride=scale,
            padding=13 // 2,
        )
        self.conv_upsample = torch.nn.ConvTranspose2d(
            in_channels=channel,
            out_channels=channel,
            kernel_size=13,
            stride=scale,
            padding=13 // 2,
        )
        self.conv_torgb = torch.nn.Conv2d(
            in_channels=channel,
            out_channels=ms_channel,
            kernel_size=3,
            stride=1,
            padding=3 // 2,
        )
        self.conv_tohsi = torch.nn.Conv2d(
            in_channels=ms_channel,
            out_channels=channel,
            kernel_size=3,
            stride=1,
            padding=3 // 2,
        )

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, LR, HR):
        out = []
        LR_up = torch.nn.functional.interpolate(
            LR, scale_factor=self.scale, mode="bicubic", align_corners=False
        )

        z = LR_up + self.init(torch.cat([LR_up, HR], dim=1))
        sz = z.shape
        m = z
        out.append(m)
        for i in range(self.num_iterations):
            mu, eta, gamma = self.sams[i](z, LR_up, HR)
            # z-subproblem
            nbla_z = (
                -self.conv_tohsi(HR - self.conv_torgb(z))
                - self.conv_upsample(LR - self.conv_downsample(z), sz)
                + mu * (z - m)
            )
            z = z - eta * nbla_z
            # m-subproblem
            m = self.denoisers[i](torch.cat([z, gamma], dim=1))
            if i < self.num_iterations - 1:
                out.append(m)
        # return out, m
        return m
