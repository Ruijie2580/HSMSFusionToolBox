import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from mamba_ssm.modules.mamba_simple import Mamba
from numpy.ma.core import transpose
from thop import profile
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat

device = torch.device("cuda")


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        if len(x.shape) == 4:
            h, w = x.shape[-2:]
            return to_4d(self.body(to_3d(x)), h, w)
        else:
            return self.body(x)


class PatchUnEmbed(nn.Module):
    def __init__(self, basefilter) -> None:
        super().__init__()
        self.nc = basefilter

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.nc, x_size[0], x_size[1])  # B Ph*Pw C
        return x


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        patch_size=1,
        stride=1,
        in_chans=64,
        embed_dim=64,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        # patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride
        )
        self.norm = LayerNorm(embed_dim, "BiasFree")

    def forward(self, x):

        B, C, H, W = x.shape

        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC

        return x


class HinResBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.2, use_HIN=True):
        super(HinResBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        resi = self.relu_1(self.conv_1(x))
        out_1, out_2 = torch.chunk(resi, 2, dim=1)
        resi = torch.cat([self.norm(out_1), out_2], dim=1)
        resi = self.relu_2(self.conv_2(resi))
        return x + resi


class ChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.attention(x)
        return y


class SpatialAttention(nn.Module):

    def __init__(self, up_size, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.up_size = up_size  # 上采样尺寸
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, x):

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        x_concat = torch.cat([avg_out, max_out], dim=1)

        attention_map = F.interpolate(
            x_concat, size=(self.up_size, self.up_size), mode="area"
        )

        attention_map = self.spatial_attention(attention_map)

        attention_map = attention_map.view(attention_map.size(0), -1)  #
        output = attention_map.view(-1, self.up_size * self.up_size, 1)
        # 32 6400 1
        return output


class SsmChannel(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        exhr_msid=2.0,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.0,
        conv_bias=True,
        bias=False,
        image_size=64,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.exhr_msid = exhr_msid
        self.d_inner = int(self.exhr_msid * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=3,
            padding=1,
            **factory_kwargs,
        )
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(
            3, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, 3, bias=True, **factory_kwargs)
        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=3,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        # D "skip" parameter
        self.Ds = nn.Parameter(torch.ones(3, device=device))  # Keep in fp32
        self.Ds._no_weight_decay = True
        self.selective_scan = selective_scan_fn
        # self.selective_scan = selective_scan_ref

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        self.patch_num = (image_size // 16, image_size // 4, image_size)
        self.convEnd = nn.Conv2d(
            self.d_model * 2, self.d_model * 2, kernel_size=1, device=device
        )
        self.CA = ChannelAttention(self.d_model * 2)

    def img_to_scale_res(self, image, patch_num):
        b, c, h, w = image.shape

        f_clone = image.detach().clone()
        assert (
            patch_num[-1] == h and patch_num[-1] == w
        ), f"{patch_num[-1]=} != ({h=}, {w=})"

        img_scale = [
            F.interpolate(f_clone, size=(size, size), mode="area") for size in patch_num
        ]

        img_scale_res = []
        img_scale_res.append(img_scale[0])

        img_res = img_scale[1] - F.interpolate(
            img_scale[0], size=(patch_num[1], patch_num[1]), mode="bilinear"
        )
        img_scale_res.append(img_res)

        img_res = img_scale[2] - F.interpolate(
            img_scale[1], size=(patch_num[2], patch_num[2]), mode="bilinear"
        )
        img_scale_res.append(img_res)

        return img_scale_res

    def scale_channel(self, img_scale):
        b = img_scale[0].shape[0]
        c = img_scale[0].shape[1]
        ca_list = [self.CA(img).view(b, c, -1) for img in img_scale]
        combined_feature = torch.cat(ca_list, dim=2)
        return combined_feature

    @staticmethod
    def multiply_ca_scale(c_attention, img_scale):
        l = c_attention.shape[2]
        c_data = (
            c_attention.permute(0, 2, 1).unsqueeze(3).unsqueeze(4)
        )  # 先转置为 (B, L, C)，然后增加两个维度

        ca_scale = [
            c_data[:, i, :, :, :] * img_scale[i] + img_scale[i] for i in range(l)
        ]
        return ca_scale

    def scale_to_img_channel(self, ca_scale, patch_num):
        b, c, h, w = ca_scale[-1].shape
        result = ca_scale

        img_res = (
            F.interpolate(result[0], size=(patch_num[1], patch_num[1]), mode="bilinear")
            + result[1]
        )
        img_res = (
            F.interpolate(img_res, size=(patch_num[2], patch_num[2]), mode="bilinear")
            + result[2]
        )
        result_img = self.convEnd(img_res)
        return result_img

    # new  scan channel
    def forward_core(self, x):
        patch_num = self.patch_num
        B, C, H, W = x.shape

        Scale = self.img_to_scale_res(x, patch_num)

        xsc = self.scale_channel(Scale)

        xsc = xsc.permute(0, 2, 1).contiguous()

        L = len(patch_num)

        x_dbl = self.x_proj(rearrange(xsc, "b d l -> (b l) d"))  # (bl d)
        dts, Bs, Cs = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dts = self.dt_proj.weight @ dts.t()

        xsc = xsc.float().view(B, L, -1)
        dts = dts.contiguous().float().view(B, L, -1)
        Bs = Bs.float().contiguous().view(B, -1, self.d_inner)
        Cs = Cs.float().contiguous().view(B, -1, self.d_inner)
        Ds = self.Ds.contiguous().float().view(-1)
        As = -torch.exp(self.A_log.float())
        dt_projs_bias = self.dt_proj.bias.float().view(-1)
        out_y = self.selective_scan(
            xsc,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, L, -1)

        out_y = out_y.permute(0, 2, 1).contiguous()
        ca_scale = self.multiply_ca_scale(out_y, Scale)
        out = self.scale_to_img_channel(ca_scale, patch_num)
        return out

    def forward(self, x):
        B, C, H, W = x.shape
        x1 = x
        x = x.permute(0, 2, 3, 1).contiguous()
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y = self.forward_core(x)
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        out = out.permute(0, 3, 1, 2).contiguous()
        return out + x1


# 多尺度扫描，空间注意力1
class SsmSpatial(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        exhr_msid=2.0,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.0,
        conv_bias=True,
        bias=False,
        image_size=64,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.exhr_msid = exhr_msid
        self.d_inner = int(self.exhr_msid * self.d_model)
        self.up_size = image_size
        self.ds_inner = self.up_size * self.up_size
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=3,
            padding=1,
            **factory_kwargs,
        )
        self.act = nn.SiLU()
        self.xs_proj = nn.Linear(
            3, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dts_proj = nn.Linear(self.dt_rank, 3, bias=True, **factory_kwargs)

        # S4D real initialization
        As = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=3,
        ).contiguous()
        As_log = torch.log(As)  # Keep A_log in fp32
        self.As_log = nn.Parameter(As_log)
        self.As_log._no_weight_decay = True

        self.Dss = nn.Parameter(torch.ones(3, device=device))  # Keep in fp32
        self.Dss._no_weight_decay = True

        self.selective_scan = selective_scan_fn
        # self.selective_scan = selective_scan_ref

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        self.patch_num = (image_size // 16, image_size // 4, image_size)
        self.convEnd = nn.Conv2d(
            self.d_model * 2 * len(self.patch_num),
            self.d_model * 2,
            kernel_size=1,
            device=device,
        )
        self.SA = SpatialAttention(self.up_size)

    def img_to_scale(self, image, patch_num):
        b, c, h, w = image.shape

        f_clone = image.detach().clone()
        assert (
            patch_num[-1] == h and patch_num[-1] == w
        ), f"{patch_num[-1]=} != ({h=}, {w=})"

        img_scale = [
            F.interpolate(f_clone, size=(size, size), mode="area") for size in patch_num
        ]
        return img_scale

    def scale_spatial(self, img_scale):
        sa_list = [self.SA(img) for img in img_scale]
        combined_feature = torch.cat(sa_list, dim=2)
        # 输出 B 6400 L
        return combined_feature

    def scale_to_img_spatial(self, sa_scale):
        B, C, H, W = sa_scale[-1].shape
        # result = [self.conv2(ca) for ca in sa_scale]  #
        result = sa_scale
        img_up = [F.interpolate(img, size=(H, W), mode="bilinear") for img in result]
        img_m = torch.cat(img_up, dim=1)
        result_img = self.convEnd(img_m)
        return result_img

    @staticmethod
    def multiply_sa_scale(s_attention, img_scale, patch_num):

        B = s_attention.shape[0]
        L = s_attention.shape[2]
        hw = s_attention.shape[1]
        h = int(math.sqrt(hw))
        sa_scale = [
            F.interpolate(
                s_attention[:, :, i].view(B, 1, h, h),
                size=(patch_num[i], patch_num[i]),
                mode="area",
            )
            * img_scale[i]
            + img_scale[i]
            for i in range(L)
        ]
        return sa_scale

    def forward_core(self, x: torch.Tensor):
        patch_num = self.patch_num
        B, C, H, W = x.shape
        scale = self.img_to_scale(x, patch_num)
        xss = self.scale_spatial(scale)

        xss = xss.permute(0, 2, 1).contiguous()
        L = len(patch_num)
        xs_dbl = self.xs_proj(rearrange(xss, "b d l -> (b l) d"))
        dtss, Bss, Css = torch.split(
            xs_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dtss = self.dts_proj.weight @ dtss.t()

        xss = xss.float().view(B, L, -1)
        dtss = dtss.contiguous().float().view(B, L, -1)  # (b, k * d, l)
        Bss = Bss.float().contiguous().view(B, -1, self.ds_inner)
        Css = Css.float().contiguous().view(B, -1, self.ds_inner)  # (b, k, d_state, l)
        Dss = self.Dss.contiguous().float().view(-1)
        Ass = -torch.exp(self.As_log.float())  # (d_inner, d_state)
        dts_projs_bias = self.dts_proj.bias.float().view(-1)  # (k * d)

        out_ys = self.selective_scan(
            xss,
            dtss,
            Ass,
            Bss,
            Css,
            Dss,
            z=None,
            delta_bias=dts_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, -1, L)

        sa_scale = self.multiply_sa_scale(out_ys, scale, patch_num)

        out_y = self.scale_to_img_spatial(sa_scale)

        return out_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, C, H, W = x.shape
        x1 = x
        x = x.permute(0, 2, 3, 1).contiguous()
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y = self.forward_core(x)

        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        out = out.permute(0, 3, 1, 2).contiguous()
        return out + x1


class ChannelMambaBlock(nn.Module):
    def __init__(self, dim, image_size=64):
        super(ChannelMambaBlock, self).__init__()
        self.encoder = SsmChannel(dim, image_size=image_size)

    def forward(self, x):
        return self.encoder(x)


class SpatialMambaBlock(nn.Module):
    def __init__(self, dim, image_size=64):
        super(SpatialMambaBlock, self).__init__()
        self.encoder = SsmSpatial(dim, image_size=image_size)

    def forward(self, x):
        return self.encoder(x)


class SingleMambaBlock(nn.Module):
    def __init__(self, dim):
        super(SingleMambaBlock, self).__init__()
        self.encoder = Mamba(dim, bimamba_type=None)
        self.norm = LayerNorm(dim, "with_bias")

    def forward(self, x):
        x = self.norm(x)
        y = self.encoder(x) + x
        return y


# 交互模块
class ScaleCrossMamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        exhr_msid=2.0,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.3,
        conv_bias=True,
        bias=False,
        image_size=64,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.exhr_msid = exhr_msid
        self.d_inner = int(self.exhr_msid * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_norm = nn.LayerNorm(self.d_model)
        self.in_proj = nn.Linear(
            self.d_model * 3, self.d_inner * 2, bias=bias, **factory_kwargs
        )
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=3,
            padding=1,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.xs_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dts_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True, **factory_kwargs
        )

        # S4D real initialization
        As = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        As_log = torch.log(As)
        self.As_log = nn.Parameter(As_log)
        self.As_log._no_weight_decay = True

        self.Dss = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.Dss._no_weight_decay = True

        self.selective_scan = selective_scan_fn

        self.patch_num = (image_size // 16, image_size // 4, image_size)

        self.k_weight1 = nn.Parameter(torch.ones(4) / 4, requires_grad=True)
        self.k_weight2 = nn.Parameter(torch.ones(4) / 4, requires_grad=True)
        self.k_weight3 = nn.Parameter(torch.ones(4) / 4, requires_grad=True)
        self.k_weight_list = [self.k_weight1, self.k_weight2, self.k_weight3]

        self.conv_h = nn.Conv2d(
            self.d_model, self.d_model, 3, stride=1, padding=1, device=device
        )
        self.conv_m = nn.Conv2d(
            self.d_model, self.d_model, 3, stride=1, padding=1, device=device
        )

        self.conv_0 = nn.Conv2d(
            self.d_model, self.d_model, 1, stride=1, padding=0, device=device
        )
        self.conv_1 = nn.Conv2d(
            self.d_model, self.d_model, 1, stride=1, padding=0, device=device
        )
        self.out_norm = nn.LayerNorm(self.d_model * 2)
        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def img_to_scale(self, hr_msi, lr_hsi, patch_num):

        B, C, H, W = hr_msi.shape
        hr_msi = self.conv_m(hr_msi)
        f_hr_msi = hr_msi.detach().clone()

        lr_hsi = self.conv_h(lr_hsi)
        f_lr_hsi = lr_hsi.detach().clone()

        assert (
            patch_num[-1] == H and patch_num[-1] == W
        ), f"{patch_num[-1]=} != ({H=}, {W=})"

        hr_msi_scale = [
            self.conv_0(F.interpolate(f_hr_msi, size=(size, size), mode="area"))
            for size in patch_num
        ]
        lr_hsi_scale = [
            self.conv_1(F.interpolate(f_lr_hsi, size=(size, size), mode="area"))
            for size in patch_num
        ]

        return hr_msi_scale, lr_hsi_scale

    def forward(self, lr_hsi, hr_msi):
        patch_num = self.patch_num
        K = 3
        h = hr_msi
        l = lr_hsi
        scale_hr_msi, scale_lr_hsi = self.img_to_scale(hr_msi, lr_hsi, patch_num)

        fusion_img = scale_lr_hsi[0]

        for index in range(len(patch_num)):
            hr_msi_f = scale_hr_msi[index]
            lr_hsi_f = scale_lr_hsi[index]
            B, C, H, W = hr_msi_f.shape
            L = H * W

            xs11 = (
                torch.cat((hr_msi_f, fusion_img, lr_hsi_f), dim=1)
                .permute(0, 2, 3, 1)
                .contiguous()
            )

            xz = self.in_proj(xs11)
            xs, z = xz.chunk(2, dim=-1)
            # B C H W
            xs1 = xs.permute(0, 3, 1, 2).contiguous()

            xss = self.act(self.conv2d(xs1)).view(B, -1, L).contiguous()

            xs_dbl = self.xs_proj(rearrange(xss, "b d l -> (b l) d"))
            dtss, Bss, Css = torch.split(
                xs_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
            )
            dtss = self.dts_proj.weight @ dtss.t()

            xss = xss.float().view(B, -1, L)
            dtss = dtss.contiguous().float().view(B, -1, L)
            Bss = Bss.float().contiguous().view(B, -1, L)
            Css = Css.float().contiguous().view(B, -1, L)
            Dss = self.Dss.contiguous().float().view(-1)
            Ass = -torch.exp(self.As_log.float())
            dts_projs_bias = self.dts_proj.bias.float().view(-1)

            out_y = self.selective_scan(
                xss,
                dtss,
                Ass,
                Bss,
                Css,
                Dss,
                z=None,
                delta_bias=dts_projs_bias,
                delta_softplus=True,
                return_last_state=False,
            ).view(B, -1, L)

            assert out_y.dtype == torch.float

            out_y = self.out_norm(out_y.permute(0, 2, 1))

            out_y = out_y.view(B, H, W, -1).contiguous() * F.silu(z)

            out_y = self.out_proj(out_y).permute(0, 3, 1, 2).contiguous()

            img_list = [out_y, hr_msi_f, fusion_img, lr_hsi_f]

            k_weight_list = [torch.softmax(i, dim=0) for i in self.k_weight_list]
            fusion_img = sum(w * out for w, out in zip(k_weight_list[index], img_list))

            if index != len(patch_num) - 1:
                fusion_img = F.interpolate(fusion_img, size=(4 * H, 4 * W), mode="area")

        if self.dropout is not None:
            fusion_img = self.dropout(fusion_img)
        fusion_img = fusion_img + l + h

        return fusion_img


class CrossMamba(nn.Module):
    def __init__(self, dim, image_size=64):
        super(CrossMamba, self).__init__()
        self.cross_mamba = ScaleCrossMamba(dim, image_size=image_size)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, lr_hsi, hr_msi):
        f = self.cross_mamba(lr_hsi, hr_msi)
        hr_hsi = self.dwconv(f)
        return hr_hsi


class CrossMamba1(nn.Module):
    def __init__(self, dim):
        super(CrossMamba1, self).__init__()
        self.cross_mamba = Mamba(dim, bimamba_type="v3")
        self.norm1 = LayerNorm(dim, "with_bias")
        self.norm2 = LayerNorm(dim, "with_bias")
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, ms, ms_resi, pan):
        ms_resi = ms + ms_resi
        ms = self.norm1(ms_resi)
        pan = self.norm2(pan)
        global_f = self.cross_mamba(self.norm1(ms), extra_emb=self.norm2(pan))
        B, HW, C = global_f.shape
        ms = global_f.transpose(1, 2).view(B, C, 64, 64)
        ms = self.dwconv(ms) + ms
        return ms, ms_resi


class Refine(nn.Module):

    def __init__(self, n_feat, out_channel):
        super(Refine, self).__init__()

        self.conv_in = nn.Conv2d(n_feat, n_feat, 3, stride=1, padding=1)
        self.process = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_feat, n_feat, 3, stride=1, padding=1),
        )
        self.attention = ChannelAttention(n_feat, 4)
        self.conv_last = nn.Conv2d(
            in_channels=n_feat,
            out_channels=out_channel,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        out1 = self.conv_in(x)
        out = self.process(out1)
        out = self.attention(out) * out + out1
        out = self.conv_last(out)
        return out


# 尺度为三
class Net(nn.Module):
    def __init__(
        self,
        hsi_dim,
        msi_dim,
        scale=4,
        base_filter=64,
        image_size=64,
    ):
        super(Net, self).__init__()
        base_filter = 64
        self.base_filter = base_filter
        self.stride = 1
        self.patch_size = 1
        self.scale = scale

        self.hr_msi_encoder = nn.Conv2d(msi_dim, base_filter, 3, 1, 1)
        self.lr_hsi_encoder = nn.Conv2d(hsi_dim, base_filter, 3, 1, 1)

        self.hr_msi_encoder1 = HinResBlock(base_filter, base_filter)
        self.lr_hsi_encoder1 = HinResBlock(base_filter, base_filter)

        self.embed_dim = base_filter * self.stride * self.patch_size

        self.mamba_scale_c = ChannelMambaBlock(self.embed_dim, image_size=image_size)

        self.mamba_scale_s = SpatialMambaBlock(self.embed_dim, image_size=image_size)

        self.shallow_fusion1 = nn.Conv2d(base_filter * 2, base_filter, 3, 1, 1)
        self.shallow_fusion2 = nn.Conv2d(base_filter * 2, base_filter, 3, 1, 1)

        self.conv_lr1 = nn.Conv2d(base_filter, base_filter, 1, 1)
        self.conv_hr1 = nn.Conv2d(base_filter, base_filter, 1, 1)

        self.cross_fusion1 = CrossMamba(self.embed_dim, image_size=image_size)

        self.output = Refine(base_filter, hsi_dim)

    def forward(self, lr_hsi, hr_msi):

        lr_hsi_bic = F.interpolate(lr_hsi, scale_factor=self.scale)

        # 嵌入层
        lr_hsi_f1 = self.lr_hsi_encoder(lr_hsi_bic)
        hr_msi_f1 = self.hr_msi_encoder(hr_msi)

        lr_hsi_f = self.lr_hsi_encoder1(lr_hsi_f1)
        hr_msi_f = self.hr_msi_encoder1(hr_msi_f1)
        lr_hsi_f = self.lr_hsi_encoder1(lr_hsi_f)
        hr_msi_f = self.hr_msi_encoder1(hr_msi_f)
        lr_hsi_f = self.lr_hsi_encoder1(lr_hsi_f)
        hr_msi_f = self.hr_msi_encoder1(hr_msi_f)
        lr_hsi_f = self.lr_hsi_encoder1(lr_hsi_f)
        hr_msi_f = self.hr_msi_encoder1(hr_msi_f)

        lr_hsi_f_o = lr_hsi_f
        hr_msi_f_o = hr_msi_f

        lr_hsi_f = self.mamba_scale_c(lr_hsi_f)
        hr_msi_f = self.mamba_scale_s(hr_msi_f)

        lr_hsi_f = self.mamba_scale_c(lr_hsi_f)
        hr_msi_f = self.mamba_scale_s(hr_msi_f)

        lr_hsi_f = self.mamba_scale_c(lr_hsi_f)
        hr_msi_f = self.mamba_scale_s(hr_msi_f)

        lr_hsi_f = self.mamba_scale_c(lr_hsi_f)
        hr_msi_f = self.mamba_scale_s(hr_msi_f)

        lr_hsi_f = self.mamba_scale_c(lr_hsi_f)
        hr_msi_f = self.mamba_scale_s(hr_msi_f)

        lr_hsi_f = lr_hsi_f_o + lr_hsi_f
        hr_msi_f = hr_msi_f_o + hr_msi_f

        lr_hsi_f = (
            self.shallow_fusion1(torch.concat([lr_hsi_f, hr_msi_f], dim=1)) + lr_hsi_f1
        )
        hr_msi_f = (
            self.shallow_fusion2(torch.concat([hr_msi_f, lr_hsi_f], dim=1)) + hr_msi_f1
        )

        lr_hsi_f = self.conv_lr1(lr_hsi_f)
        hr_msi_f = self.conv_hr1(hr_msi_f)

        hr_hsi_f = self.cross_fusion1(lr_hsi_f, hr_msi_f)

        hr_hsi = self.output(hr_hsi_f) + lr_hsi_bic
        return hr_hsi


if __name__ == "__main__":
    # 将输入传入模型
    shape = (1, 31, 8, 8)
    lr_hsi = torch.randn(*shape)
    shape2 = (1, 3, 64, 64)
    hr_msi = torch.randn(*shape2)

    model = Net().to(device)

    output = model(lr_hsi.to(device), hr_msi.to(device))
    flops, params = profile(model, (lr_hsi.to(device), hr_msi.to(device)))
    print("flops: ", flops, "params: ", params)
    print("flops: %.2f M, params: %.2f M" % (flops / 1000000.0, params / 1000000.0))
    print(f"FLOPs: {flops / 1e9:.4f} G")
    print(f"Params: {params / 1e6:.4f} M")
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: % .4fM" % (total / 1e6))
    print(f"flops: {flops / 1e9:.4f} G", "parameter: % .4fM" % (total / 1e6))
