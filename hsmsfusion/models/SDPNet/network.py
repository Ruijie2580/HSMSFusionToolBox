import math
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=1, bias=bias, stride=stride
    )


def dwt_init(x):
    # print(x.shape)
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = (
        in_batch,
        int(in_channel / (r**2)),
        r * in_height,
        r * in_width,
    )
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel : out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2 : out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3 : out_channel * 4, :, :] / 2
    h = torch.zeros([out_batch, out_channel, out_height, out_width])

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = True

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = True

    def forward(self, x):

        return iwt_init(x)


class WSA(nn.Module):
    def __init__(self, kernel_size=5, bias=False):
        super(WSA, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(
                2,
                1,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                bias=bias,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # torch.max will output 2 things, and we want the 1st one
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, 1, keepdim=True)
        channel_pool = torch.cat(
            [max_pool, avg_pool], dim=1
        )  # [N,2,H,W]  could add 1x1 conv -> [N,3,H,W]
        y = self.conv_du(channel_pool)
        return x * y


class WCA(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(WCA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel * 4, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel * 4, 1, padding=0, bias=bias),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class SSWR_HSI(nn.Module):
    def __init__(
        self, n_feat, o_feat, kernel_size=3, reduction=16, bias=False, act=nn.PReLU()
    ):
        super(SSWR_HSI, self).__init__()

        self.dwt = DWT().to(device)
        self.iwt = IWT().to(device)

        modules_body = [
            conv(n_feat * 4, n_feat, kernel_size, bias=bias).to(device),
            act,
            conv(n_feat, n_feat * 2, kernel_size, bias=bias).to(device),
        ]
        self.body = nn.Sequential(*modules_body).to(device)

        self.WSA = WSA().to(device)
        self.WCA = WCA(n_feat, reduction, bias=bias).to(device)

        self.conv1x1 = nn.Conv2d(n_feat * 2, n_feat * 4, kernel_size=1, bias=bias).to(
            device
        )
        self.conv3x3 = nn.Conv2d(
            n_feat, o_feat, kernel_size=3, padding=1, bias=bias
        ).to(device)
        self.activate = act
        self.conv1x1_final = nn.Conv2d(n_feat, o_feat, kernel_size=1, bias=bias).to(
            device
        )

    def forward(self, x):
        residual = x
        _, _, h, w = x.shape
        pad_h = (2 - h % 2) % 2
        pad_w = (2 - w % 2) % 2
        if h > 1 and w > 1:
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        wavelet_path_in = x
        x_dwt = self.dwt(wavelet_path_in)
        res = self.WCA(x_dwt)
        res = res + x_dwt
        wavelet_path = self.iwt(res)
        wavelet_path = wavelet_path[:, :, :h, :w]
        wavelet_path = wavelet_path[:, :, :, :w]
        wavelet_path = wavelet_path.to(self.conv3x3.weight.device)
        out = self.activate(self.conv3x3(wavelet_path))
        out += self.conv1x1_final(residual)
        return out


class SSWR_MSI(nn.Module):
    def __init__(
        self, n_feat, o_feat, kernel_size=3, reduction=16, bias=False, act=nn.PReLU()
    ):
        super(SSWR_MSI, self).__init__()

        self.dwt = DWT().to(device)
        self.iwt = IWT().to(device)

        modules_body = [
            conv(n_feat * 4, n_feat, kernel_size, bias=bias).to(device),
            act,
            conv(n_feat, n_feat * 2, kernel_size, bias=bias).to(device),
        ]
        self.body = nn.Sequential(*modules_body).to(device)

        self.WSA = WSA().to(device)
        self.WCA = WCA(n_feat * 2, reduction, bias=bias).to(device)

        self.conv1x1 = nn.Conv2d(n_feat * 2, n_feat * 4, kernel_size=1, bias=bias).to(
            device
        )
        self.conv3x3 = nn.Conv2d(
            n_feat, o_feat, kernel_size=3, padding=1, bias=bias
        ).to(device)
        self.activate = act
        self.conv1x1_final = nn.Conv2d(n_feat, o_feat, kernel_size=1, bias=bias).to(
            device
        )

    def forward(self, x):
        x = x.to(device)
        residual = x
        _, _, h, w = x.shape
        pad_h = (2 - h % 2) % 2
        pad_w = (2 - w % 2) % 2
        x_padded = nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        x_dwt = self.dwt(x_padded)
        res = self.WSA(x_dwt)
        res = res + x_dwt
        wavelet_path = self.iwt(res)
        wavelet_path = wavelet_path.to(x.device)
        wavelet_path = wavelet_path[:, :, :h, :w]
        wavelet_path = wavelet_path[:, :, :, :w]
        out = self.activate(self.conv3x3(wavelet_path))
        out += self.conv1x1_final(residual)
        return out


class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2.0,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.0,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
        )
        self.x_proj_weight = nn.Parameter(
            torch.stack([t.weight for t in self.x_proj], dim=0)
        )  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
        )
        self.dt_projs_weight = nn.Parameter(
            torch.stack([t.weight for t in self.dt_projs], dim=0)
        )  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(
            torch.stack([t.bias for t in self.dt_projs], dim=0)
        )  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(
            self.d_state, self.d_inner, copies=4, merge=True
        )  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    @staticmethod
    def dt_init(
        dt_rank,
        d_inner,
        dt_scale=1.0,
        dt_init="random",
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        **factory_kwargs,
    ):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack(
            [
                x.view(B, -1, L),
                torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L),
            ],
            dim=1,
        ).view(B, 2, -1, L)
        xs = torch.cat(
            [x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1
        )  # (1, 4, 192, 3136)

        x_dbl = torch.einsum(
            "b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight
        )
        dts, Bs, Cs = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2
        )
        dts = torch.einsum(
            "b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight
        )
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        out_y = self.selective_scan(
            xs,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = (
            torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3)
            .contiguous()
            .view(B, -1, L)
        )
        invwh_y = (
            torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3)
            .contiguous()
            .view(B, -1, L)
        )

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class SAGM(nn.Module):
    def __init__(self, channels, d_state=16, save_dir="attention_data"):
        super().__init__()
        self.msi_attn = SS2D(d_model=channels, d_state=d_state)
        self.proj = nn.Conv2d(channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.save_dir = save_dir
        # os.makedirs(self.save_dir, exist_ok=True)  # 确保目录存在

    def forward(
        self, F_HSI, F_MSI, save_filename=None, save_epoch=None, current_epoch=None
    ):
        B, C, H_m, W_m = F_MSI.shape
        F_HSI_up = F.interpolate(
            F_HSI, size=(H_m, W_m), mode="bicubic", align_corners=False
        )
        F_MSI_attn = self.msi_attn(F_MSI.permute(0, 2, 3, 1)).permute(
            0, 3, 1, 2
        )  # 维度转换
        attention_map = self.sigmoid(self.proj(F_MSI_attn))
        self.saved_attention_map = (
            attention_map.detach().cpu().numpy()
        )  # 只存储 NumPy 格式的数据
        enhanced_HSI = F_HSI_up * attention_map + F_MSI * (1 - attention_map)

        # if save_filename is not None and current_epoch is not None and current_epoch % 998 == 0:
        #     save_path = os.path.join(self.save_dir, f"{save_filename}_epoch{current_epoch}.npy")
        #     np.save(save_path, self.saved_attention_map)

        return enhanced_HSI


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, -1, 1, 1) * y + bias.view(1, -1, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, -1, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return (
            gx,
            (grad_output * y).sum(dim=(0, 2, 3)),
            grad_output.sum(dim=(0, 2, 3)),
            None,
        )


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class FGU(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * torch.tanh(x2)


class DMlp(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim * 2, 3, 1, 1, groups=dim),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 1, 1, 0),
        )
        self.sg = FGU()  # 替代 GELU
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv_0(x)  # [B, 2H, H, W]
        x = self.sg(x)  # [B, H, H, W]
        x = self.conv_1(x)  # [B, C, H, W]
        return x


class Path(nn.Module):
    def __init__(self, dim=36):
        super(Path, self).__init__()
        self.linear_0 = nn.Conv2d(dim, dim * 2, 1, 1, 0)
        self.linear_1 = nn.Conv2d(
            dim, dim * 2, 1, 1, 0
        )  # 注意这里输出变成2×C，用于SimpleGate
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)

        self.lde = DMlp(dim, 2)

        self.dw_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.sg = FGU()  # 使用 FGU 替代 GELU
        self.down_scale = 8

        self.alpha = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.belt = nn.Parameter(torch.zeros((1, dim, 1, 1)))

    def forward(self, f):
        _, _, h, w = f.shape
        out = self.linear_0(f)
        y = out[:, ::2, :, :]
        x = out[:, 1::2, :, :]

        x_s = self.dw_conv(
            F.adaptive_max_pool2d(x, (h // self.down_scale, w // self.down_scale))
        )
        x_v = torch.var(x, dim=(-2, -1), keepdim=True)

        att = self.linear_1(
            x_s * self.alpha + x_v * self.belt
        )  # shape = [B, 2C, H', W']
        att = self.sg(att)
        att = F.interpolate(att, size=(h, w), mode="bicubic")
        x_l = x * att
        y_d = self.lde(y)
        return x_l + y_d


class DPFM(nn.Module):
    def __init__(
        self,
        n_bands,
        n_select_bands,
        out_channels=96,
        DW_Expand=2,
        FFN_Expand=2,
        drop_out_rate=0.0,
    ):
        super().__init__()
        in_channels = out_channels * 2
        dw_channel = in_channels * DW_Expand

        self.conv1 = nn.Conv2d(in_channels, dw_channel, kernel_size=1)
        self.conv2 = nn.Conv2d(
            dw_channel, dw_channel, kernel_size=3, padding=1, groups=dw_channel
        )
        self.conv3 = nn.Conv2d(dw_channel // 2, in_channels, kernel_size=1)
        self.path = Path(dim=dw_channel // 2)

        self.sg = FGU()

        ffn_channel = FFN_Expand * in_channels
        self.conv4 = nn.Conv2d(in_channels, ffn_channel, kernel_size=1)
        self.conv5 = nn.Conv2d(ffn_channel // 2, in_channels, kernel_size=1)

        self.norm1 = LayerNorm2d(in_channels)
        self.norm2 = LayerNorm2d(in_channels)

        self.dropout1 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )
        self.dropout2 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )

        self.beta = nn.Parameter(torch.zeros((1, in_channels, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, in_channels, 1, 1)))

        self.project_out = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, hsi_feat, msi_feat):
        hsi_feat = F.interpolate(
            hsi_feat, size=msi_feat.shape[2:], mode="bicubic", align_corners=False
        )
        inp = torch.cat([hsi_feat, msi_feat], dim=1)

        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.path(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        out = self.project_out(y)
        return out


class SuperResolution(nn.Module):
    def __init__(self, n_bands):
        super(SuperResolution, self).__init__()
        self.conv = nn.Conv2d(96, n_bands, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class DynamicGateFusion(nn.Module):
    def __init__(self, in_channels, gate_channels=None, with_spatial=True):

        super(DynamicGateFusion, self).__init__()
        self.with_spatial = with_spatial
        gate_channels = gate_channels or in_channels // 2

        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels, gate_channels, kernel_size=1), nn.Sigmoid()
        )

    def forward(self, x1, x2):
        fusion = torch.cat([x1, x2], dim=1)  # [B, 2C, H, W]
        gate = self.gate_conv(fusion)  # [B, C, H, W] or [B, C, 1, 1]

        return x1 * gate + x2 * (1 - gate)


class SDPNet(nn.Module):
    def __init__(self, n_bands, n_select_bands):
        super(SDPNet, self).__init__()
        self.SSWR_HSI = SSWR_HSI(n_bands, 96).to(device)
        self.SSWR_MSI = SSWR_MSI(n_select_bands, 96).to(device)
        self.joint_extractor = DPFM(n_bands, n_select_bands)
        self.SAGM = SAGM(channels=96).to(device)
        self.srr = SuperResolution(n_bands).to(device)
        self.gate = DynamicGateFusion(
            in_channels=192, gate_channels=96, with_spatial=True
        )

    def forward(self, lhsi, hmsi, epoch=None):
        F_HSI = self.SSWR_HSI(lhsi)
        F_MSI = self.SSWR_MSI(hmsi)

        F_joint = self.joint_extractor(F_HSI, F_MSI)

        F_fusion = self.SAGM(
            F_HSI, F_MSI, save_filename="msi_attention", current_epoch=epoch
        )
        F_final = self.gate(F_fusion, F_joint)
        output = self.srr(F_final)
        # return output, 0, 0, 0, 0, 0
        return output
