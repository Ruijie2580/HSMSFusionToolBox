"""
Paper: Correlation-aware learning self-guiding network for hyperspectral fusion imaging.
DOI: https://doi.org/10.1016/j.inffus.2026.104263
Repository: https://github.com/HengYang01/CLSNet
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out


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


def shift_back(inputs, step=2):  # input [bs,28,256,310]  output [bs, 28, 256, 256]
    [bs, nC, row, col] = inputs.shape
    down_sample = 256 // row
    step = float(step) / float(down_sample * down_sample)
    out_col = row
    for i in range(nC):
        inputs[:, i, :, :out_col] = inputs[
            :, i, :, int(step * i) : int(step * i) + out_col
        ]
    return inputs[:, :, :, :out_col]


class MS_MSA(nn.Module):
    def __init__(
        self,
        dim,
        dim_head,
        heads,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in, y_in, f_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        y = y_in.reshape(b, h * w, c)
        f = f_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(y)
        v_inp = self.to_v(f)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads),
            (q_inp, k_inp, v_inp),
        )
        v = v
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = k @ q.transpose(-2, -1)  # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v  # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)  # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)).permute(
            0, 2, 3, 1
        )
        out = out_c + out_p

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
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


class MSAB1(nn.Module):
    def __init__(
        self,
        dim,
        dim_head,
        heads,
        num_blocks,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(
                nn.ModuleList(
                    [
                        MS_MSA(dim=dim, dim_head=dim_head, heads=heads),
                        PreNorm(dim, FeedForward(dim=dim)),
                    ]
                )
            )

    def forward(self, x, y, z):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        for attn, ff in self.blocks:
            x = attn(x, y, z) + x
        out = x.permute(0, 3, 1, 2)
        return out


class MSAB2(nn.Module):
    def __init__(
        self,
        dim,
        dim_head,
        heads,
        num_blocks,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(
                nn.ModuleList(
                    [
                        MS_MSA(dim=dim, dim_head=dim_head, heads=heads),
                        PreNorm(dim, FeedForward(dim=dim)),
                    ]
                )
            )

    def forward(self, x, y, z):
        for attn, ff in self.blocks:
            x = attn(x, y, z) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out


class HIGF(nn.Module):
    def __init__(
        self,
        dim_num,
        heads,
    ):
        super().__init__()
        self.MSAB_1 = MSAB1(
            dim=dim_num, num_blocks=1, dim_head=heads, heads=dim_num // heads
        )
        self.MSAB_2 = MSAB1(
            dim=dim_num, num_blocks=1, dim_head=heads, heads=dim_num // heads
        )
        self.LIGI_1 = MSAB1(
            dim=dim_num, num_blocks=1, dim_head=heads, heads=dim_num // heads
        )
        self.LIGI_2 = MSAB1(
            dim=dim_num, num_blocks=1, dim_head=heads, heads=dim_num // heads
        )
        self.GIGI_1 = MSAB2(
            dim=dim_num, num_blocks=1, dim_head=heads, heads=dim_num // heads
        )
        self.GIGI_2 = MSAB2(
            dim=dim_num, num_blocks=1, dim_head=heads, heads=dim_num // heads
        )
        self.outc = nn.Conv2d(dim_num * 2, dim_num, kernel_size=1)
        self.fuc1 = nn.Conv2d(dim_num * 2, dim_num, kernel_size=1)
        self.fuc2 = nn.Conv2d(dim_num * 2, dim_num, kernel_size=1)
        self.final = nn.Conv2d(dim_num * 2, dim_num, kernel_size=3, padding=1)

    def forward(self, x, y, z):
        zin = z
        x = x.permute(0, 2, 3, 1)
        y = y.permute(0, 2, 3, 1)
        z = z.permute(0, 2, 3, 1)
        xa = self.MSAB_1(x, x, x).permute(0, 2, 3, 1)
        yb = self.MSAB_2(y, y, y).permute(0, 2, 3, 1)
        fa = self.LIGI_1(x, z, z).permute(0, 2, 3, 1)
        fb = self.LIGI_2(y, z, z).permute(0, 2, 3, 1)

        fa = self.GIGI_1(xa, fa, fa)
        fa = torch.cat((fa, xa.permute(0, 3, 1, 2)), dim=1)
        fa = self.fuc1(fa)

        fb = self.GIGI_2(yb, fb, fb)
        fb = torch.cat((fb, yb.permute(0, 3, 1, 2)), dim=1)
        fb = self.fuc2(fb)

        ff = torch.cat((fa, fb), dim=1)
        out = self.outc(ff)
        out = torch.cat((out, zin), dim=1)
        out = self.final(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, channel):
        super(ResBlock, self).__init__()
        self.channel = channel
        self.layers = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.channel, self.channel, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, x):
        out = self.layers(x)
        out = out + x
        return out


class SCSC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SCSC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 3, 1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 3, 1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.down = nn.Conv2d(2 * self.in_channels, self.in_channels, 3, 1, padding=1)
        self.out = nn.Conv2d(self.in_channels, self.out_channels, 3, 1, padding=1)
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(1)
        squeeze_factor = 6
        self.linear1 = nn.Linear(1, in_channels // squeeze_factor)
        self.linear2 = nn.Linear(in_channels, in_channels // squeeze_factor)
        self.linear3 = nn.Linear(in_channels // squeeze_factor, 1)
        self.low_dim = in_channels // squeeze_factor

    def forward(self, xin, yin):
        x2 = self.pool1(xin).squeeze(-1)
        x2 = self.linear1(x2)
        y2 = self.pool2(yin).squeeze(-1)
        y2 = y2.transpose(1, 2)
        y2 = self.linear2(y2)
        f1 = x2 * y2
        f2 = self.linear3(f1).unsqueeze(2)
        f_out = yin * f2
        out = torch.cat((f_out, yin), dim=1)
        out = self.down(out)
        out = out + xin
        return out


class Block(nn.Module):
    def __init__(self, dim=62, stage=3):
        super(Block, self).__init__()
        self.dim = dim
        self.stage = stage

        self.SCSC_1 = SCSC(dim, dim)
        self.SCSC_2 = SCSC(dim, dim)
        self.SCSC_3 = SCSC(dim, dim)
        self.SCSC_4 = SCSC(dim, dim)

        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avgpool3 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.fea_extrax1 = nn.Conv2d(dim, dim, 3, 1, padding=1)
        self.fea_extray1 = nn.Conv2d(dim, dim, 3, 1, padding=1)
        self.fea_extrax2 = nn.Conv2d(dim, dim, 3, 1, padding=1)
        self.fea_extray2 = nn.Conv2d(dim, dim, 3, 1, padding=1)
        self.fea_extrax3 = nn.Conv2d(dim, dim, 3, 1, padding=1)
        self.fea_extray3 = nn.Conv2d(dim, dim, 3, 1, padding=1)

        self.HIGF_1 = HIGF(dim_num=dim, heads=31)
        self.HIGF_2 = HIGF(dim_num=dim, heads=31)
        self.HIGF_3 = HIGF(dim_num=dim, heads=31)

        self.FeaDownSample1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.FeaDownSample2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.FeaDownSample3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.FeaDownSample4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.FeaDownSample5 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.FeaDownSample6 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.FeaDownSample7 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.FeaDownSample8 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.FeaDownSample9 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.fea_extrax4 = nn.Conv2d(dim, dim, 3, 1, padding=1)
        self.fea_extray4 = nn.Conv2d(dim, dim, 3, 1, padding=1)
        self.HIGF_4 = HIGF(dim_num=dim, heads=31)

        self.FeaUpSample1 = nn.ConvTranspose2d(
            dim, dim, stride=2, kernel_size=2, padding=0, output_padding=0
        )
        self.FeaUpSample2 = nn.ConvTranspose2d(
            dim, dim, stride=2, kernel_size=2, padding=0, output_padding=0
        )
        self.FeaUpSample3 = nn.ConvTranspose2d(
            dim, dim, stride=2, kernel_size=2, padding=0, output_padding=0
        )

        self.Fution1 = nn.Conv2d(dim * 2, dim, 1, 1, bias=False)
        self.Fution2 = nn.Conv2d(dim * 2, dim, 1, 1, bias=False)
        self.Fution3 = nn.Conv2d(dim * 2, dim, 1, 1, bias=False)

        self.HIGF_5 = HIGF(dim_num=dim, heads=31)
        self.HIGF_6 = HIGF(dim_num=dim, heads=31)
        self.HIGF_7 = HIGF(dim_num=dim, heads=31)

        self.deres1 = ResBlock(dim)
        self.deres2 = ResBlock(dim)
        self.deres3 = ResBlock(dim)

        self.mapping = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x_in, y_in, fea_in):
        x = x_in
        y1 = y_in
        fea = fea_in

        fea_encoder = []
        x1 = self.fea_extrax1(x)
        spat = self.SCSC_1(x, x1)
        x1 = x + spat
        fea = self.HIGF_1(x1, y1, fea)
        fea_encoder.append(fea)
        fea = self.FeaDownSample1(fea)
        x12 = self.FeaDownSample2(x1)
        y2 = self.FeaDownSample3(y1)

        x2 = self.fea_extrax2(x12)
        spat = self.SCSC_2(x12, x2)
        x2 = x12 + spat
        fea = self.HIGF_2(x2, y2, fea)
        fea_encoder.append(fea)
        fea = self.FeaDownSample4(fea)
        x23 = self.FeaDownSample5(x2)
        y3 = self.FeaDownSample6(y2)

        x3 = self.fea_extrax3(x23)
        spat = self.SCSC_3(x23, x3)
        x3 = x23 + spat
        fea = self.HIGF_3(x3, y3, fea)
        fea_encoder.append(fea)
        fea = self.FeaDownSample7(fea)
        x34 = self.FeaDownSample8(x3)
        y4 = self.FeaDownSample9(y3)

        x4 = self.fea_extrax4(x34)
        spat = self.SCSC_4(x34, x4)
        x4 = x34 + spat
        fea = self.HIGF_4(x4, y4, fea)

        fea = self.FeaUpSample1(fea)
        fea = self.Fution1(torch.cat([fea, fea_encoder[self.stage - 1]], dim=1))
        fea = self.HIGF_5(x3, y3, fea)

        fea = self.FeaUpSample2(fea)
        fea = self.Fution2(torch.cat([fea, fea_encoder[self.stage - 2]], dim=1))
        fea = self.HIGF_6(x2, y2, fea)

        fea = self.FeaUpSample3(fea)
        fea = self.Fution3(torch.cat([fea, fea_encoder[self.stage - 3]], dim=1))
        fea = self.HIGF_7(x1, y1, fea)

        out = self.mapping(fea)
        return out


class CLSNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=31, scale=4, n_feat=62, stage=3):
        super(CLSNet, self).__init__()
        self.stage = stage
        self.conv_in_x = nn.Conv2d(
            in_channels, n_feat, kernel_size=3, padding=(3 - 1) // 2, bias=False
        )
        self.conv_in_y = nn.Conv2d(
            out_channels, n_feat, kernel_size=3, padding=(3 - 1) // 2, bias=False
        )
        self.conv_in = nn.Conv2d(n_feat * 2, n_feat, kernel_size=1, bias=False)
        self.CLSNetBlock = Block(dim=n_feat, stage=3)
        self.conv_out = nn.Conv2d(
            n_feat, out_channels, kernel_size=3, padding=(3 - 1) // 2, bias=False
        )
        self.upsamp = nn.Upsample(scale_factor=scale, mode="nearest")

    def forward(self, y, x):  # LR HSI    HR MSI
        b, c, h_inp, w_inp = x.shape
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode="reflect")
        xin = self.conv_in_x(x)
        yin = self.conv_in_y(self.upsamp(y))
        xy = torch.cat((xin, yin), dim=1)
        xy = self.conv_in(xy)
        h = self.CLSNetBlock(xin, yin, xy)
        h = h + yin
        h = self.conv_out(h)
        return h[:, :, :h_inp, :w_inp]


# from thop import profile

# a = torch.rand(1, 31, 8, 8).cuda()
# b = torch.rand(1, 3, 64, 64).cuda()
# model = CLSNet().cuda()
# print(model)
# flops, params = profile(model, inputs=(a, b))
# print(f"FLOPs: {flops / 1e9:.4f} G")  # 转换为GFLOPs
# print(f"Params: {params / 1e6:.4f} M")  # 转换为MParams
