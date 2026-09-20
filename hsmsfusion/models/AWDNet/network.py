import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np
import math
from typing import Tuple

# -----------------------------
# SEM：浅层特征提取模块
# -----------------------------
class SEM(nn.Module):
    def __init__(self, in_channels, out_channels, scale=0.1):
        super(SEM, self).__init__()
        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, in_channels, kernel_size=3, padding=1),
        )
        self.scale = scale

    def forward(self, x):
        main_out = self.main_branch(x)
        out = self.scale * main_out + x
        return out


class Upsample(nn.Module):
    def __init__(self, scale_factor=2):
        super(Upsample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)

    def forward(self, x):
        return self.upsample(x)


# ============================================================================
# 常用正交/近似正交小波的分析滤波器（低通Lo_D），高通Hi_D通过QMF生成
# ============================================================================
def qmf_hi_from_lo(lo: torch.Tensor) -> torch.Tensor:
    # lo: [L], 返回 hi: [L]，hi[n] = (-1)^n * lo[L-1-n]
    L = lo.numel()
    idx = torch.arange(L - 1, -1, -1, device=lo.device, dtype=torch.long)
    # 向量化生成 (-1)^n
    n = torch.arange(L, device=lo.device, dtype=lo.dtype)
    sign = torch.cos(math.pi * n)  # cos(n*pi) == (-1)^n
    hi = sign * lo.index_select(0, idx)
    return hi

def get_lo_coeffs(name: str, device, dtype):
    name = name.lower()
    if name == "db4":
        lo = torch.tensor([
            -0.0105974017850021,  0.0328830116668852,  0.0308413818355607, -0.1870348117188811,
            -0.0279837694169839,  0.6308807679295904,  0.7148465705529154,  0.2303778133088964
        ], device=device, dtype=dtype)
    elif name == "db6":
        # Daubechies 6 (length 12)
        lo = torch.tensor([
             0.00107730108499558,  -0.00477725751101065,  -0.00055384220099380,
             0.03158203931748603,   0.02752286553030533,  -0.0975016055873225,
            -0.12976686756709563,   0.2262646939654400,    0.3152503517092432,
            -0.7511339080210959,    0.4946238903984534,    0.1115407433501095
        ], device=device, dtype=dtype)
    elif name == "sym6":
        # Symlets 6 (length 12)
        lo = torch.tensor([
            -0.007800708325034148,  0.001767711864242804,  0.04472490177066578,
            -0.02106029251230056,  -0.07263752278660000,  0.3379294217282401,
             0.7876411410301940,    0.4910559419267466,  -0.04831174258563200,
            -0.1179901111484105,    0.003490712084217470, 0.01540410932702737
        ], device=device, dtype=dtype)
    elif name == "coif5":
        # Coiflets 5 (length 30)
        lo = torch.tensor([
            -3.459977283621256e-05, -7.098330313814114e-05,  0.0004662169601128863,
             0.001117518770890601,  -0.002574517688750223,  -0.00900797613666158,
             0.015880544863615904,  0.03455502757306163,    -0.08230192710688598,
            -0.07179982161931202,    0.42848347637761874,    0.7937772226256206,
             0.4051769024096169,    -0.06112339000267287,   -0.06577191128185562,
             0.023452696141836267,   0.007782596427325418,  -0.003793512864491014,
            -0.0002606761356811993,  0.000107502882505652,   1.10319778524429e-05,
            -5.520763127949e-06,    -1.0682196848076e-06,    5.236425333584e-07,
             1.125098976034e-07,    -5.417490769329e-08,    -8.8631e-09,
             4.2921e-09,              6.7e-10,               -3.2e-10
        ], device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown FIR wavelet: {name}")
    # 归一化到 L2≈1（避免不同长度滤波器能量差太大）
    lo = lo / (lo.pow(2).sum().sqrt() + 1e-12)
    return lo


# -----------------------------
# Wavelet 1D（逐像素门控，6候选：db4/db6/sym6/coif5/bior53/bior97）
# -----------------------------
class Wavelet1D(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_ratio: float = 0.25):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Lifting参数（bior）
        self._53 = {"a": [-0.5], "b": [0.25], "K": 1.0}
        self._97 = {"a": [-1.586134342, -0.05298011854, 0.8829110762, 0.4435068522], "K": 1.149604398}

        # 候选（6个）
        self.fir_names = ["db4", "db6", "sym6", "coif5"]
        self.lift_names = ["bior53", "bior97"]
        self.candidates = self.fir_names + self.lift_names
        K = len(self.candidates)

        # 逐像素门控
        gate_hidden = max(4, int(in_channels * hidden_ratio))
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, gate_hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_hidden, K, kernel_size=1)
        )

        # 投影
        self.proj = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1)

        # Lifting缩放
        self.scale97_A = nn.Parameter(torch.tensor(self._97["K"], dtype=torch.float32))
        self.scale97_D = nn.Parameter(torch.tensor(1.0 / self._97["K"], dtype=torch.float32))
        self.scale53_A = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.scale53_D = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    @staticmethod
    def _reflect_pad_1d(x, pad_left: int, pad_right: int):
        if pad_left == 0 and pad_right == 0:
            return x
        return F.pad(x, (pad_left, pad_right), mode="reflect")

    @staticmethod
    def _split_even_odd(x):
        N, _, L = x.shape
        if (L % 2) == 1:
            x = F.pad(x, (0, 1), mode="reflect")
        even = x[:, :, 0::2]
        odd  = x[:, :, 1::2]
        return even, odd

    @staticmethod
    def _neighbor_sum(x):
        k = torch.tensor([0.5, 0.0, 0.5], dtype=x.dtype, device=x.device).reshape(1,1,3)
        x = Wavelet1D._reflect_pad_1d(x, 1, 1)
        return F.conv1d(x, k)

    def _apply_pair(self, even, odd, coeff, is_predict: bool):
        if is_predict:
            odd = odd + coeff * self._neighbor_sum(even)
        else:
            even = even + coeff * self._neighbor_sum(odd)
        return even, odd

    # ---- LIFTING (bior) ----
    def _lifting_53(self, x_1d):
        even, odd = self._split_even_odd(x_1d)
        even, odd = self._apply_pair(even, odd, torch.as_tensor(self._53["a"][0], device=x_1d.device, dtype=x_1d.dtype), True)
        even, odd = self._apply_pair(even, odd, torch.as_tensor(self._53["b"][0], device=x_1d.device, dtype=x_1d.dtype), False)
        A = self.scale53_A * even
        D = self.scale53_D * odd
        return A, D

    def _lifting_97(self, x_1d):
        even, odd = self._split_even_odd(x_1d)
        al, be, ga, de = self._97["a"]
        even, odd = self._apply_pair(even, odd, torch.as_tensor(al, device=x_1d.device, dtype=x_1d.dtype), True)
        even, odd = self._apply_pair(even, odd, torch.as_tensor(be, device=x_1d.device, dtype=x_1d.dtype), False)
        even, odd = self._apply_pair(even, odd, torch.as_tensor(ga, device=x_1d.device, dtype=x_1d.dtype), True)
        even, odd = self._apply_pair(even, odd, torch.as_tensor(de, device=x_1d.device, dtype=x_1d.dtype), False)
        A = self.scale97_A * even
        D = self.scale97_D * odd
        return A, D

    # ---- FIR (db/sym/coif) —— 不降采样、长度保持C ----
    def _fir_analyze_1d(self, x_1d, lo_1d: torch.Tensor):
        """
        x_1d: [BHW, 1, C]
        lo_1d: [L]
        返回 (A,D) ，维度都保持 [BHW,1,C]
        """
        L = lo_1d.numel()
        lo = lo_1d.view(1, 1, L)                       # [out=1,in=1,k=L]
        hi = qmf_hi_from_lo(lo_1d).view(1, 1, L)

        pad = (L - 1) // 2
        if (L % 2) == 0:
            # 偶长度时保持对称，左右各 pad-1 与 pad
            xlp = F.pad(x_1d, (pad-1, pad), mode="reflect")
        else:
            xlp = F.pad(x_1d, (pad, pad), mode="reflect")

        A = F.conv1d(xlp, lo)
        D = F.conv1d(xlp, hi)
        # A,D: [BHW,1,C]
        return A, D

    def _one_level(self, x_1d, candidate: str):
        if candidate == "bior53":
            return self._lifting_53(x_1d)
        elif candidate == "bior97":
            return self._lifting_97(x_1d)
        elif candidate in self.fir_names:
            lo = get_lo_coeffs(candidate, device=x_1d.device, dtype=x_1d.dtype)
            return self._fir_analyze_1d(x_1d, lo)
        else:
            raise ValueError(f"Unknown candidate: {candidate}")

    def forward(self, x):
        """
        逐像素门控：gate(x) -> [B,K,H,W]，对每个像素在候选维度 softmax。
        """
        B, C, H, W = x.shape
        K = len(self.candidates)

        # 逐像素权重
        logits = self.gate(x)                      # [B,K,H,W]
        gate_w = torch.softmax(logits, dim=1)      # [B,K,H,W]

        # 在光谱维做 1D 分解
        x_reshaped = x.permute(0, 2, 3, 1).reshape(B * H * W, 1, C)  # [BHW,1,C]

        A_list, D_list = [], []
        for cand in self.candidates:
            A, D = self._one_level(x_reshaped, cand)
            # 保长到C（FIR路径已保持；bior路径偶数分解后需要拉伸回C）
            if A.shape[-1] != C:
                A = F.interpolate(A, size=C, mode="linear", align_corners=False)
            if D.shape[-1] != C:
                D = F.interpolate(D, size=C, mode="linear", align_corners=False)
            A_list.append(A)
            D_list.append(D)

        # [BHW,K,1,C]
        A_cat = torch.stack(A_list, dim=1)
        D_cat = torch.stack(D_list, dim=1)

        # 将权重展开到 BHW
        w_map = gate_w.permute(0, 2, 3, 1).reshape(B * H * W, K)  # [BHW,K]
        w_map = w_map / (w_map.sum(dim=1, keepdim=True) + 1e-12)
        wA = w_map.view(-1, K, 1, 1)

        A_fused = (A_cat * wA).sum(dim=1)   # [BHW,1,C]
        D_fused = (D_cat * wA).sum(dim=1)

        Y = torch.cat([A_fused, D_fused], dim=1)               # [BHW,2,C]
        Y = Y.reshape(B, H, W, 2, C).permute(0, 3, 4, 1, 2)    # [B,2,C,H,W]
        Y = Y.reshape(B, 2 * C, H, W)

        out = self.proj(Y)                                     # [B,out_channels,H,W]
        return out


# -----------------------------
# Wavelet 2D（逐像素门控，6候选：db4/db6/sym6/coif5/bior53/bior97）
# -----------------------------
class Wavelet2D(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_ratio: float = 0.25):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Lifting参数（bior）
        self._53 = {"aP": -0.5, "bU": 0.25, "KA": 1.0, "KD": 1.0}
        self._97 = {"alpha": -1.586134342, "beta": -0.05298011854, "gamma": 0.8829110762,
                    "delta": 0.4435068522, "KA": 1.149604398, "KD": 1.0 / 1.149604398}

        # 候选
        self.fir_names = ["db4", "db6", "sym6", "coif5"]
        self.lift_names = ["bior53", "bior97"]
        self.candidates = self.fir_names + self.lift_names
        K = len(self.candidates)

        # 逐像素门控
        hidden = max(4, int(in_channels * hidden_ratio))
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, K, 1)
        )

        # Lifting缩放
        self.scale53_A = nn.Parameter(torch.tensor(self._53["KA"], dtype=torch.float32))
        self.scale53_D = nn.Parameter(torch.tensor(self._53["KD"], dtype=torch.float32))
        self.scale97_A = nn.Parameter(torch.tensor(self._97["KA"], dtype=torch.float32))
        self.scale97_D = nn.Parameter(torch.tensor(self._97["KD"], dtype=torch.float32))

        # 高频抑制
        self.hf_scale = nn.Parameter(torch.tensor(0.9, dtype=torch.float32))

        self.proj = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1)

    @staticmethod
    def _reflect_pad2d(x, pad):
        if pad == (0,0,0,0):
            return x
        return F.pad(x, pad, mode="reflect")

    @staticmethod
    def _split_even_odd_1d(x, dim):
        _, _, H, W = x.shape
        if dim == 'W':
            if (W % 2) == 1:
                x = F.pad(x, (0, 1, 0, 0), mode="reflect")
            even = x[..., :, 0::2]
            odd  = x[..., :, 1::2]
        elif dim == 'H':
            if (H % 2) == 1:
                x = F.pad(x, (0, 0, 0, 1), mode="reflect")
            even = x[..., 0::2, :]
            odd  = x[..., 1::2, :]
        else:
            raise ValueError("dim must be 'H' or 'W'")
        return even, odd

    def _neighbor_sum_1d(self, x, dim):
        if dim == 'W':
            k = torch.tensor([[0.5, 0.0, 0.5]], dtype=x.dtype, device=x.device).reshape(1,1,1,3)
            x = self._reflect_pad2d(x, (1,1,0,0))
            y = F.conv2d(x, k)
        else:
            k = torch.tensor([[0.5],[0.0],[0.5]], dtype=x.dtype, device=x.device).reshape(1,1,3,1)
            x = self._reflect_pad2d(x, (0,0,1,1))
            y = F.conv2d(x, k)
        return y

    def _apply_pair_1d(self, even, odd, coeff, is_predict, dim):
        if is_predict:
            odd  = odd  + coeff * self._neighbor_sum_1d(even, dim)
        else:
            even = even + coeff * self._neighbor_sum_1d(odd,  dim)
        return even, odd

    # ---- LIFTING (bior) in 2D ----
    def _lifting_53_1d(self, x, dim):
        even, odd = self._split_even_odd_1d(x, dim)
        even, odd = self._apply_pair_1d(even, odd, torch.as_tensor(self._53["aP"], device=x.device, dtype=x.dtype), True,  dim)
        even, odd = self._apply_pair_1d(even, odd, torch.as_tensor(self._53["bU"], device=x.device, dtype=x.dtype), False, dim)
        A = self.scale53_A * even
        D = self.scale53_D * odd
        return A, D

    def _lifting_97_1d(self, x, dim):
        even, odd = self._split_even_odd_1d(x, dim)
        al, be, ga, de = self._97["alpha"], self._97["beta"], self._97["gamma"], self._97["delta"]
        even, odd = self._apply_pair_1d(even, odd, torch.as_tensor(al, device=x.device, dtype=x.dtype), True,  dim)
        even, odd = self._apply_pair_1d(even, odd, torch.as_tensor(be, device=x.device, dtype=x.dtype), False, dim)
        even, odd = self._apply_pair_1d(even, odd, torch.as_tensor(ga, device=x.device, dtype=x.dtype), True,  dim)
        even, odd = self._apply_pair_1d(even, odd, torch.as_tensor(de, device=x.device, dtype=x.dtype), False, dim)
        A = self.scale97_A * even
        D = self.scale97_D * odd
        return A, D

    def _one_level_1d(self, x, candidate, dim):
        if candidate == "bior53":
            return self._lifting_53_1d(x, dim)
        elif candidate == "bior97":
            return self._lifting_97_1d(x, dim)
        elif candidate in self.fir_names:
            # 用可分离FIR: 先沿指定维度做 1D 卷积（不降采样）
            lo = get_lo_coeffs(candidate, device=x.device, dtype=x.dtype)
            L = lo.numel()
            if dim == 'W':
                k_lo = lo.view(1,1,1,L)
                k_hi = qmf_hi_from_lo(lo).view(1,1,1,L)
                pad_l = (L - 1) // 2
                pad = (pad_l-1, pad_l) if (L % 2)==0 else (pad_l, pad_l)
                xpad = self._reflect_pad2d(x, (pad[0], pad[1], 0, 0))
                Lpart = F.conv2d(xpad, k_lo)
                Hpart = F.conv2d(xpad, k_hi)
            else:
                k_lo = lo.view(1,1,L,1)
                k_hi = qmf_hi_from_lo(lo).view(1,1,L,1)
                pad_l = (L - 1) // 2
                pad = (0, 0, pad_l-1, pad_l) if (L % 2)==0 else (0, 0, pad_l, pad_l)
                xpad = self._reflect_pad2d(x, (pad[0], pad[1], pad[2], pad[3]))
                Lpart = F.conv2d(xpad, k_lo)
                Hpart = F.conv2d(xpad, k_hi)
            return Lpart, Hpart
        else:
            raise ValueError(f"Unknown candidate: {candidate}")

    def _one_level_2d(self, x, candidate):
        # 记录输入尺寸，返回前强制回到该尺寸（保险1）
        N, _, H_in, W_in = x.shape
        # 先W后H（可分离）
        Lw, Hw = self._one_level_1d(x, candidate, dim='W')
        LL, LH = self._one_level_1d(Lw, candidate, dim='H')
        HL, HH = self._one_level_1d(Hw, candidate, dim='H')

        def up2orig(t):
            if t.shape[-2:] == (H_in, W_in):
                return t
            return F.interpolate(t, size=(H_in, W_in), mode="bilinear", align_corners=False)

        LL = up2orig(LL)
        LH = up2orig(LH) * self.hf_scale
        HL = up2orig(HL) * self.hf_scale
        HH = up2orig(HH) * self.hf_scale
        return LL, LH, HL, HH

    def forward(self, x):
        B, C, H, W = x.shape
        K = len(self.candidates)

        logits = self.gate(x)                    # [B,K,H,W]
        gate_w = torch.softmax(logits, dim=1)    # [B,K,H,W]

        x_bc = x.reshape(B * C, 1, H, W)

        LL_list, LH_list, HL_list, HH_list = [], [], [], []
        for cand in self.candidates:
            LL, LH, HL, HH = self._one_level_2d(x_bc, cand)

            # 兜底统一尺寸（保险2）
            if LL.shape[-2:] != (H, W):
                LL = F.interpolate(LL, size=(H, W), mode="bilinear", align_corners=False)
            if LH.shape[-2:] != (H, W):
                LH = F.interpolate(LH, size=(H, W), mode="bilinear", align_corners=False)
            if HL.shape[-2:] != (H, W):
                HL = F.interpolate(HL, size=(H, W), mode="bilinear", align_corners=False)
            if HH.shape[-2:] != (H, W):
                HH = F.interpolate(HH, size=(H, W), mode="bilinear", align_corners=False)

            LL_list.append(LL); LH_list.append(LH); HL_list.append(HL); HH_list.append(HH)

        # [B*C,K,1,H,W]
        LL_cat = torch.stack(LL_list, dim=1)
        LH_cat = torch.stack(LH_list, dim=1)
        HL_cat = torch.stack(HL_list, dim=1)
        HH_cat = torch.stack(HH_list, dim=1)

        w_bc = gate_w.repeat_interleave(C, dim=0).unsqueeze(2)        # [B*C,K,1,H,W]
        w_bc = w_bc / (w_bc.sum(dim=1, keepdim=True) + 1e-12)

        LL_f = (LL_cat * w_bc).sum(dim=1)
        LH_f = (LH_cat * w_bc).sum(dim=1)
        HL_f = (HL_cat * w_bc).sum(dim=1)
        HH_f = (HH_cat * w_bc).sum(dim=1)

        Y = torch.cat([LL_f, LH_f, HL_f, HH_f], dim=1)  # [B*C,4,H,W]
        Y = Y.reshape(B, C * 4, H, W)

        out = self.proj(Y)
        return out


class MSConv(nn.Module):
    def __init__(self, in_channels):
        super(MSConv, self).__init__()
        # 定义3种尺度的卷积，padding保证空间尺寸不变
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)
        self.conv7x7 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3)

    def forward(self, x):
        # 分别执行3种尺度卷积
        conv3 = self.conv3x3(x)
        conv5 = self.conv5x5(x)
        conv7 = self.conv7x7(x)
        # 通道维度拼接多尺度结果
        return torch.cat([conv3, conv5, conv7], dim=1)

class HyperspectralProcessor(nn.Module):
    def __init__(self, in_channels, height, width):
        super(HyperspectralProcessor, self).__init__()
        self.original_in_channels = in_channels  # 原始输入通道数
        self.height = height
        self.width = width

        # 计算补齐后的通道数（保证能被2、4、8整除）
        self.padded_in_channels = self._get_padded_channels(in_channels)

        # 为不同分组策略创建MSConv实例
        self.ms_conv_g2 = MSConv(self.padded_in_channels // 2)
        self.ms_conv_g4 = MSConv(self.padded_in_channels // 4)
        self.ms_conv_g8 = MSConv(self.padded_in_channels // 8)

        # 最终1×1卷积：将多分组+多尺度的通道数调整回原始输入通道数
        self.final_conv = nn.Conv2d(3 * self.padded_in_channels, self.original_in_channels, kernel_size=1)

    def _get_padded_channels(self, in_channels):
        # 确保通道数是8的倍数
        if in_channels % 8 == 0:
            return in_channels
        return ((in_channels + 7) // 8) * 8

    def _pad_channels(self, x):
        # 通道数不足时，用最后一个通道补齐
        batch_size, channels, height, width = x.shape
        if channels == self.padded_in_channels:
            return x
        pad_size = self.padded_in_channels - channels
        last_channel = x[:, -1:, :, :]
        padded_channels = last_channel.repeat(1, pad_size, 1, 1)
        return torch.cat([x, padded_channels], dim=1)

    def forward_return_groups(self, x):
        x_padded = self._pad_channels(x)  # 通道补齐
        batch_size, channels, height, width = x_padded.shape
        results = []  # 存储不同分组的输出

        # 遍历3种分组策略：2、4、8
        for num_groups in [2, 4, 8]:
            group_size = channels // num_groups
            grouped_x = x_padded.view(batch_size, num_groups, group_size, height, width)
            group_results = []  # 存储当前分组下各子组的结果

            # 遍历当前分组的每个子组
            for i in range(num_groups):
                single_group = grouped_x[:, i, :, :, :]  # 提取单个子组
                # 根据分组数，调用对应的MSConv实例
                if num_groups == 2:
                    combined = self.ms_conv_g2(single_group)
                elif num_groups == 4:
                    combined = self.ms_conv_g4(single_group)
                else:
                    combined = self.ms_conv_g8(single_group)
                group_results.append(combined)

            # 拼接当前分组下所有子组的结果
            grouped_result = torch.cat(group_results, dim=1)
            # 1×1卷积调整通道数回原始尺寸
            final_out = self.final_conv(grouped_result)
            results.append(final_out)

        return results  # 返回3种分组策略的结果：[out2, out4, out8]

# ================== MSAttention ==================
class MSAttention(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.1, window_size: int = 16):
        super(MSAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.scale = self.head_dim ** -0.5
        self.dim = dim
        self.window_size = window_size

        # 线性投影
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # MLP
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    # ---------- 实用工具 ----------
    @staticmethod
    def _best_hw(n: int):
        """把长度 n 分解成尽量接近正方形的 (H,W)。"""
        r = int(math.sqrt(n))
        best = (1, n)
        min_gap = n - 1
        for h in range(1, r + 1):
            if n % h == 0:
                w = n // h
                if abs(w - h) < min_gap:
                    min_gap = abs(w - h)
                    best = (h, w)
        return best  # (H,W)

    @staticmethod
    def _pad_to_window_2d(x: torch.Tensor, window: int):
        """x: [B,C,H,W] -> pad 到 window 的整数倍"""
        B, C, H, W = x.shape
        pad_h = (window - H % window) % window
        pad_w = (window - W % window) % window
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))  # (L,R,T,B)
        return x, (pad_h, pad_w)

    @staticmethod
    def _window_partition(x: torch.Tensor, window: int):
        """x: [B,C,Hp,Wp] -> windows: [B*nW, window*window, C]，以及元数据"""
        B, C, Hp, Wp = x.shape
        x = x.view(B, C, Hp // window, window, Wp // window, window)          # [B,C,nWh,w,nWw,w]
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()                           # [B,nWh,nWw,w,w,C]
        windows = x.view(-1, window * window, C)                               # [B*nW, w*w, C]
        meta = (Hp, Wp, Hp // window, Wp // window)                            # 复原需要
        return windows, meta

    @staticmethod
    def _window_reverse(windows: torch.Tensor, window: int, meta):
        """windows: [B*nW, w*w, C] -> [B,C,Hp,Wp]"""
        Hp, Wp, nWh, nWw = meta
        B = windows.shape[0] // (nWh * nWw)
        C = windows.shape[-1]
        x = windows.view(B, nWh, nWw, window, window, C)                       # [B,nWh,nWw,w,w,C]
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(                     # [B,C,Hp,Wp]
            B, C, nWh * window, nWw * window
        )
        return x

    def _x_branch(self, x_tokens, y_tokens):
        """
        X分支：在 X 的网格上，把 Y resize 到 X 的网格，再做窗口跨注意力。
        输入/输出: [B,N,C]
        """
        B, N, C = x_tokens.shape
        Hx, Wx = self._best_hw(N)

        # 还原为特征图 [B,C,H,W]
        x_map = x_tokens.transpose(1, 2).contiguous().view(B, C, Hx, Wx)

        # 把 y resize 到 x 的网格
        By, My, Cy = y_tokens.shape
        Hy, Wy = self._best_hw(My)
        y_map = y_tokens.transpose(1, 2).contiguous().view(B, C, Hy, Wy)
        y_map_rs = F.interpolate(y_map, size=(Hx, Wx), mode="bilinear", align_corners=False)

        # padding 到窗口整数倍
        x_pad, pad_xy = self._pad_to_window_2d(x_map, self.window_size)
        y_pad, _ = self._pad_to_window_2d(y_map_rs, self.window_size)

        # 分窗口 -> [B*nW, w*w, C]
        x_win, meta = self._window_partition(x_pad, self.window_size)
        y_win, _meta_y = self._window_partition(y_pad, self.window_size)

        # Q,K,V（窗口内）
        q_x = self.q_proj(x_win)  # [B*nW, T, C]
        k_y = self.k_proj(y_win)
        v_x = self.v_proj(x_win)

        # 多头
        Bn, T, C = q_x.shape
        h, d = self.num_heads, self.head_dim
        q_x = q_x.view(Bn, T, h, d).permute(0, 2, 1, 3)          # [Bn,h,T,d]
        k_y = k_y.view(Bn, T, h, d).permute(0, 2, 1, 3)
        v_x = v_x.view(Bn, T, h, d).permute(0, 2, 1, 3)

        attn_x = (q_x @ k_y.transpose(-2, -1)) * self.scale      # [Bn,h,T,T]
        attn_x = self.dropout(attn_x.softmax(dim=-1))
        out_x = attn_x @ v_x                                     # [Bn,h,T,d]
        out_x = out_x.transpose(1, 2).contiguous().view(Bn, T, C)  # [Bn,T,C]

        # 窗口内线性投影并拼回整图
        out_x = self.out_proj(out_x)                              # [Bn,T,C]
        out_x_map = self._window_reverse(out_x, self.window_size, meta)  # [B,C,Hp,Wp]

        # 去 padding
        Hp, Wp = out_x_map.shape[-2:]
        ph, pw = pad_xy
        if ph or pw:
            out_x_map = out_x_map[:, :, :Hp - ph, :Wp - pw]

        # 拉平 + 残差 + MLP（与原逻辑一致）
        out_x_tokens = out_x_map.flatten(2).transpose(1, 2).contiguous()  # [B,N,C]
        out_x_tokens = out_x_tokens + x_tokens
        out_x_tokens = self.mlp(out_x_tokens) + out_x_tokens
        return out_x_tokens

    def _y_branch(self, x_tokens, y_tokens):
        """
        Y分支：在 Y 的网格上，把 X resize 到 Y 的网格，再做窗口跨注意力。
        输入/输出: [B,M,C]
        """
        B, M, C = y_tokens.shape
        Hy, Wy = self._best_hw(M)

        y_map = y_tokens.transpose(1, 2).contiguous().view(B, C, Hy, Wy)

        # 把 x resize 到 y 的网格
        Bx, Nx, Cx = x_tokens.shape
        Hx, Wx = self._best_hw(Nx)
        x_map = x_tokens.transpose(1, 2).contiguous().view(B, C, Hx, Wx)
        x_map_rs = F.interpolate(x_map, size=(Hy, Wy), mode="bilinear", align_corners=False)

        # padding
        y_pad, pad_y = self._pad_to_window_2d(y_map, self.window_size)
        x_pad, _ = self._pad_to_window_2d(x_map_rs, self.window_size)

        # 分窗口
        y_win, meta = self._window_partition(y_pad, self.window_size)
        x_win, _meta_x = self._window_partition(x_pad, self.window_size)

        # Q,K,V（窗口内）
        q_y = self.q_proj(y_win)
        k_x = self.k_proj(x_win)
        v_y = self.v_proj(y_win)

        Bn, T, C = q_y.shape
        h, d = self.num_heads, self.head_dim
        q_y = q_y.view(Bn, T, h, d).permute(0, 2, 1, 3)
        k_x = k_x.view(Bn, T, h, d).permute(0, 2, 1, 3)
        v_y = v_y.view(Bn, T, h, d).permute(0, 2, 1, 3)

        attn_y = (q_y @ k_x.transpose(-2, -1)) * self.scale
        attn_y = self.dropout(attn_y.softmax(dim=-1))
        out_y = attn_y @ v_y
        out_y = out_y.transpose(1, 2).contiguous().view(Bn, T, C)

        out_y = self.out_proj(out_y)
        out_y_map = self._window_reverse(out_y, self.window_size, meta)

        Hp, Wp = out_y_map.shape[-2:]
        ph, pw = pad_y
        if ph or pw:
            out_y_map = out_y_map[:, :, :Hp - ph, :Wp - pw]

        out_y_tokens = out_y_map.flatten(2).transpose(1, 2).contiguous()  # [B,M,C]
        out_y_tokens = out_y_tokens + y_tokens
        out_y_tokens = self.mlp(out_y_tokens) + out_y_tokens
        return out_y_tokens

    def forward(self, x, y):
        """
        x: [B, N, C]
        y: [B, M, C]
        返回: [B, N+M, C]  （接口保持不变）
        """
        # X 分支（在 X 网格内做局部跨注意力）
        out_x = self._x_branch(x, y)
        # Y 分支（在 Y 网格内做局部跨注意力）
        out_y = self._y_branch(x, y)
        return torch.cat([out_x, out_y], dim=1)


# ================== 整合模块 ==================
class GCSE(nn.Module):
    def __init__(self, in_channels, height, width, dim, num_heads=8):
        super(GCSE, self).__init__()
        self.height = height
        self.width = width
        self.dim = dim

        self.processor = HyperspectralProcessor(in_channels, height, width)

        # 各分组 + 原始输入做通道对齐
        self.align_2 = nn.Conv2d(in_channels, dim, kernel_size=1)
        self.align_4 = nn.Conv2d(in_channels, dim, kernel_size=1)
        self.align_8 = nn.Conv2d(in_channels, dim, kernel_size=1)
        self.align_orig = nn.Conv2d(in_channels, dim, kernel_size=1)

        # 三个 MSAttention
        self.attn2 = MSAttention(dim, num_heads=num_heads)
        self.attn4 = MSAttention(dim, num_heads=num_heads)
        self.attn8 = MSAttention(dim, num_heads=num_heads)

        # 融合 1×1 conv
        self.fuse_conv = nn.Conv2d(6 * dim, dim, kernel_size=1)
        # 外部必须提供的参考分支（如对齐后的 lr-HSI / sem_hsi_out）
        self.ref_map = None

    @staticmethod
    def _ensure_hw(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """把 x 的空间尺寸对齐到 (H, W)，不改变 batch/通道数。"""
        if x.shape[-2:] == (H, W):
            return x
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

    def forward(self, x):
        if self.ref_map is None:
            raise RuntimeError(
                "GCSE.ref_map 未设置。请在调用 forward 之前设置，例如："
                " gcse.ref_map = hsi_upsampled  或  gcse.ref_map = sem_hsi_out"
            )

        # 1) 三个分组卷积的结果（X 端）
        final_outs = self.processor.forward_return_groups(x)   # [out2, out4, out8]

        B, C, H, W = x.shape
        N = H * W

        # 2) 参考分支（Y 端）——严格使用 self.ref_map
        ref = self.ref_map
        ref = self._ensure_hw(ref, H, W)           # 空间对齐到 x
        if ref.device != x.device:                 # 设备对齐（安全）
            ref = ref.to(x.device)
        if ref.dtype != x.dtype:                   # 精度对齐（安全）
            ref = ref.to(x.dtype)

        # 3) 原始/参考输入 -> tokens（Y 端）
        orig_feat = self.align_orig(ref).flatten(2).transpose(1, 2)  # [B, N, dim]

        # 4) 三个分支：分组输出（X 端） 与 参考 tokens（Y 端）做跨注意力
        out2 = self.attn2(self.align_2(final_outs[0]).flatten(2).transpose(1, 2), orig_feat)
        out4 = self.attn4(self.align_4(final_outs[1]).flatten(2).transpose(1, 2), orig_feat)
        out8 = self.attn8(self.align_8(final_outs[2]).flatten(2).transpose(1, 2), orig_feat)

        # 5) 拼接三个分支
        concat_out = torch.cat([out2, out4, out8], dim=1)  # [B, 6N, dim]

        # 6) 融合回 [B, N, dim]
        feat_map = concat_out.transpose(1, 2).reshape(B, 6 * self.dim, H, W)  # [B, 6*dim, H, W]
        fused = self.fuse_conv(feat_map)                                      # [B, dim, H, W]
        fused_out = fused.flatten(2).transpose(1, 2)                          # [B, N, dim]

        return fused_out


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class SCSGM(nn.Module):
    r"""
    光谱压缩与空间引导模块（Spectral Compression & Spatial Guidance Module, 双输入版）

    输入:
        x_hsi: [B, C, Hx, Wx]  —— 被调制的 HSI 特征
        y_msi: [B, Cm, Hy, Wy] —— 引导源（MSI 特征）
    输出:
        out: [B, C, Hx, Wx]

    公式结构与单输入版完全一致，只是显式传入 y_msi，不再依赖 set_guide。
    """

    def __init__(
        self,
        in_channels: int,
        guide_channels: int,
        reduction: int = 4,
        kernel_size: int = 7,
        alpha: float = 0.15,
        use_bn: bool = False,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size 必须为奇数"
        self.C = int(in_channels)
        self.Cm = int(guide_channels)
        self.alpha = float(alpha)
        mid = max(8, self.C // reduction)

        # ---------- (1) 通道注意力 ----------
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.C, mid, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, self.C, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # ---------- (2) 空间引导 ----------
        sg_layers = [
            nn.Conv2d(self.Cm, self.Cm, kernel_size=kernel_size,
                      padding=kernel_size // 2, groups=self.Cm, bias=not use_bn),
        ]
        if use_bn:
            sg_layers += [nn.BatchNorm2d(self.Cm)]
        sg_layers += [
            nn.ReLU(inplace=True),
            nn.Conv2d(self.Cm, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        ]
        self.sg = nn.Sequential(*sg_layers)

        # ---------- (3) 联合门控 ----------
        self.gate = nn.Sequential(
            nn.Conv2d(self.C + 1, self.C, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # ---------- (4) 输出映射 ----------
        self.post = nn.Identity()

    @staticmethod
    def _ensure_same_hw(t: torch.Tensor, ref_hw: Tuple[int, int]) -> torch.Tensor:
        """将输入 t 对齐到参考尺寸 (H, W)。"""
        H, W = ref_hw
        if t.shape[-2:] == (H, W):
            return t
        return F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)

    def forward(self, x_hsi: torch.Tensor, y_msi: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x_hsi: [B, C, H, W]
        y_msi: [B, Cm, Hy, Wy]，可为 None
        """
        B, C, H, W = x_hsi.shape
        assert C == self.C, f"in_channels 不匹配: got {C}, expect {self.C}"

        # (1) 通道注意力
        As = self.se(x_hsi)                   # [B, C, 1, 1]
        As_up = As.expand(-1, -1, H, W)       # [B, C, H, W]

        # (2) 空间引导
        if y_msi is not None:
            assert y_msi.dim() == 4 and y_msi.size(1) == self.Cm, \
                f"guide_channels 不匹配: got {y_msi.size(1)}, expect {self.Cm}"
            if y_msi.device != x_hsi.device:
                y_msi = y_msi.to(x_hsi.device)
            if y_msi.dtype != x_hsi.dtype:
                y_msi = y_msi.to(dtype=x_hsi.dtype)

            y_msi = self._ensure_same_hw(y_msi, (H, W))
            Sm = self.sg(y_msi)               # [B, 1, H, W]
        else:
            Sm = x_hsi.new_full((B, 1, H, W), 0.5)

        # (3) 联合门控
        G = self.gate(torch.cat([As_up, Sm], dim=1))  # [B, C, H, W]

        # (4) 残差缩放调制
        out = x_hsi * (1.0 + self.alpha * (G - 0.5))
        out = self.post(out)
        return out

class AWD(nn.Module):
    """
    AWD（不冻结参数版）：
      - 保持和原始接口一致：forward(x_lr, x_hr, tokens_fused=None) -> (sr_hsi, 0, 0, 0, 0, 0)
      - 提供 load_mid_fusion(ckpt_path) 用于加载“中间融合链路”权重，但不会冻结任何参数。
    """
    def __init__(self, scale_ratio, n_select_bands, n_bands, width=48, num_blks=8,
                 drop_path_rate=0., drop_out_rate=0.1, fusion_from=-1, fusion_to=1000, dual=True):
        super(AWD, self).__init__()
        self.scale_ratio = scale_ratio
        self.n_bands = n_bands
        self.n_select_bands = n_select_bands
        self.width = width
        self.num_blks = num_blks
        self.dual = dual

        # ======= HSI 分支（左上）=======
        self.upsample_hsi = nn.Upsample(scale_factor=scale_ratio, mode='bilinear', align_corners=False)
        self.sem_hsi = SEM(n_bands, n_bands)            # 可选增强（保留）
        self.wavelet_1d = Wavelet1D(n_bands, n_bands)   # 可选光谱子带（保留）
        self.conv1x1_1 = nn.Conv2d(n_bands * 2, n_bands, kernel_size=1)

        # ======= MSI 分支（左下）=======
        self.sem_msi = SEM(n_select_bands, n_select_bands)
        self.wavelet_2d = Wavelet2D(n_select_bands, n_select_bands)
        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(n_select_bands + n_bands, n_select_bands, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_select_bands, n_select_bands, 3, padding=1)
        )

        # 融合到 n_bands（中部轻映射，用于右侧分支与最终拼接）
        self.conv1x1_final = nn.Sequential(
            nn.Conv2d(n_select_bands * 2 + n_bands, n_bands, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_bands, n_bands, 3, padding=1)
        )

        # ======= 右上注意力（lazy init）=======
        # 运行时实例化：GCSE(in_channels=n_bands, height=H_up, width=W_up, dim=width, num_heads=8)
        self.hsi_attention = None

        # （按你的要求显式实例化，当前不在 forward 中调用）
        self.msconv_bottom = MSConv(self.n_bands)

        # ======= 右上融合后：特征拼接 -> 回到 n_bands =======
        #   输入拼接: [B, width + n_bands, H, W]  ->  输出: [B, n_bands, H, W]
        self.fuse_head = nn.Sequential(
            nn.Conv2d(self.width + self.n_bands, self.width, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.width, self.n_bands, kernel_size=1)
        )

        # ========MSI分支右下（保留占位，forward 未使用）=======
        self.scsgm = SCSGM(
            in_channels=self.n_bands,
            guide_channels=self.n_select_bands,  # ← MSI 的通道数
            reduction=4, kernel_size=7, alpha=0.15, use_bn=False
        )

        self.bottom_align = nn.Conv2d(3 * self.n_bands, self.width, 1)
        self.hr_align = nn.Conv2d(self.n_select_bands, self.width, 1)
        self.bottom_cross_attn = MSAttention(self.width)
        self.msa_fuse_bottom = MSAttention(self.width)

        # 可选的“仅 tokens → SR-HSI”的直连头（保留以便做 ablation；本实现不使用）
        self.fuse_head_direct = nn.Sequential(
            nn.Conv2d(self.width, self.width // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.width // 2, self.n_bands, kernel_size=1)
        )

    def forward(self, x_lr, x_hr, tokens_fused=None):
        """
        x_lr: [B, n_bands, h_l, w_l]    # 低分辨率 HSI
        x_hr: [B, n_select_bands, H, W] # 高分辨率 MSI
        """
        B = x_lr.size(0)
        H_up, W_up = x_hr.shape[-2:]

        # ================= 左上：HSI光谱主路 =================
        # 上采样到 HR 尺寸
        hsi_upsampled = self.upsample_hsi(x_lr)  # [B, n_bands, H_up, W_up]
        # 浅层增强（图中 SEM）
        sem_hsi_out = self.sem_hsi(hsi_upsampled)  # [B, n_bands, H_up, W_up]
        # 1D-DWT（光谱分解）
        w1d_out = self.wavelet_1d(hsi_upsampled)  # [B, n_bands, H_up, W_up]
        # 与原谱特征合一路（图中 “1D-DWT → Concat → Conv1×1”）
        spec_feat = self.conv1x1_1(torch.cat([w1d_out, hsi_upsampled], dim=1))  # [B, n_bands, H_up, W_up]

        # ================= 左下：MSI 空间主路 =================
        # 2D-DWT（空间子带）
        w2d_out = self.wavelet_2d(x_hr)  # [B, n_select_bands, H_up, W_up]
        # 浅层增强（图中 MSI 的 SEM）
        sem_msi_out = self.sem_msi(x_hr)  # [B, n_select_bands, H_up, W_up]

        # ====== 中部融合（两路多次 Concat 后的 Conv1×1）======
        # 先把 HSI 的谱特征与 MSI 的 2D 子带建立中间联系（图中中间的 Concat → Conv1×1）
        mid_conv = self.conv1x1_2(torch.cat([spec_feat, w2d_out], dim=1))  # [B, n_select_bands, H_up, W_up]
        # 再与 HSI 的 SEM、MSI 的 SEM 进行总融合（图中再次 Concat）
        final_feat = torch.cat([sem_hsi_out, mid_conv, sem_msi_out], dim=1)  # [B, n_bands + 2*n_sel, H_up, W_up]
        # 轻映射得到“谱侧”特征图（供右侧 group 分支与最终拼接使用）
        final_feat = self.conv1x1_final(final_feat)  # [B, n_bands, H_up, W_up]

        # ================= 右上：Group=2/4/8 → MSConv → MSAttention =================
        # 懒加载 GCSE（右上三支 + Concat + Conv1×1 的整体）
        if self.hsi_attention is None:
            self.hsi_attention = GCSE(
                in_channels=self.n_bands,
                height=H_up,
                width=W_up,
                dim=self.width,
                num_heads=8
            ).to(final_feat.device)

        # 指定右上参考分支（Y 端）
        #   这里采用对齐后的 lr-HSI（也可切换为 sem_hsi_out）
        self.hsi_attention.ref_map = hsi_upsampled

        # 经过右上三支得到 X1 的 tokens
        x_tokens = self.hsi_attention(final_feat)  # [B, N, width]，N=H_up*W_up

        # ================= 右下：SCSGM 生成 Y1，并与 X1 跨注意力（保持你的旧逻辑） =================
        y1_map = self.scsgm(final_feat, sem_msi_out)  # [B, n_bands, H_up, W_up]
        y_tokens = self.hsi_attention.align_orig(y1_map).flatten(2).transpose(1, 2)  # [B, N, width]
        tokens_cat = self.msa_fuse_bottom(x_tokens, y_tokens)  # [B, 2N, width]
        N = x_tokens.shape[1]
        tokens_fused = tokens_cat[:, :N, :]  # 取 X 网格对应的融合结果作为主输出

        # ============== tokens → 特征图 ==============
        top_feat_map = tokens_fused.transpose(1, 2).contiguous().view(B, self.width, H_up, W_up)  # [B, width, H, W]

        # ==============（你要求的改动）与 final_feat 做通道拼接，再卷积回到 n_bands ==============
        fused_map = torch.cat([top_feat_map, final_feat], dim=1)  # [B, width + n_bands, H, W]
        sr_hsi = self.fuse_head(fused_map)                       # [B, n_bands, H, W]

        # 接口保持不变
        zero = torch.zeros_like(sr_hsi)
        # return sr_hsi, zero, zero, zero, zero, zero
        return sr_hsi