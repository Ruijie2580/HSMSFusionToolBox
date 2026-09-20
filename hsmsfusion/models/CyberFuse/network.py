import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6, affine=False):
        super().__init__()
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        # x: (B, C, H, W)

        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)

        x = (x - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            x = x * self.weight + self.bias

        return x


class BaseFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fusion = nn.Conv2d(dim * 2, dim, 1, 1, 0)

    def forward(self, x, y):
        return self.fusion(torch.cat([x, y], dim=1))


class B1(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.stripconv = nn.Sequential(
            nn.Conv2d(dim, dim, 5, 1, 2, groups=dim),
            nn.Conv2d(dim, dim, (1, 29), 1, (0, 29 // 2), groups=dim),
            nn.Conv2d(dim, dim, (29, 1), 1, (29 // 2, 0), groups=dim),
            nn.Conv2d(dim, dim, 1, 1, 0),
        )
        # self.pconv = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, x):
        x2 = self.stripconv(x)
        return x * x2


class B2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.stripconv = nn.Sequential(
            nn.Conv2d(dim, dim, 5, 1, 2, groups=dim),
            nn.Conv2d(dim, dim, (1, 19), 1, (0, 19 // 2), groups=dim),
            nn.Conv2d(dim, dim, (19, 1), 1, (19 // 2, 0), groups=dim),
            nn.Conv2d(dim, dim, 1, 1, 0),
        )

    def forward(self, x):
        x2 = self.stripconv(x)
        return x * x2


class B3(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.middle_conv = nn.Conv2d(dim, dim, 9, 1, 9 // 2, groups=dim)
        self.pconv = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, x):
        x2 = self.middle_conv(x)
        x2 = self.pconv(x2)
        return x * x2


class B4(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.local_conv = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)
        self.pconv = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, x):
        x2 = self.local_conv(x)
        x2 = self.pconv(x2)
        return x * x2


class GroupLKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert dim % 4 == 0, f"dim must be divisible by 4, but got {dim}"
        group_dim = dim // 4

        self.conv = nn.Conv2d(dim, dim, 1, 1, 0)

        self.pwconv1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.pwconv2 = nn.Conv2d(dim, dim, 1, 1, 0)

        self.B1 = B1(group_dim)
        self.B2 = B2(group_dim)
        self.B3 = B3(group_dim)
        self.B4 = B4(group_dim)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
            ),
        )

    def forward(self, x):
        x1 = x2 = x
        x1 = self.conv(x1)

        x11, x12, x13, x14 = torch.chunk(x1, 4, dim=1)
        x11 = self.B1(x11)
        x12 = self.B2(x12)
        x13 = self.B3(x13)
        x14 = self.B4(x14)
        x1 = torch.cat([x11, x12, x13, x14], dim=1)
        x1 = self.pwconv1(x1)
        x = x1 * x2
        x = x * self.sca(x)
        x = self.pwconv2(x)
        return x


class FFN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pwconv1 = nn.Conv2d(dim, dim * 2, 1, 1, 0)
        self.pwconv2 = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.pwconv1(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x = x1 * x2
        x = self.pwconv2(x)
        return x


class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.norm2 = LayerNorm2d(dim)
        self.GLKA = GroupLKA(dim)
        # self.GLKA = MSLKA(dim)
        self.ffn = FFN(dim)
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        res = x
        x = self.norm1(x)
        x = self.GLKA(x)
        x = res + self.beta * x
        res = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = res + self.gamma * x
        return x


class Blocks(nn.Module):
    def __init__(self, dim, num_blocks=2):
        super().__init__()
        # self.blocks = nn.ModuleList([NAFBlock(dim) for _ in range(num_blocks)])
        self.blocks = nn.ModuleList([Block(dim) for _ in range(num_blocks)])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        # x = self.attn(x)
        return x


class Prior(nn.Module):
    def __init__(
        self,
        dim_h,
        dim_m,
        embed_dim=64,
        num_blocks=[4, 4, 4, 8],
        num_multiplyer=[1, 1, 1, 1],
    ):
        super().__init__()

        self.init_hsi = nn.Conv2d(dim_h, embed_dim, 3, 1, 1)
        self.init_msi = nn.Conv2d(dim_m, embed_dim, 3, 1, 1)

        depth = len(num_blocks)
        dims = [embed_dim * m for m in num_multiplyer]

        # ---------------- Encoder ----------------
        self.down_msi = nn.ModuleList()
        self.down_hsi = nn.ModuleList()

        for i in range(depth - 1):
            self.down_msi.append(nn.Conv2d(dims[i], dims[i + 1], 3, 2, 1))
            self.down_hsi.append(nn.Conv2d(dims[i], dims[i + 1], 3, 2, 1))

        # ---------------- Decoder ----------------
        self.ups = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.fusers = nn.ModuleList()
        self.skip_fuse = nn.ModuleList()

        for i in reversed(range(depth - 1)):

            self.ups.append(nn.ConvTranspose2d(dims[i + 1], dims[i], 3, 2, 1, 1))

            self.skip_fuse.append(BaseFusion(dims[i]))

            self.decoder.append(Blocks(dims[i], num_blocks[i]))

        self.bottle = Blocks(dims[-1], num_blocks[-1])

        self.out = nn.Conv2d(embed_dim, dim_h, 3, 1, 1)

    def run_block(self, block, x):
        x = block(x)
        if isinstance(x, tuple):
            x = x[0]
        return x

    def forward(self, hs, ms):

        x_hs = self.init_hsi(hs)
        x_ms = self.init_msi(ms)

        skip_hs = [x_hs]
        skip_ms = [x_ms]

        # ---------------- Encoder ----------------
        for down_hs, down_ms in zip(self.down_hsi, self.down_msi):

            x_hs = down_hs(x_hs)
            x_ms = down_ms(x_ms)

            skip_hs.append(x_hs)
            skip_ms.append(x_ms)

        # ---------------- Bottle ----------------
        # x = self.init_fuse(x_hs, x_ms)
        x = x_hs + x_ms
        x = self.run_block(self.bottle, x)

        # ---------------- Decoder ----------------
        for i, (up, dec, skip_fus) in enumerate(
            zip(self.ups, self.decoder, self.skip_fuse)
        ):

            x = up(x)

            hs_skip = skip_hs[-(i + 2)]
            ms_skip = skip_ms[-(i + 2)]
            # skip = fuse(hs_skip, ms_skip)
            skip = hs_skip + ms_skip

            # x = torch.cat([x, skip], dim=1)

            x = skip_fus(x, skip)

            x = self.run_block(dec, x)

        out = self.out(x)

        return out + hs


class SpectralDegradation(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)

    def forward(self, x):
        return self.conv(x)

    def transpose(self, x):
        return F.conv_transpose2d(x, self.conv.weight, bias=None)


class SpatialDegradation(nn.Module):
    def __init__(self, in_c, scale):
        super().__init__()

        self.scale = scale

        # blur convolution (simulate PSF)
        self.blur = nn.Conv2d(
            in_c,
            in_c,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=in_c,
            bias=False,
        )

        # downsample convolution
        self.down = nn.Conv2d(
            in_c,
            in_c,
            kernel_size=3,
            stride=scale,
            padding=1,
            groups=in_c,
            bias=False,
        )

    def forward(self, x):
        x = self.blur(x)
        x = self.down(x)
        return x

    def transpose(self, x):

        # transpose of downsample
        x = F.conv_transpose2d(
            x,
            self.down.weight,
            stride=self.scale,
            padding=1,
            groups=self.down.groups,
            output_padding=self.scale - 1,
        )

        # transpose of blur
        x = F.conv_transpose2d(
            x,
            self.blur.weight,
            stride=1,
            padding=2,
            groups=self.blur.groups,
        )

        return x


class Fidelity(nn.Module):
    def __init__(self, ntier=2, spe_de=None, spa_de=None):
        super().__init__()
        self.ntier = ntier
        self.spe_de = spe_de
        self.spa_de = spa_de

        self.ETA = nn.Parameter(torch.full((ntier,), 0.5))
        self.DELTA = nn.Parameter(torch.full((ntier,), 0.1))

    def forward(self, X, Y_h, Y_m, stage_idx):
        eta = self.ETA[stage_idx]
        delta = self.DELTA[stage_idx]

        diff_spe = self.spe_de(X) - Y_m
        grad_spe = self.spe_de.transpose(diff_spe)

        diff_spa = self.spa_de(X) - Y_h
        grad_spa = self.spa_de.transpose(diff_spa)

        X_new = X - eta * grad_spe - delta * grad_spa

        return X_new


class Net(nn.Module):
    def __init__(self, hsi_channels=31, msi_channels=3, scale=4, ntier=2):
        super(Net_T, self).__init__()

        self.scale = scale
        self.ntier = ntier

        self.hsi_channels = hsi_channels
        self.msi_channels = msi_channels

        self.spa_de = SpatialDegradation(hsi_channels, scale)
        self.spe_de = SpectralDegradation(hsi_channels, msi_channels)

        self.fidelity = Fidelity(ntier, self.spe_de, self.spa_de)

        self.prior = nn.ModuleList(
            [Prior(hsi_channels, msi_channels) for _ in range(ntier + 1)]
        )

    def forward(self, LRHS, HRMS):
        B = LRHS.shape[0]

        X = F.interpolate(
            LRHS,
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False,
        )

        X = self.prior[0](X, HRMS)

        for i in range(self.ntier):
            X = self.fidelity(X, LRHS, HRMS)

            X = self.prior[i + 1](X, HRMS)

        return X


class GainNet(nn.Module):
    def __init__(self, channels, hidden_dim=48):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, hidden_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 9, 1, 9 // 2, groups=hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, channels, 3, 1, 1)
        # self.conv1 = DSConv(channels, hidden_dim, 3)
        # self.conv2 = DSConv(hidden_dim, hidden_dim, 7)
        # self.conv3 = DSConv(hidden_dim, channels, 3)

        self.act = nn.SiLU()

    def forward(self, error):
        feat = self.conv1(error)
        feat = self.act(feat)

        feat = self.conv2(feat)
        feat = self.act(feat)

        out = F.sigmoid(self.conv3(feat))

        return out


class PIDController(nn.Module):
    def __init__(self, channels, leaky=1):
        super().__init__()

        # 初始化策略：
        # kp 初始较大，主导初期优化；ki, kd 初始较小，随后慢慢起效
        # 注意：这里存的是“原像”，前向传播时会经过 softplus 保证非负
        self.kp_raw = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.5)
        self.ki_raw = nn.Parameter(
            torch.zeros(1, channels, 1, 1) - 2.0
        )  # softplus(-2) 接近于一个很小的正数
        self.kd_raw = nn.Parameter(torch.zeros(1, channels, 1, 1) - 2.0)

        # 积分遗忘因子 (Leaky Integral)，通常设为 0.8 ~ 0.95
        self.leaky = leaky

    def forward(self, error, prev_error, integral):
        # 1. 计算离散微分和带衰减的积分
        derivative = error - prev_error

        # Anti-windup
        integral = self.leaky * integral + error

        # Stability Guarantee
        kp = F.softplus(self.kp_raw)
        ki = F.softplus(self.ki_raw)
        kd = F.softplus(self.kd_raw)

        # 3. 计算控制信号 (更新量)
        control = kp * error + ki * integral + kd * derivative

        return control, integral


class PIController(nn.Module):
    def __init__(self, channels, leaky=1):
        super().__init__()

        self.kp_raw = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.5)
        self.ki_raw = nn.Parameter(torch.zeros(1, channels, 1, 1) - 2.0)

        self.leaky = leaky

    def forward(self, error, prev_error, integral):
        # integral (带 anti-windup)
        integral = self.leaky * integral + error

        kp = F.softplus(self.kp_raw)
        ki = F.softplus(self.ki_raw)

        control = kp * error + ki * integral

        return control, integral


class PDController(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.kp_raw = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.5)
        self.kd_raw = nn.Parameter(torch.zeros(1, channels, 1, 1) - 2.0)

    def forward(self, error, prev_error, integral):
        # derivative
        derivative = error - prev_error

        # 保持 integral 传递（但不使用），避免接口破坏
        # 不更新 integral（或也可以继续更新但不参与控制）
        integral = integral  # no-op

        kp = F.softplus(self.kp_raw)
        kd = F.softplus(self.kd_raw)

        control = kp * error + kd * derivative

        return control, integral


class KalmanPIDFidelity(nn.Module):

    def __init__(self, hsi_channels, msi_channels, spe_de, spa_de, total_steps):
        super().__init__()
        self.spe_de = spe_de
        self.spa_de = spa_de

        self.gain_s = GainNet(msi_channels)
        self.gain_p = GainNet(hsi_channels)

        self.pid_s = PIDController(msi_channels)
        self.pid_p = PIDController(hsi_channels)
        # self.pid_s = PIController(msi_channels)
        # self.pid_p = PIController(hsi_channels)
        # self.pid_s = PDController(msi_channels)
        # self.pid_p = PDController(hsi_channels)

        self.alpha = nn.Parameter(torch.full((total_steps,), 0.1))
        # self.DELTA = nn.Parameter(torch.full((ntier,), 0.1))

        # 可视化缓存
        self.vis_update_s_all = []
        self.vis_update_p_all = []
        self.vis_update_all = []

    def init_state(self, X, Y_h, Y_m):

        B = X.shape[0]
        # B = 1

        H_hr, W_hr = Y_m.shape[2:]
        H_lr, W_lr = Y_h.shape[2:]

        prev_e_s = torch.zeros(B, Y_m.shape[1], H_hr, W_hr, device=X.device)
        prev_e_p = torch.zeros(B, Y_h.shape[1], H_lr, W_lr, device=X.device)

        int_s = torch.zeros_like(prev_e_s)
        int_p = torch.zeros_like(prev_e_p)

        return (prev_e_s, prev_e_p, int_s, int_p)

    def forward(self, X, Y_h, Y_m, state, stage_idx):
        prev_e_s, prev_e_p, int_s, int_p = state

        e_s = Y_m - self.spe_de(X)
        e_p = Y_h - self.spa_de(X)

        K_s = self.gain_s(e_s)
        K_p = self.gain_p(e_p)

        u_s, int_s = self.pid_s(e_s, prev_e_s, int_s)
        u_p, int_p = self.pid_p(e_p, prev_e_p, int_p)

        update_s = self.spe_de.transpose(K_s * u_s)
        update_p = self.spa_de.transpose(K_p * u_p)
        # update_s = self.spe_de.transpose(u_s)
        # update_p = self.spa_de.transpose(u_p)

        update = update_s + update_p

        # ==================================================
        # 可视化缓存
        # ==================================================
        if not self.training:
            self.vis_update_s_all.append(u_s.detach().cpu())
            self.vis_update_p_all.append(u_p.detach().cpu())
            self.vis_update_all.append(update.detach().cpu())

        X_new = X + self.alpha[stage_idx] * update

        new_state = (e_s, e_p, int_s, int_p)
        return X_new, new_state


class Net(nn.Module):
    def __init__(
        self, hsi_channels=31, msi_channels=3, scale=4, ntier=2, n_sub_iters=3
    ):
        super(Net, self).__init__()

        self.scale = scale
        self.ntier = ntier
        self.n_sub_iters = n_sub_iters
        self.total_steps = ntier * n_sub_iters

        self.hsi_channels = hsi_channels
        self.msi_channels = msi_channels

        self.spa_de = SpatialDegradation(hsi_channels, scale)
        self.spe_de = SpectralDegradation(hsi_channels, msi_channels)

        # self.fidelity = Fidelity(ntier, self.spe_de, self.spa_de)
        self.fidelity = KalmanPIDFidelity(
            hsi_channels, msi_channels, self.spe_de, self.spa_de, self.total_steps
        )

        self.prior = nn.ModuleList(
            [Prior(hsi_channels, msi_channels) for _ in range(ntier + 1)]
        )

    def forward(self, LRHS, HRMS):
        self.fidelity.vis_update_s_all = []
        self.fidelity.vis_update_p_all = []
        self.fidelity.vis_update_all = []

        B = LRHS.shape[0]

        X = F.interpolate(
            LRHS,
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False,
        )

        outputs = []

        X = self.prior[0](X, HRMS)

        state = self.fidelity.init_state(X, LRHS, HRMS)

        # outputs.append(X)
        idx = 0

        for i in range(self.ntier):

            for sub_step in range(self.n_sub_iters):
                X, state = self.fidelity(X, LRHS, HRMS, state, idx)
                idx += 1

            X = self.prior[i + 1](X, HRMS)

            # outputs.append(X)

        return X
