import torch
import torch.nn as nn
import torch.nn.functional as F
from . import common
from .ENLA import ENLA
import math
from . import common_1

class ENLCA(nn.Module):
    def __init__(self, channel=128, reduction=2, ksize=3, scale=3, stride=1, softmax_scale=10, average=True,
                 conv=common.default_conv, res_scale=0.1):
        super(ENLCA, self).__init__()
        self.conv_match1 = common.BasicBlock(conv, channel, channel // reduction, 1, bn=False, act=None)
        self.conv_match2 = common.BasicBlock(conv, channel, channel // reduction, 1, bn=False, act=None)
        self.conv_assembly = common.BasicBlock(conv, channel, channel, 1, bn=False, act=None)
        self.res_scale = res_scale
        self.attn_fn = ENLA(
            dim_heads=channel // reduction,
            nb_features=128,
        )
        self.k = math.sqrt(6)
        n_select_bands = 5
        self.conv_CRAC = nn.Sequential(
            common_1.Conv2d_CG(channel, channel // reduction),
            nn.ReLU(inplace=True),
            # common_1.Conv2d_CG(n_bands, n_bands)
        )
        self.conv_CRAC1 = nn.Sequential(
            common_1.Conv2d_CG(channel, channel),
            nn.ReLU(inplace=True),
            # common_1.Conv2d_CG(n_bands, n_bands)
        )
    def forward(self, input):
        # x_embed_1 =self.conv_CRAC(input)
        # x_embed_2 = self.conv_CRAC(input)
        # x_assembly = self.conv_CRAC1(input)

        x_embed_1 = self.conv_match1(input)
        x_embed_2 = self.conv_match2(input)
        x_assembly = self.conv_assembly(input) #[B,C,H,W]

        x_embed_2 = F.normalize(x_embed_2, p=2, dim=1,eps=5e-5)*self.k     #归一化：将某一个维度除以那个维度对应的范数(默认是2范数)
        x_embed_1 = F.normalize(x_embed_1, p=2, dim=1, eps=5e-5)*self.k
        N, C, H, W = x_embed_1.shape
        loss = 0
        #if self.training:
        # score = torch.matmul(x_embed_1.permute(0, 2, 3, 1).view((N, H * W, C)),
        #                      x_embed_2.view(N, C, H * W))  # [N,H*W,H*W] 矩阵乘法
        # score = torch.exp(score)  #对输入input逐元素进行以自然数e为底指数运算
        # score = torch.sort(score, dim=2, descending=True)[0]  #排序，由大到小
        # positive = torch.mean(score[:, :, :15], dim=2)  #求平均值
        # negative = torch.mean(score[:, :, 50:65], dim=2)  # [N,H*W]
        # loss = F.relu(-1 * torch.log(positive / (negative + 1e-6))+1)
        # loss = torch.mean(loss)

        x_embed_1 = x_embed_1.permute(0, 2, 3, 1).view(N,1, H * W, C)
        x_embed_2 = x_embed_2.permute(0, 2, 3, 1).view(N,1, H * W, C)
        x_assembly = x_assembly.permute(0, 2, 3, 1).view(N,1, H*W, -1)
        x_final = self.attn_fn(x_embed_1, x_embed_2, x_assembly).squeeze(1)# (1, H*W, C)
        # return x_final.permute(0, 2, 1).view(N, -1, H, W)*self.res_scale+input, loss
        return x_final.permute(0, 2, 1).view(N, -1, H, W) * self.res_scale


class ENLCA_spec(nn.Module):
    def __init__(self, channel=128, reduction=2, ksize=3, scale=3, stride=1, softmax_scale=10, average=True,
                 conv=common.default_conv, res_scale=0.1):
        super(ENLCA_spec, self).__init__()
        self.conv_match1 = common.BasicBlock(conv, channel, channel // reduction, 1, bn=False, act=None)
        self.conv_match2 = common.BasicBlock(conv, channel, channel // reduction, 1, bn=False, act=None)
        self.conv_assembly = common.BasicBlock(conv, channel, channel, 1, bn=False, act=None)
        self.res_scale = res_scale
        self.attn_fn = ENLA(
            dim_heads=channel // reduction,
            nb_features=128,
        )
        self.k = math.sqrt(6)
        n_select_bands = 5
        self.conv_CRAC = nn.Sequential(
            common_1.Conv2d_CG(channel, channel // reduction),
            nn.ReLU(inplace=True),
            # common_1.Conv2d_CG(n_bands, n_bands)
        )
        self.conv_CRAC1 = nn.Sequential(
            common_1.Conv2d_CG(channel, channel),
            nn.ReLU(inplace=True),
            # common_1.Conv2d_CG(n_bands, n_bands)
        )
    def forward(self, input):
        # x_embed_1 =self.conv_CRAC(input)
        # x_embed_2 = self.conv_CRAC(input)
        # x_assembly = self.conv_CRAC1(input)

        x_embed_1 = self.conv_match1(input)
        x_embed_2 = self.conv_match2(input)
        x_assembly = self.conv_assembly(input) #[B,C,H,W]

        x_embed_2 = F.normalize(x_embed_2, p=2, dim=1,eps=5e-5)*self.k     #归一化：将某一个维度除以那个维度对应的范数(默认是2范数)
        x_embed_1 = F.normalize(x_embed_1, p=2, dim=1, eps=5e-5)*self.k
        N, C, H, W = x_embed_1.shape
        loss = 0
        #if self.training:
        # score = torch.matmul(x_embed_1.permute(0, 2, 3, 1).view((N, H * W, C)),
        #                      x_embed_2.view(N, C, H * W))  # [N,H*W,H*W] 矩阵乘法
        # score = torch.exp(score)  #对输入input逐元素进行以自然数e为底指数运算
        # score = torch.sort(score, dim=2, descending=True)[0]  #排序，由大到小
        # positive = torch.mean(score[:, :, :15], dim=2)  #求平均值
        # negative = torch.mean(score[:, :, 50:65], dim=2)  # [N,H*W]
        # loss = F.relu(-1 * torch.log(positive / (negative + 1e-6))+1)
        # loss = torch.mean(loss)

        x_embed_1 = x_embed_1.permute(0, 2, 3, 1).view(N,1, H * W, C)
        x_embed_2 = x_embed_2.permute(0, 2, 3, 1).view(N,1, H * W, C)
        x_assembly = x_assembly.permute(0, 2, 3, 1).view(N,1, H*W, -1)
        x_final = self.attn_fn(x_embed_1, x_embed_2, x_assembly).squeeze(1)# (1, H*W, C)
        return x_final.permute(0, 2, 1).view(N, -1, H, W)*self.res_scale