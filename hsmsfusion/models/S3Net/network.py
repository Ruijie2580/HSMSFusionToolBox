import torch
import torch.nn as nn
import torch.nn.functional as F
from . import common
from . import attention


class S3_Net(nn.Module):
    def __init__(
        self,
        scale_ratio,
        n_select_bands,
        n_bands,
        arch="S3_Net",
    ):
        """Load the pretrained ResNet and replace top fc layer."""
        super(S3_Net, self).__init__()
        self.scale_ratio = scale_ratio
        self.n_bands = n_bands
        self.arch = arch
        self.n_select_bands = n_select_bands
        self.weight = nn.Parameter(torch.tensor([0.5]))

        self.conv_fus = nn.Sequential(
            nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv_spat = nn.Sequential(
            nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.tail = nn.Sequential(
            nn.Conv2d(n_bands * 3, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.tail1 = nn.Sequential(
            nn.Conv2d(n_bands * 3, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.conv1 = nn.Conv2d(
            n_bands + n_select_bands, n_bands, kernel_size=3, stride=1, padding=1
        )
        self.upsampler = nn.Sequential(
            nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        conv = common.default_conv
        # define head module
        n_feats = n_bands
        kernel_size = 3
        in_channels = n_bands
        res_scale = 1
        m_head = [conv(in_channels, n_feats, kernel_size)]
        m_body = [attention.ENLCA(channel=n_bands, reduction=4, res_scale=res_scale)]

        m_body_spec = [
            attention.ENLCA_spec(channel=n_bands, reduction=4, res_scale=res_scale)
        ]
        self.head = nn.Sequential(*m_head)
        self.body = nn.ModuleList(m_body)
        self.body_spec = nn.ModuleList(m_body_spec)

    def lrhr_interpolate(self, x_lr, x_hr):
        x_lr = F.interpolate(x_lr, scale_factor=self.scale_ratio, mode="bilinear")
        gap_bands = self.n_bands / (self.n_select_bands - 1.0)
        for i in range(0, self.n_select_bands - 1):
            x_lr[:, int(gap_bands * i), ::] = x_hr[:, i, ::]
        x_lr[:, int(self.n_bands - 1), ::] = x_hr[:, self.n_select_bands - 1, ::]

        return x_lr

    def forward(self, x_lr, x_hr):
        if self.arch == "S3_Net":
            x = F.interpolate(x_lr, scale_factor=self.scale_ratio, mode="bilinear")
            x = torch.cat((x, x_hr), 1)
            x = self.conv1(x)
            res = x
            for i in range(1):
                if i % 8 == 0:
                    res = self.body[i](res)
                else:
                    res = self.body[i](res)

            ###################################################
            # body_spec (ENLCA_spec) uses 1×1 convs — operates on NCHW directly,
            # no permutation needed (original permute broke channel dim).
            res1 = x
            for i in range(1):
                res1 = self.body_spec[i](res1)

            res2 = x
            for i in range(1):
                res2 = self.body_spec[i](res2)

            y = torch.cat((res, res1, res2), dim=1)
            y = self.tail(y)

            # #############################2#################################
            res = y
            for i in range(1):
                res = self.body[i](res)
            ##################################################
            res1 = y
            for i in range(1):
                res1 = self.body_spec[i](res1)

            res2 = y
            for i in range(1):
                res2 = self.body_spec[i](res2)
            z = torch.cat((res, res1, res2), dim=1)
            z = self.tail1(z)
        return z


if __name__ == "__main__":
    # import os
    # os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # input_tensor = torch.rand(6, 3, 256, 256)
    from thop import profile

    input_tensor1 = torch.rand(1, 144, 32, 32).cuda()
    input_tensor2 = torch.rand(1, 5, 128, 128).cuda()
    import args_parser

    args = args_parser.args_parser()
    model = S3_Net(args.arch, 4, 5, 144).cuda()
    # model = nn.DataParallel(model).cuda()
    with torch.no_grad():
        output_tensor = model(input_tensor1, input_tensor2)
    # print(output_tensor.shape)
    macs, params = profile(model, inputs=(input_tensor1, input_tensor2))
    print("Parameters number is {}; Flops: {}".format(params, macs))
    print("Parameters number is ", sum(param.numel() for param in model.parameters()))

    print(torch.__version__)
