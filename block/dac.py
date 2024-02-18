import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 3.


class SEModule_small(nn.Module):
    def __init__(self, channel):
        super(SEModule_small, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel, bias=False),
            Hsigmoid()
        )

    def forward(self, x):
        y = self.fc(x)
        return x * y


class DACBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, grounps=1,
                 bias=True, K=3, temperature=1):
        super(DACBlock, self).__init__()
        self.K = K
        self.temperature = temperature
        self.padding = (padding, padding)
        self.stride = (stride, stride)
        if padding - kernel_size // 2 >= 0:
            #   Common use case. E.g., k=3, p=1 or k=5, p=2
            self.crop = 0
            #   Compared to the KxK layer, the padding of the 1xK layer and Kx1 layer should be adjust to align the sliding windows (Fig 2 in the paper)
            # [填充的行数，填充的列数]
            hor_padding = [padding - kernel_size // 2, padding]
            ver_padding = [padding, padding - kernel_size // 2]
        else:
            #   A negative "padding" (padding - kernel_size//2 < 0, which is not DTD common use case) is cropping.
            #   Since nn.Conv2d does not support negative padding, we implement it manually
            self.crop = kernel_size // 2 - padding
            hor_padding = [0, padding]
            ver_padding = [padding, 0]
        self.squ_conv = nn.Conv2d(in_channels=in_planes, out_channels=out_planes,
                                  kernel_size=(kernel_size, kernel_size), stride=stride,
                                  padding=padding, bias=bias, dilation=dilation, groups=grounps)
        self.ver_conv = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=(kernel_size, 1),
                                  stride=stride,
                                  padding=ver_padding, bias=bias, dilation=dilation, groups=grounps)
        self.hor_conv = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=(1, kernel_size),
                                  stride=stride,
                                  padding=hor_padding, bias=bias, dilation=dilation, groups=grounps)
        self.dim = int(math.sqrt(in_planes))
        squeeze = max(in_planes, self.dim ** 2) // 16
        if squeeze < 4:
            squeeze = 4
        self.hs = Hsigmoid()

        self.avg_pool = nn.AdaptiveAvgPool2d(2)

        self.fc = nn.Sequential(nn.Linear(in_planes * 4, squeeze, bias=False),
                                SEModule_small(squeeze))

        self.fc_kernel = nn.Linear(squeeze, self.K, bias=False)
        self.fc_channel = nn.Linear(squeeze, in_planes, bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.fc(self.avg_pool(x).view(b, c * 4))
        softmax_attention = F.softmax(self.fc_kernel(y) / self.temperature, -1).view(b, -1, 1, 1, 1)
        x = self.hs(self.fc_channel(y)).view(b, -1, 1, 1) * x

        output = softmax_attention[:, 0] * self.squ_conv(x)
        if self.crop > 0:
            output += softmax_attention[:, 1] * self.ver_conv(x[:, :, :, self.crop:-self.crop])
            output += softmax_attention[:, 2] * self.hor_conv(x[:, :, self.crop:-self.crop, :])
        else:
            output += softmax_attention[:, 1] * self.ver_conv(x)
            output += softmax_attention[:, 2] * self.hor_conv(x)
        return output


if __name__ == '__main__':
    x = torch.randn(5, 512, 14, 14)
    model = DACBlock(in_planes=512, out_planes=512, kernel_size=3, padding=1, grounps=1)
    model(x)
