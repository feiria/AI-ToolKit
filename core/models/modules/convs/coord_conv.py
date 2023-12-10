import torch
import torch.nn as nn
from mmcv.cnn import normal_init


class CoordConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 act=False,
                 norm=False):
        super(CoordConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels+2, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups)

        self.act = False
        self.norm = False
        if act:
            self.act = True
            self.act = nn.SiLU(inplace=True)
        if norm:
            self.norm = True
            self.norm = nn.GroupNorm(out_channels, out_channels)
        self._init_weights()

    def forward(self, inputs):
        x_range = torch.linspace(-1, 1, inputs.shape[-1], device=inputs.device)
        y_range = torch.linspace(-1, 1, inputs.shape[-2], device=inputs.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([inputs.shape[0], 1, -1, -1])
        x = x.expand([inputs.shape[0], 1, -1, -1])
        coord_features = torch.cat([x, y], 1)
        out = torch.cat([inputs, coord_features], 1)
        out = self.conv(out)
        if self.norm:
            out = self.norm(out)
        if self.act:
            out = self.act(out)
        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
