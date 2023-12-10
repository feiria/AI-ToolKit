import torch
import torch.nn as nn


class BoundaryActivation(nn.Module):
    """
    ICCV 2021
    Towards Real-world X-ray Security Inspection: A High-Quality Benchmark and Lateral Inhibition Module
    for Prohibited Items Detection
    """

    def __init__(self, in_channels):
        super(BoundaryActivation, self).__init__()
        self.reduce_conv = nn.Conv2d(in_channels=in_channels * 5, out_channels=in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.size()
        x1 = self.up_to_bottom(x, h)
        x2 = self.bottom_to_up(x, h)
        x3 = self.left_to_right(x, w)
        x4 = self.right_to_left(x, w)
        x_con = torch.cat((x, x1, x2, x3, x4), 1)
        x_merge = self.reduce_conv(x_con)
        return x_merge

    # 从上到下，每一行各列元素去已扫描过行的当前列最大值
    def up_to_bottom(self, x_raw: torch.tensor, height: int):
        x = torch.clone(x_raw)
        for i in range(height):
            x[:, :, i] = torch.max(x[:, :, :i + 1], 2, True)[0].squeeze(2)
        return x

    # 从下到上，每一行各列元素去已扫描过行的当前列最大值
    def bottom_to_up(self, x_raw: torch.tensor, height: int):
        x = torch.clone(x_raw)
        # 从最后一行一直扫到第一行
        for i in range(height - 1, -1, -1):
            x[:, :, i] = torch.max(x[:, :, i:], 2, True)[0].squeeze(2)
        return x

    # 从左到右，每一行各列元素去已扫描过行的当前列最大值
    def left_to_right(self, x_raw: torch.tensor, height: int):
        # 顺时针旋转90°
        x_clk_rot = torch.rot90(x_raw, -1, [2, 3])
        x = torch.clone(x_clk_rot)
        x = self.up_to_bottom(x, height)
        # 逆时针旋转90°
        x = torch.rot90(x, 1, [2, 3])
        return x

    # 从右到左
    def right_to_left(self, x_raw: torch.tensor, height: int):
        # 顺时针旋转90°
        x_clk_rot = torch.rot90(x_raw, -1, [2, 3])
        x = torch.clone(x_clk_rot)
        x = self.bottom_to_up(x, height)
        # 逆时针旋转90°
        x = torch.rot90(x, 1, [2, 3])
        return x
