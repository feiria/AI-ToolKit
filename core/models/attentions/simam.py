import torch
import torch.nn as nn


class SimAM(torch.nn.Module):
    """
        SimAM: A simple, parameter-free attention module for convolutional neural networks
    """

    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '(' + ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "SimAM"

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1

        d = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = d / (4 * (d.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activation(y)
