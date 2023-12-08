import pytest
import torch
from torch import nn

from core.models.modules.base import ConvModule


def test_conv_module():
    with pytest.raises(AssertionError):
        # conv_cfg must be a dict or None
        conv_cfg = 'conv'
        ConvModule(3, 8, 2, conv_cfg=conv_cfg)

    with pytest.raises(AssertionError):
        # norm_cfg must be a dict or None
        norm_cfg = 'norm'
        ConvModule(3, 8, 2, norm_cfg=norm_cfg)

    with pytest.raises(KeyError):
        # softmax is not supported
        act_cfg = dict(type='softmax')
        ConvModule(3, 8, 2, act_cfg=act_cfg)

    conv = ConvModule(3, 8, 2, norm_cfg=dict(type='BN'))
    assert conv.with_activation
    assert hasattr(conv, 'activate')
    assert conv.with_norm
    assert hasattr(conv, 'norm')
    x = torch.rand(1, 3, 256, 256)
    output = conv(x)
    assert output.shape == (1, 8, 255, 255)

    # conv + act
    conv = ConvModule(3, 8, 2)
    assert conv.with_activation
    assert hasattr(conv, 'activate')
    assert not conv.with_norm
    assert conv.norm is None
    x = torch.rand(1, 3, 256, 256)
    output = conv(x)
    assert output.shape == (1, 8, 255, 255)

    # conv
    conv = ConvModule(3, 8, 2, act_cfg=None)
    assert not conv.with_norm
    assert conv.norm is None
    assert not conv.with_activation
    assert not hasattr(conv, 'activate')
    x = torch.rand(1, 3, 256, 256)
    output = conv(x)
    assert output.shape == (1, 8, 255, 255)

    # with_spectral_norm=True
    conv = ConvModule(3, 8, 3, padding=1, with_spectral_norm=True)
    assert hasattr(conv.conv, 'weight_orig')
    output = conv(x)
    assert output.shape == (1, 8, 256, 256)

    # padding_mode='reflect'
    conv = ConvModule(3, 8, 3, padding=1, padding_mode='reflect')
    assert isinstance(conv.padding_layer, nn.ReflectionPad2d)
    output = conv(x)
    assert output.shape == (1, 8, 256, 256)

    # leaky relu
    conv = ConvModule(3, 8, 3, padding=1, act_cfg=dict(type='LeakyReLU'))
    assert isinstance(conv.activate, nn.LeakyReLU)
    output = conv(x)
    assert output.shape == (1, 8, 256, 256)

    # tanh
    conv = ConvModule(3, 8, 3, padding=1, act_cfg=dict(type='Tanh'))
    assert isinstance(conv.activate, nn.Tanh)
    output = conv(x)
    assert output.shape == (1, 8, 256, 256)

    # Sigmoid
    conv = ConvModule(3, 8, 3, padding=1, act_cfg=dict(type='Sigmoid'))
    assert isinstance(conv.activate, nn.Sigmoid)
    output = conv(x)
    assert output.shape == (1, 8, 256, 256)

    # PReLU
    conv = ConvModule(3, 8, 3, padding=1, act_cfg=dict(type='PReLU'))
    assert isinstance(conv.activate, nn.PReLU)
    output = conv(x)
    assert output.shape == (1, 8, 256, 256)

    conv = ConvModule(3, 8, 3, padding=1, act_cfg=dict(type='HardSwish'))
    assert isinstance(conv.activate, nn.Hardswish)
    output = conv(x)
    assert output.shape == (1, 8, 256, 256)

    # HSigmoid
    conv = ConvModule(3, 8, 3, padding=1, act_cfg=dict(type='HardSigmoid'))
    assert isinstance(conv.activate, nn.Hardsigmoid)
    output = conv(x)
    assert output.shape == (1, 8, 256, 256)
