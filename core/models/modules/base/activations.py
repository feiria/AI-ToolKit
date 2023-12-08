from typing import Dict

from torch import nn

ACTIVATION_LAYERS = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "PReLU": nn.PReLU,
    "ReLU6": nn.ReLU6,
    "ELU": nn.ELU,
    "Sigmoid": nn.Sigmoid,
    "Tanh": nn.Tanh,
    "HardSwish": nn.Hardswish,
    "HardSigmoid": nn.Hardsigmoid,
}


def build_activation_layer(cfg: Dict) -> nn.Module:
    if cfg is None:
        cfg_ = dict(type='ReLU')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in ACTIVATION_LAYERS:
        raise KeyError(f'Unrecognized layer type {layer_type}')
    else:
        activation_layer = ACTIVATION_LAYERS.get(layer_type)

    layer = activation_layer(**cfg_)

    return layer