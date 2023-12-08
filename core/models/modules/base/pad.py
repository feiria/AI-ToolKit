from typing import Dict

from torch import nn

PADDING_LAYERS = {
    "zero": nn.ZeroPad2d,
    "reflect": nn.ReflectionPad2d,
    "replicate": nn.ReplicationPad2d
}


def build_padding_layer(cfg: Dict, *args, **kwargs) -> nn.Module:
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')

    cfg_ = cfg.copy()
    padding_type = cfg_.pop('type')
    if padding_type not in PADDING_LAYERS:
        raise KeyError(f'Unrecognized padding type {padding_type}.')
    else:
        padding_layer = PADDING_LAYERS.get(padding_type)

    layer = padding_layer(*args, **kwargs, **cfg_)

    return layer
