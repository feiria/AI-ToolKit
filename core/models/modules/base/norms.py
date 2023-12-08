import inspect
from typing import Dict, Tuple, Union
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm


NORM_LAYERS = {
    "BN": nn.BatchNorm2d,
    "BN1d": nn.BatchNorm1d,
    "BN2d": nn.BatchNorm2d,
    "BN3d": nn.BatchNorm3d,
    "SyncBN": nn.SyncBatchNorm,
    "GN": nn.GroupNorm,
    "LN": nn.LayerNorm,
    "IN": nn.InstanceNorm2d,
    "IN1d": nn.InstanceNorm1d,
    "IN2d": nn.InstanceNorm2d,
    "IN3d": nn.InstanceNorm3d,
}


def build_norm_layer(cfg: Dict,
                     num_features: int,
                     postfix: Union[int, str] = '') -> Tuple[str, nn.Module]:
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in NORM_LAYERS:
        raise KeyError(f'Unrecognized norm type {layer_type}')

    norm_layer = NORM_LAYERS.get(layer_type)
    abbr = infer_abbr(norm_layer)

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
        if layer_type == 'SyncBN' and hasattr(layer, '_specify_ddp_gpu_num'):
            layer._specify_ddp_gpu_num(1)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer


def infer_abbr(class_type):
    if not inspect.isclass(class_type):
        raise TypeError(
            f'class_type must be a type, but got {type(class_type)}')
    if hasattr(class_type, '_abbr_'):
        return class_type._abbr_
    if issubclass(class_type, _InstanceNorm):  # IN is a subclass of BN
        return 'in'
    elif issubclass(class_type, _BatchNorm):
        return 'bn'
    elif issubclass(class_type, nn.GroupNorm):
        return 'gn'
    elif issubclass(class_type, nn.LayerNorm):
        return 'ln'
    else:
        class_name = class_type.__name__.lower()
        if 'batch' in class_name:
            return 'bn'
        elif 'group' in class_name:
            return 'gn'
        elif 'layer' in class_name:
            return 'ln'
        elif 'instance' in class_name:
            return 'in'
        else:
            return 'norm_layer'
