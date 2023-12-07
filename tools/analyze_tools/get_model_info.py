import sys
import warnings
from functools import partial
from typing import Any, Callable, Dict, Optional, TextIO, Tuple

import numpy as np
import torch
import torch.nn as nn


def get_model_complexity_info(model: nn.Module,
                              input_shape: tuple,
                              print_per_layer_stat: bool = True,
                              as_strings: bool = True,
                              input_constructor: Optional[Callable] = None,
                              flush: bool = False,
                              ost: TextIO = sys.stdout) -> tuple:

    assert type(input_shape) is tuple
    assert len(input_shape) >= 1
    assert isinstance(model, nn.Module)

    flops_model = add_flops_counting_methods(model)
    flops_model.eval()
    flops_model.start_flops_count()
    if input_constructor:
        input = input_constructor(input_shape)
        _ = flops_model(**input)
    else:
        try:
            batch = torch.ones(()).new_empty(
                (1, *input_shape),
                dtype=next(flops_model.parameters()).dtype,
                device=next(flops_model.parameters()).device)
        except StopIteration:
            # Avoid StopIteration for models which have no parameters,
            # like `nn.Relu()`, `nn.AvgPool2d`, etc.
            batch = torch.ones(()).new_empty((1, *input_shape))

        _ = flops_model(batch)

    flops_count, params_count = flops_model.compute_average_flops_cost()
    if print_per_layer_stat:
        print_model_with_flops(
            flops_model, flops_count, params_count, ost=ost, flush=flush)
    flops_model.stop_flops_count()

    if as_strings:
        return flops_to_string(flops_count), params_to_string(params_count)

    return flops_count, params_count


def flops_to_string(flops: float,
                    units: Optional[str] = 'GFLOPs',
                    precision: int = 2) -> str:
    if units is None:
        if flops // 10**9 > 0:
            return str(round(flops / 10.**9, precision)) + ' GFLOPs'
        elif flops // 10**6 > 0:
            return str(round(flops / 10.**6, precision)) + ' MFLOPs'
        elif flops // 10**3 > 0:
            return str(round(flops / 10.**3, precision)) + ' KFLOPs'
        else:
            return str(flops) + ' FLOPs'
    else:
        if units == 'GFLOPs':
            return str(round(flops / 10.**9, precision)) + ' ' + units
        elif units == 'MFLOPs':
            return str(round(flops / 10.**6, precision)) + ' ' + units
        elif units == 'KFLOPs':
            return str(round(flops / 10.**3, precision)) + ' ' + units
        else:
            return str(flops) + ' FLOPs'


def params_to_string(num_params: float,
                     units: Optional[str] = None,
                     precision: int = 2) -> str:
    if units is None:
        if num_params // 10**6 > 0:
            return str(round(num_params / 10**6, precision)) + ' M'
        elif num_params // 10**3:
            return str(round(num_params / 10**3, precision)) + ' k'
        else:
            return str(num_params)
    else:
        if units == 'M':
            return str(round(num_params / 10.**6, precision)) + ' ' + units
        elif units == 'K':
            return str(round(num_params / 10.**3, precision)) + ' ' + units
        else:
            return str(num_params)


def print_model_with_flops(model: nn.Module,
                           total_flops: float,
                           total_params: float,
                           units: Optional[str] = 'GFLOPs',
                           precision: int = 3,
                           ost: TextIO = sys.stdout,
                           flush: bool = False) -> None:

    def accumulate_params(self):
        if is_supported_instance(self):
            return self.__params__
        else:
            sum = 0
            for m in self.children():
                sum += m.accumulate_params()
            return sum

    def accumulate_flops(self):
        if is_supported_instance(self):
            return self.__flops__ / model.__batch_counter__
        else:
            sum = 0
            for m in self.children():
                sum += m.accumulate_flops()
            return sum

    def flops_repr(self):
        accumulated_num_params = self.accumulate_params()
        accumulated_flops_cost = self.accumulate_flops()
        return ', '.join([
            params_to_string(
                accumulated_num_params, units='M', precision=precision),
            f'{accumulated_num_params / total_params:.3%} Params',
            flops_to_string(
                accumulated_flops_cost, units=units, precision=precision),
            f'{accumulated_flops_cost / total_flops:.3%} FLOPs',
            self.original_extra_repr()
        ])

    def add_extra_repr(m):
        m.accumulate_flops = accumulate_flops.__get__(m)
        m.accumulate_params = accumulate_params.__get__(m)
        flops_extra_repr = flops_repr.__get__(m)
        if m.extra_repr != flops_extra_repr:
            m.original_extra_repr = m.extra_repr
            m.extra_repr = flops_extra_repr
            assert m.extra_repr != m.original_extra_repr

    def del_extra_repr(m):
        if hasattr(m, 'original_extra_repr'):
            m.extra_repr = m.original_extra_repr
            del m.original_extra_repr
        if hasattr(m, 'accumulate_flops'):
            del m.accumulate_flops

    model.apply(add_extra_repr)
    print(model, file=ost, flush=flush)
    model.apply(del_extra_repr)


def get_model_parameters_number(model: nn.Module) -> float:
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params


def add_flops_counting_methods(net_main_module: nn.Module) -> nn.Module:
    net_main_module.start_flops_count = start_flops_count.__get__(
        net_main_module)
    net_main_module.stop_flops_count = stop_flops_count.__get__(
        net_main_module)
    net_main_module.reset_flops_count = reset_flops_count.__get__(
        net_main_module)
    net_main_module.compute_average_flops_cost = compute_average_flops_cost.__get__(
        net_main_module)

    net_main_module.reset_flops_count()

    return net_main_module

def start_flops_count(self) -> None:
    add_batch_counter_hook_function(self)

    def add_flops_counter_hook_function(module: nn.Module) -> None:
        if is_supported_instance(module):
            if hasattr(module, '__flops_handle__'):
                return
            else:
                handle = module.register_forward_hook(
                    get_modules_mapping()[type(module)])

            module.__flops_handle__ = handle

    self.apply(partial(add_flops_counter_hook_function))


def stop_flops_count(self) -> None:
    remove_batch_counter_hook_function(self)
    self.apply(remove_flops_counter_hook_function)


def reset_flops_count(self) -> None:
    add_batch_counter_variables_or_reset(self)
    self.apply(add_flops_counter_variable_or_reset)


def compute_average_flops_cost(self) -> Tuple[float, float]:
    batches_count = self.__batch_counter__
    flops_sum = 0
    for module in self.modules():
        if is_supported_instance(module):
            flops_sum += module.__flops__
    params_sum = get_model_parameters_number(self)
    return flops_sum / batches_count, params_sum