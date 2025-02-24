
import importlib
from copy import deepcopy
from os import path as osp

from utils import get_root_logger, scandir
from .swinir_arch import SwinIR
from .promptir_arch import PromptIR
def build_network(opt,is_print=True):
    opt = deepcopy(opt)
    network_type = opt.pop('type')


    if network_type == 'SwinIR':
        net = SwinIR(**opt)
    elif network_type == 'PromptIR':
        net = PromptIR(**opt)
    else:
        raise ValueError(f'Network type {network_type} is not supported.')
    logger = get_root_logger()
    if is_print:
        logger.info(f'Network [{net.__class__.__name__}] is created.')
    return net
