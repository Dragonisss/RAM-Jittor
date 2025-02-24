from os import path as osp
from .losses import (L1Loss, MaskL1Loss)
__all__ = ['L1Loss', 'MaskL1Loss']

def build_loss(opt):
    loss_type = opt.pop('type')
    if loss_type == 'L1Loss':
        return L1Loss(**opt)
    elif loss_type == 'MaskL1Loss':
        return MaskL1Loss(**opt)
    else:
        raise ValueError(f'Loss type {loss_type} is not supported.')
