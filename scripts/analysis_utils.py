
import logging
import math
import os
from abc import abstractmethod
from os import path as osp

import numpy as np
import jittor as jt
from jittor import nn

from archs import build_network
from data import build_dataset
from models.base_model import BaseModel
from utils import (get_root_logger, get_time_str,
                       imwrite, make_exp_dirs, tensor2img)
from utils.options import dict2str

def get_mask(alpha, order_array, h, w):
    """Generate a binary mask based on alpha value."""
    mask_count = int(np.ceil(len(order_array) * alpha))
    mask_idx = order_array[:mask_count]
    mask = np.zeros(len(order_array), dtype=int)
    mask[mask_idx] = 1
    mask = mask.reshape(h, w)
    mask = jt.array(mask).float()
    return mask

def get_soft_mask(alpha, order_array, h, w, k=100):
    """Generate a soft mask based on alpha value using sigmoid approximation."""
    mask_count = int(np.ceil(len(order_array) * alpha))
    mask_idx = order_array[:mask_count]
    mask = np.zeros(len(order_array), dtype=float)
    
    # Sigmoid approximation
    for i, idx in enumerate(mask_idx):
        mask[idx] = 1 / (1 + math.exp(-1 * k * (alpha - i / len(order_array))))
    
    mask = mask.reshape(h, w)
    mask = jt.array(mask).float()
    return mask

def reduce_func(method):
    """Return the corresponding reduction function."""
    if method == 'sum':
        return jt.sum
    elif method == 'mean':
        return jt.mean
    elif method == 'count':
        return lambda x: sum(x.size())
    else:
        raise NotImplementedError()

def attr_grad(tensor, reduce='sum'):
    """Calculate attribute gradient."""
    h_x = tensor.size()[2]
    w_x = tensor.size()[3]
    h_grad = jt.pow(tensor[:, :, :h_x - 1, :] - tensor[:, :, 1:, :], 2)
    w_grad = jt.pow(tensor[:, :, :, :w_x - 1] - tensor[:, :, :, 1:], 2)
    grad = jt.pow(h_grad[:, :, :, :-1] + w_grad[:, :, :-1, :], 1 / 2)
    return reduce_func(reduce)(grad)

def attr_gabor_generator(gabor_filter):
    """Generate attribute gabor function."""
    filter = jt.array(gabor_filter).view((1, 1,) + gabor_filter.shape).repeat(1, 3, 1, 1)
    
    def attr_gabor(tensor, h, w, window=8, reduce='sum'):
        after_filter = nn.conv2d(tensor, filter, bias=None)
        crop = after_filter[:, :, h: h + window, w: w + window]
        return reduce_func(reduce)(crop)
    
    return attr_gabor

def save(tensor, step):
    """Save tensor as image."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]   
    mean_out = -1 * jt.array(mean) / jt.array(std)
    std_out = 1 / jt.array(std)
    output = normalize(tensor, mean_out, std_out)
    output = tensor2img(output)
    imwrite(output, f"result/cond/test/{step}.png")        

class Hook_back_loop:
    """钩子类,用于捕获前向和后向信息"""
    def __init__(self, module, module_name):
        self.name = module_name
        self.hook = module.register_forward_hook(self.forward_hook)
    
    def forward_hook(self, module, inp, out):    
        self.input = inp
        self.output = out
        # jittor中需要手动设置grad_fn来捕获梯度
        self.output.register_hook(self.backward_hook)
    
    def backward_hook(self, grad):
        self.grad = grad
        return grad
    
class BaseAnalysis(BaseModel):
    def __init__(self, opt):
        self.opt = opt
        jt.flags.use_cuda = jt.has_cuda
        self.setup_environment()
        self._load_model()
        self.hook_list = self._register_hooks()
        self.test_loaders = self.build_test_loaders()
        self.optimizer = jt.optim.Adam(self.model.parameters(), lr=0.001)
    def setup_environment(self):
        # mkdir and initialize loggers
        make_exp_dirs(self.opt)
        log_file = osp.join(self.opt['path']['log'], f"test_{self.opt['name']}_{get_time_str()}.log")
        self.logger = get_root_logger(logger_name='ram', log_level=logging.INFO, log_file=log_file)
        self.logger.info(dict2str(self.opt))
        
    
    def _load_model(self):
        self.model = build_network(self.opt['network_g'], False)
        # self.model.eval()
        load_path = self.opt['path'].get('pretrain_network_g')
        if load_path:
            self.load_network(self.model, load_path, strict=False)

    def _register_hooks(self):
        module_name_list = []
        hook_list = []
        name_list = []
        def get_module_from_name(name):
            name_parts = name.split('.')[:-1]
            module_name = 'self.model'

            for part in name_parts:
                if part == 'mask_token' or part == 'weight' or part == 'bias':
                    continue
                if part.isdigit():
                    module_name += f'[{part}]'
                else:
                    module_name += f'.{part}'
            return module_name, '.'.join(name_parts)

        for name,param in self.model.named_parameters():
            module_name,name = get_module_from_name(name)

        
            if module_name != 'self.model':
                if len(module_name_list)==0 or module_name_list[-1] != module_name:
                    module_name_list.append(module_name)
                    name_list.append(name)
                    module = eval(module_name)
                    hook_list.append(Hook_back_loop(module, name))

        return hook_list

    def build_test_loaders(self):
        test_loaders = []
        for _, dataset_opt in sorted(self.opt['datasets'].items()):
            dataset_opt['gt_size'] = self.opt['gt_size']
            test_set = build_dataset(dataset_opt)
            test_loader = jt.dataset.DataLoader(test_set, batch_size=1, shuffle=False)
            test_loaders.append(test_loader)
        return test_loaders
    
    @abstractmethod
    def analyze(self):
        pass

    def _get_interpolated_img_from_mask_attribute_path(self, base_img, final_img, alpha, order_array):
        mask = get_soft_mask(alpha, order_array, *final_img.shape[2:], k=self.opt.get('k', 10000))
        interpolated_img = base_img + mask * (final_img - base_img)
        interpolated_img.requires_grad = True
        return interpolated_img

    def _get_interpolated_img_from_linear_path(self, base_img, final_img, alpha, order_array):
        interpolated_img = base_img + alpha * (final_img - base_img)
        interpolated_img.requires_grad = True
        return interpolated_img

    def _save_results(self, data, folder_name):
        sorted_data = np.sort(data)[::-1]
        sorted_location = np.argsort(data)[::-1]
        sorted_name = np.array([hook.name for hook in self.hook_list])[sorted_location]
        
        save_folder = f"./options/{folder_name}/{self.opt['name']}"
        os.makedirs(save_folder, exist_ok=True)
        np.savetxt(f"{save_folder}/filter_{folder_name}.txt", sorted_data, delimiter=',', fmt='%f')
        np.savetxt(f"{save_folder}/filter_index.txt", sorted_location, delimiter=',', fmt='%d')
        np.savetxt(f"{save_folder}/filter_name.txt", sorted_name, delimiter=',', fmt='%s')


