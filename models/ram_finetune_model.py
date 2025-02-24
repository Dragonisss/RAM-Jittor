import os.path as osp
from collections import OrderedDict

import numpy as np
import jittor as jt
from tqdm import tqdm

from archs import build_network
from metrics import calculate_metric
from utils import get_root_logger, imwrite, tensor2img
from utils.registry import MODEL_REGISTRY

from .ram_base_model import RAMBaseModel


@MODEL_REGISTRY.register()
class RAMFinetuneModel(RAMBaseModel):
    """MIM Stage 2 model for image restoration."""

    def __init__(self, opt):
        super(RAMFinetuneModel, self).__init__(opt)

    def load_pretrained_models(self):
        # 加载预训练的finetune网络
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)
            self.net_g.eval()
        if self.opt['is_train']:
            logger = get_root_logger()
            logger.info('Start setting up finetuning parameters...')
            
            filter_name_list = np.loadtxt(self.opt['finetune_block']['filter_txt_path'], dtype=str)
            filter_ratio = self.opt['finetune_block']['filter_ratio']
            logger.info(f'Loading filter list from {self.opt["finetune_block"]["filter_txt_path"]}')
            logger.info(f'Filter ratio set to: {filter_ratio}')
            
            finetune_name_list = filter_name_list[:int(len(filter_name_list)*filter_ratio)]
            finetune_name_list = [name for name in finetune_name_list]
            logger.info(f'Will finetune {len(finetune_name_list)} modules')
            
            for name,param in self.net_g.named_parameters():
                layer_name = '.'.join(name.split('.')[:-1])
                if layer_name in finetune_name_list:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                    
            logger.info('Finetuning parameter setup completed')

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            self.init_ema_model()

        self.load_pretrained_models()
        self.setup_optimizers()
        self.setup_schedulers()
        self.setup_loss_functions()
        
    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        logger = get_root_logger()
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
                logger.warning(f'Params {k} will be optimized.')
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        super(RAMFinetuneModel, self).feed_data(data)
        self.mask = data['mask'] if 'mask' in data else None

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq, self.mask)

        l_total = 0
        loss_dict = OrderedDict()

        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt, self.mask)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        if self.cri_perceptual:
            l_percep = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep

        self.optimizer_g.backward(l_total)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        with jt.no_grad():
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                self.output = self.net_g_ema(self.lq).stop_grad().cpu()
            else:
                self.net_g.eval()
                self.output = self.net_g(self.lq).stop_grad().cpu()
                self.net_g.train()
        logger = get_root_logger()
        logger.info(f'Test {self.output.shape}')

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, test_num=-1, save_num=-1):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img, test_num, save_num)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, test_num=-1, save_num=-1):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)
        normalize_opt = self.opt['val'].get('normalize', None)
        
        if normalize_opt:
            mean = -1 * jt.array(normalize_opt['mean']) / jt.array(normalize_opt['std'])
            std = 1 / jt.array(normalize_opt['std'])

        if with_metrics:
            if not hasattr(self, 'metric_results'):
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            self._initialize_best_metric_results(dataset_name)
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            if idx >= test_num > 0:
                break
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            if normalize_opt:
                visuals['result'] = (visuals['result'] - mean.reshape(-1,1,1)) / std.reshape(-1,1,1)
                visuals['gt'] = (visuals['gt'] - mean.reshape(-1,1,1)) / std.reshape(-1,1,1)
                visuals['lq'] = (visuals['lq'] - mean.reshape(-1,1,1)) / std.reshape(-1,1,1)

            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            jt.gc()

            if save_img and idx < save_num:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(np.hstack((tensor2img([visuals['lq']]), sr_img, gt_img)), save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')

        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = super(RAMFinetuneModel, self).get_current_visuals()
        if self.mask is not None:
            out_dict['mask'] = self.mask
        return out_dict