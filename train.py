from archs.swinir_arch import SwinIR
from archs.promptir_arch import PromptIR,PixelUnshuffle
from data.ram_dataset import RAMDataset
import jittor as jt
from utils.options import copy_opt_file, dict2str, parse_options
import os.path as osp
import os
import math
from utils.logger import get_root_logger,AvgTimer,MessageLogger
from utils import make_exp_dirs, get_time_str,scandir,check_resume  
from models.ram_pretrain_model import RAMPretrainModel
from models.ram_finetune_model import RAMFinetuneModel
import time
from tqdm import tqdm
import logging

def create_train_val_dataloader(opt):
    # create train and val dataloaders
    train_set = RAMDataset(opt['datasets']['train'])
    print(opt['datasets']['train'])
    dataset_opt = opt['datasets']['train']
    dataset_opt['gt_size'] = opt['gt_size']
    if opt['dist']:  # distributed training
        batch_size = dataset_opt['batch_size_per_gpu']
        num_workers = dataset_opt['num_worker_per_gpu']
    else:  # non-distributed training
        multiplier = 1 if opt['num_gpu'] == 0 else opt['num_gpu']
        batch_size = dataset_opt['batch_size_per_gpu'] * multiplier
        num_workers = dataset_opt['num_worker_per_gpu'] * multiplier
    
    train_loader = jt.dataset.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True)
    
    num_iter_per_epoch = math.ceil(
                len(train_set)  / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
    total_iters = int(opt['train']['total_iter'])
    total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
    print('Training statistics:'
                f'\n\tNumber of train images: {len(train_set)}'
                f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                f'\n\tWorld size (gpu number): {opt["world_size"]}'
                f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
    return train_loader,total_epochs,total_iters

def load_resume_state(opt):
    resume_state_path = None
    if opt['auto_resume']:
        print("auto_resume___")
        state_path = osp.join('experiments', opt['name'], 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt['path']['resume_state'] = resume_state_path
    else:
        if opt['path'].get('resume_state'):
            resume_state_path = opt['path']['resume_state']

    if resume_state_path is None:
        resume_state = None
    else:
        resume_state = jt.load(resume_state_path)
        check_resume(opt, resume_state['iter'])
    return resume_state

def main():
    root_path =root_path = osp.abspath(osp.join(__file__, osp.pardir))
    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path

    # resume_state = None
    resume_state = load_resume_state(opt)
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))


    copy_opt_file(args.opt, opt['path']['experiments_root'])
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='ram', log_level=logging.INFO, log_file=log_file)
    logger.info(dict2str(opt))
    result = create_train_val_dataloader(opt)
    train_loader, total_epochs, total_iters = result

    if opt['model_type'] == 'RAMPretrainModel':
        model = RAMPretrainModel(opt)
    elif opt['model_type'] == 'RAMFinetuneModel':
        model = RAMFinetuneModel(opt)
    
    if resume_state:  # resume training
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        start_epoch = 0
        current_iter = 0

    msg_logger = MessageLogger(opt, current_iter, None)
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()
    for epoch in range(start_epoch, total_epochs + 1):
        for step,train_data in enumerate(train_loader):
            data_timer.record()

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            # training
            model.feed_data(train_data)
            model.optimize_parameters(current_iter)
            iter_timer.record()
            if current_iter == 1:
                # reset start time in msg_logger for more accurate eta_time
                # not work in resume mode
                msg_logger.reset_start_time()
            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)
                if opt['logger']['train_image_visual']:
                    model.save_image(epoch,current_iter)
                    
            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            # validation
            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                if len(val_loaders) > 1:
                    logger.warning('Multiple validation datasets are *only* supported by SRModel.')
                for val_loader in val_loaders:
                    model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])

            data_timer.start()
            iter_timer.start()
        # end of iter

    # end of epoch

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get('val') is not None:
        for val_loader in val_loaders:
            model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    main()