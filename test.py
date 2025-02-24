import logging
import jittor as jt
from os import path as osp

from data import build_dataset
from utils import get_root_logger, get_time_str, make_exp_dirs
from utils.options import dict2str, parse_options
from models.ram_finetune_model import RAMFinetuneModel


def test_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)

    jt.flags.use_cuda = opt['num_gpu'] != 0

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='ram', log_level=logging.INFO, log_file=log_file)
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        dataset_opt['gt_size'] = opt['gt_size']
        test_set = build_dataset(dataset_opt)
        test_loader = jt.dataset.DataLoader(
            test_set,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = RAMFinetuneModel(opt)
    
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'],test_num=opt.get('test_num',-1),save_num=opt.get('save_num',-1))


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
