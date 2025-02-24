from copy import deepcopy

from .psnr_ssim import calculate_psnr, calculate_ssim
__all__ = ['calculate_psnr', 'calculate_ssim']


def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    if metric_type == 'calculate_psnr':
        metric = calculate_psnr(**data, **opt)
    elif metric_type == 'calculate_ssim':
        metric = calculate_ssim(**data, **opt)
    else:
        raise ValueError(f"Unsupported metric type: {metric_type}")
    return metric
