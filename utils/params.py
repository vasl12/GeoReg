from dataclasses import dataclass
from typing import Tuple, Union, Optional, Iterable

import numpy as np


@dataclass
class Params:
    # Data
    dataset: str = 'brain2d'  # brain, brain2d, lung, mnist
    data_dir: str = '.'
    label_dir: str = '.'
    same_expression: bool = True  # rafd/camcan only. True: same facial expression / t1t1
    spatial_dims: Tuple[int] = (1, 1)
    num_seg_classes: Optional[int] = None
    # Dataloader
    num_train: int = None
    num_val: int = None
    num_test: int = None
    batch_size: int = 8
    num_workers: int = 8
    # Augmentation
    with_affine: bool = False
    with_deform: bool = False
    angle: Union[float, Iterable[float]] = np.pi / 4  # In radians
    scale: Union[float, Iterable[float]] = 0.0
    translation: Union[float, Iterable[float]] = 0.2
    intersubject: bool = True
    # Network
    att_nhead: int = 1
    inh_coord_interp_type: str = 'bspline'  # 'nearest_relative', 'nearest', 'bilinear', 'bicubic', 'mha', 'bspline'
    inh_feat_interp_type: str = 'bilinear'  # 'nearest', 'bilinear', 'bicubic', 'mha', "None"
    inh_refinement: bool = False
    global_bottleneck: bool = False
    channels: tuple = (32, 64, 64)
    pool_sizes: tuple = (2, 2)  # tuple treats each layer differently (fine -> coarse)
    # Same pattern as channels: finest to coarsest
    k_size_inh: int = 5  # tuple treats each layer differently (fine -> coarse)
    k_size_cross: Union[Iterable[float], int] = (5, 3, 3)  # tuple treats each layer differently (fine -> coarse)
    k_size_sibling: Union[Iterable[float], int] = (3, 3, 3)  # tuple treats each layer differently (fine -> coarse)
    k_size_parent: Union[Iterable[float], int] = (3, 3)  # tuple treats each layer differently (fine -> coarse)
    pos_econding: str = "cape"  # fourier, cape, nerf
    # wandb
    wandb_disable: str = "False"
    is_sweep: bool = False
    # gpu
    gpu: int = 0
    # training
    lr: float = 1e-4
    dropout: float = 0.0
    # Trainer
    check_val_every_n_epoch: int = 10
    max_epochs: int = 10000
    # Regularisation
    # Loss on each coarser resolution is compensated by its pooling factor
    reg_order0_alpha: Union[Iterable[float], float] = 0e0  # tuple treats each layer differently (fine -> coarse)
    reg_order0_type: str = 'l2'
    reg_order2_alpha: Union[Iterable[float], float] = 5e2  # tuple treats each layer differently (fine -> coarse)
    reg_order2_type: str = 'l2'
    reg_order2_from_start: bool = True
    reg_order2_detach: bool = False
    reg_consist_alpha: Union[Iterable[float], float] = 0.0  # tuple treats each layer differently (fine -> coarse)
    # loss
    loss_function: str = 'mse'
    window_size: int = 9
    loss_type: str = 'pixel'  # pixel, feat, disp
    loss_weight: Union[Tuple[float], float] = 1.0  # tuple treats each layer differently (fine -> coarse)
    evaluation_metrics: Tuple[str] = ("disp_metrics", "image_metrics", "seg_metrics") # disp_metrics, image_metrics, seg_metrics, landmark_metrics
    iters: Union[Tuple[float], int] = 1  # tuple treats each layer differently (fine -> coarse)

    ckpt_path: str = 'path/to/checkpoint.pth'
    current_dir: str = None


def process_params(params: Params):
    if isinstance(params.reg_order0_alpha, (float, int)):
        params.reg_order0_alpha = [params.reg_order0_alpha] * len(params.channels)
    assert len(params.reg_order0_alpha) == len(params.channels), "Number of reg_order0_alpha must match number of channels"
    params.reg_order0_alpha = [float(i) for i in params.reg_order0_alpha]

    if isinstance(params.reg_order2_alpha, (float, int)):
        params.reg_order2_alpha = [params.reg_order2_alpha] * len(params.channels)
    assert len(params.reg_order2_alpha) == len(params.channels), "Number of reg_order2_alpha must match number of channels"
    params.reg_order2_alpha = [float(i) for i in params.reg_order2_alpha]

    if isinstance(params.reg_consist_alpha, (float, int)):
        params.reg_consist_alpha = [params.reg_consist_alpha] * (len(params.channels) - 1)
    assert len(params.reg_consist_alpha) == len(params.channels) - 1, "Number of reg_consist_alpha must match number of (channels - 1)"
    params.reg_consist_alpha = [float(i) for i in params.reg_consist_alpha]

    if isinstance(params.loss_weight, (float, int)):
        params.loss_weight = [params.loss_weight] * len(params.channels)
    assert len(params.loss_weight) == len(params.channels), "Number of loss_weight must match number of channels"
    params.loss_weight = [float(i) for i in params.loss_weight]
    assert params.loss_weight[0] > 0.0, "Loss weight for finest resolution must be greater than 0"

    if isinstance(params.iters, int):
        params.iters = [params.iters] * len(params.channels)
    assert len(params.iters) == len(params.channels), "Number of iters must match number of channels"

    if isinstance(params.k_size_cross, int):
        params.k_size_cross = [params.k_size_cross] * len(params.channels)
    assert len(params.k_size_cross) == len(params.channels), "Number of k_size_cross must match number of channels"

    if isinstance(params.k_size_sibling, int):
        params.k_size_sibling = [params.k_size_sibling] * len(params.channels)
    assert len(params.k_size_sibling) == len(params.channels), "Number of k_size_sibling must match number of channels"
    assert all([(i % 2 == 1) or (i == 0) for i in params.k_size_sibling]), "Even neighbourhoods are illegal. The kernel needs to be centere on self."

    if isinstance(params.k_size_parent, int):
        params.k_size_parent = [params.k_size_parent] * (len(params.channels) - 1)
    assert len(params.k_size_parent) == len(params.channels) - 1, "Number of k_size_parent must match number of (channels - 1)"

    if isinstance(params.k_size_inh, int):
        params.k_size_inh = [params.k_size_inh] * (len(params.channels) - 1)
    assert len(params.k_size_inh) == len(params.channels) - 1, "Number of k_size_inh must match number of (channels - 1)"

    if isinstance(params.pool_sizes, int):
        params.pool_sizes = [params.pool_sizes] * (len(params.channels) - 1)
    assert len(params.pool_sizes) == len(params.channels) - 1, "Number of pool_sizes must match number of (channels - 1)"

    # Make sure we do not have tuples otherwise saving to yaml will not be handled.
    for field in params.__dataclass_fields__:
        value = getattr(params, field)
        if isinstance(value, tuple):
            value = list(value)
            setattr(params, field, value)
