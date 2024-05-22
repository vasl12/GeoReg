import torch
from torch import nn
import math
import numpy as np


def normalize(x):
    min_ = x.min()
    max_ = x.max()
    return (x - min_) / (max_ - min_)


def make_coordinate_tensor(dims=(28, 28, 28), gpu=True):
    """Make a coordinate tensor."""

    coordinate_tensor = [torch.linspace(-1, 1, dims[i]) for i in range(len(dims))]
    coordinate_tensor = torch.meshgrid(*coordinate_tensor)
    coordinate_tensor = torch.stack(coordinate_tensor, dim=0)
    # coordinate_tensor = coordinate_tensor.view([torch.prod(torch.tensor(dims)), len(dims)])

    if gpu:
        coordinate_tensor = coordinate_tensor.cuda()

    return coordinate_tensor


class CAPE(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, coord: torch.Tensor, big_k: int):
        k = torch.arange(0, big_k//2, device=coord.device)[None]
        wx = torch.pow(10, 2 * k / big_k) * torch.cos(k)
        wy = torch.pow(10, 2 * k / big_k) * torch.sin(k)
        cos = torch.cos(torch.pi * (coord[..., :1] @ wx + coord[..., 1:] @ wy))
        sin = torch.sin(torch.pi * (coord[..., :1] @ wx + coord[..., 1:] @ wy))
        out = torch.cat((cos, sin), -1)
        return out


def calculate_distances(p1, p2):
    assert len(p1.shape) == len(p2.shape)
    assert p1.shape[-1] == p2.shape[-1]
    dists = p1.unsqueeze(-2) - p2.unsqueeze(-3)
    dists = dists * dists
    dists = dists.sum(-1)
    return dists


def calculate_closest_indices(c1, c2):
    # TODO: double check
    """ We assume the c1 coordinates live in the coordinate space of c2.
    c1 is mapped into the index space of c2, the round and convert to torch.long to obtain
    which c2 indices each c1 coordinate should map to. """
    # We assume the (-1, -1) index will point us to how far it is from the [1.0, 1.0] edge coordinate of the image
    c1_ = c1 * (1.0 / c2[0, -1, -1, :])  # Normalize to [-1.0, 1.0]
    c1_ = (c1_ + 1) / 2  # Normalize to [0.0, 1.0]
    c2_shape = torch.tensor(c2.shape[1:-1], device=c1.device)
    c1_ = c1_ * (c2_shape - 1.0)  # Map to c2 index space
    c1_ = c1_.clip(min=torch.zeros_like(c2_shape), max=(c2_shape - 1.0))
    c1_idx = torch.round(c1_).to(torch.long)  # Round to closest index, then to integer
    return c1_idx



def measure_seg_metrics(fixed_seg, warped_seg):
    """ Calculate segmentation """
    """input shapes (B, C, H, W)"""

    seg_gt = fixed_seg
    seg_pred = warped_seg
    assert seg_gt.ndim == seg_pred.ndim

    results = dict()
    for label_cls in torch.unique(seg_gt):
        # calculate DICE score for each class
        # TODO: remove 5 when 3D
        if label_cls == 0 or label_cls == 5:
            # skip background
            continue
        results[f'dice_class_{label_cls}'] = calculate_dice(seg_gt, seg_pred, label_class=label_cls)

    # calculate mean dice
    results['mean_dice'] = np.mean([dice.cpu().numpy() for k, dice in results.items()])
    return results

def calculate_dice(mask1, mask2, label_class=0):
    """
    Dice score of a specified class between two label masks.
    (classes are encoded but by label class number not one-hot )

    Args:
        mask1: (numpy.array, shape (N, 1, *sizes)) segmentation mask 1
        mask2: (numpy.array, shape (N, 1, *sizes)) segmentation mask 2
        label_class: (int or float)

    Returns:
        volume_dice
    """
    mask1_pos = (mask1 == label_class).type(torch.cuda.FloatTensor)
    mask2_pos = (mask2 == label_class).type(torch.cuda.FloatTensor)

    assert mask1.ndim == mask2.ndim
    axes = tuple(range(2, mask1.ndim))
    pos1and2 = torch.sum(mask1_pos * mask2_pos, axis=axes)
    pos1 = torch.sum(mask1_pos, axis=axes)
    pos2 = torch.sum(mask2_pos, axis=axes)
    return torch.mean(2 * pos1and2 / (pos1 + pos2 + 1e-7))



def fast_trilinear_interpolation(input_array, x_indices, y_indices, z_indices):
    x_indices = (x_indices + 1) * (input_array.shape[0] - 1) * 0.5
    y_indices = (y_indices + 1) * (input_array.shape[1] - 1) * 0.5
    z_indices = (z_indices + 1) * (input_array.shape[2] - 1) * 0.5

    x0 = torch.floor(x_indices.detach()).to(torch.long)
    y0 = torch.floor(y_indices.detach()).to(torch.long)
    z0 = torch.floor(z_indices.detach()).to(torch.long)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    x0 = torch.clamp(x0, 0, input_array.shape[0] - 1)
    y0 = torch.clamp(y0, 0, input_array.shape[1] - 1)
    z0 = torch.clamp(z0, 0, input_array.shape[2] - 1)
    x1 = torch.clamp(x1, 0, input_array.shape[0] - 1)
    y1 = torch.clamp(y1, 0, input_array.shape[1] - 1)
    z1 = torch.clamp(z1, 0, input_array.shape[2] - 1)

    x = x_indices - x0
    y = y_indices - y0
    z = z_indices - z0

    output = (
        input_array[x0, y0, z0] * (1 - x) * (1 - y) * (1 - z)
        + input_array[x1, y0, z0] * x * (1 - y) * (1 - z)
        + input_array[x0, y1, z0] * (1 - x) * y * (1 - z)
        + input_array[x0, y0, z1] * (1 - x) * (1 - y) * z
        + input_array[x1, y0, z1] * x * (1 - y) * z
        + input_array[x0, y1, z1] * (1 - x) * y * z
        + input_array[x1, y1, z0] * x * y * (1 - z)
        + input_array[x1, y1, z1] * x * y * z
    )
    return output
