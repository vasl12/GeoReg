import math
from typing import Tuple, List, Union, Optional, Iterable
import torch
import numpy as np


def params_to_mat_3d(rotation: torch.Tensor, translation: torch.Tensor, scaling: torch.Tensor):
    cos = torch.cos(rotation)
    sin = torch.sin(rotation)
    rotation_z = torch.eye(3, dtype=rotation.dtype, device=rotation.device).repeat((rotation.shape[0], 1, 1))
    rotation_z[:, 0, 0] = cos[:, 0]
    rotation_z[:, 0, 1] = -sin[:, 0]
    rotation_z[:, 1, 0] = sin[:, 0]
    rotation_z[:, 1, 1] = cos[:, 0]

    rotation_y = torch.eye(3, dtype=rotation.dtype, device=rotation.device).repeat((rotation.shape[0], 1, 1))
    rotation_y[:, 0, 0] = cos[:, 1]
    rotation_y[:, 0, 2] = sin[:, 1]
    rotation_y[:, 2, 0] = -sin[:, 1]
    rotation_y[:, 2, 2] = cos[:, 1]

    rotation_x = torch.eye(3, dtype=rotation.dtype, device=rotation.device).repeat((rotation.shape[0], 1, 1))
    rotation_x[:, 1, 1] = cos[:, 2]
    rotation_x[:, 1, 2] = -sin[:, 2]
    rotation_x[:, 2, 1] = sin[:, 2]
    rotation_x[:, 2, 2] = cos[:, 2]

    rotation = torch.bmm(rotation_x, rotation_y)
    rotation = torch.bmm(rotation, rotation_z)

    scaling = torch.diag_embed(scaling)
    rotation = torch.bmm(rotation, scaling)

    affines = torch.eye(4, dtype=rotation.dtype, device=rotation.device).repeat((rotation.shape[0], 1, 1))
    affines[:, :3, :3] = rotation
    affines[:, :3, 3] = translation
    return affines


def params_to_mat_2d(rotation: torch.Tensor, translation: torch.Tensor, scaling: torch.Tensor):
    cos = torch.cos(rotation)
    sin = torch.sin(rotation)
    rotation = torch.eye(2, dtype=rotation.dtype, device=rotation.device).repeat((rotation.shape[0], 1, 1))
    rotation[:, 0, 0] = cos[:, 0]
    rotation[:, 0, 1] = -sin[:, 0]
    rotation[:, 1, 0] = sin[:, 0]
    rotation[:, 1, 1] = cos[:, 0]

    scaling = torch.diag_embed(scaling)
    rotation = torch.bmm(rotation, scaling)

    affines = torch.eye(3, dtype=rotation.dtype, device=rotation.device).repeat((rotation.shape[0], 1, 1))
    affines[:, :2, :2] = rotation
    affines[:, :2, 2] = translation
    return affines


def random_affine_displacements_3d(shape: Tuple[int],
                                   theta_range: Union[float, Tuple[float, ...]] = (math.pi, math.pi, math.pi),
                                   trans_range: Union[float, Tuple[float, ...]] = (0.1, 0.1, 0.1),
                                   scale_range: Union[float, Tuple[float, ...]] = (0.05, 0.05, 0.05),
                                   ) -> torch.Tensor:
    if isinstance(theta_range, (float, int)):
        theta_range = [theta_range]*3
    if isinstance(trans_range, (float, int)):
        trans_range = [trans_range]*3
    if isinstance(scale_range, (float, int)):
        scale_range = [scale_range]*3
    if len(theta_range) == 2:
        raise ValueError("Give me either 1 element or 3!")
    if len(trans_range) == 2:
        raise ValueError("Give me either 1 element or 3!")
    if len(scale_range) == 2:
        raise ValueError("Give me either 1 element or 3!")
    if len(theta_range) == 1:
        theta_range = [theta_range[0]]*3
    if len(trans_range) == 1:
        trans_range = [trans_range[0]]*3
    if len(scale_range) == 1:
        scale_range = [scale_range[0]]*3

    batch_size, channel_dim, spatial_dims = shape[0], shape[1], shape[2:]
    theta_range = torch.tensor(theta_range)
    trans_range = torch.tensor(trans_range)
    scale_range = torch.tensor(scale_range)
    coord_start = torch.meshgrid([torch.linspace(-1., 1., i) for i in spatial_dims], indexing="ij")
    rotation = torch.rand((shape[0], 3)) * theta_range * 2 - theta_range
    translation = torch.rand((shape[0], 3)) * trans_range * 2 - trans_range
    scaling = 1 + torch.rand((shape[0], 3)) * scale_range * 2 - scale_range
    affines = params_to_mat_3d(rotation, translation, scaling)
    aug_dim = torch.ones_like(coord_start[0])
    coord_start_aug = torch.stack([*coord_start, aug_dim], -1)
    coord_start_aug = coord_start_aug.tile((batch_size, *[1 for _ in coord_start_aug.shape]))
    coord_start_aug_ = coord_start_aug.reshape((batch_size, -1, coord_start_aug.shape[-1]))
    coord_end_aug_ = torch.bmm(affines, coord_start_aug_.transpose(1, 2)).transpose(1, 2)
    coord_end_aug = coord_end_aug_.reshape(coord_start_aug.shape)
    displ_aug = coord_end_aug - coord_start_aug
    displ = displ_aug[..., :len(spatial_dims)]
    return displ


def random_affine_displacements_2d(shape: Tuple[int],
                                   theta_range: Union[Tuple[float,], Tuple[float,]] = (math.pi,),
                                   trans_range: Union[Tuple[float, float], Tuple[float, float, float]] = (0.1, 0.1),
                                   scale_range: Union[Tuple[float, float], Tuple[float, float, float]] = (0.05, 0.05),
                                   ) -> torch.Tensor:
    if isinstance(theta_range, (float, int)):
        theta_range = [theta_range]
    if isinstance(trans_range, (float, int)):
        trans_range = [trans_range] * 2
    if isinstance(scale_range, (float, int)):
        scale_range = [scale_range] * 2
    if len(theta_range) > 1:
        raise ValueError("Give me 1 element, not more!")
    if len(trans_range) > 2:
        raise ValueError("Give me either 1 or 2 elements, not more!")
    if len(scale_range) > 2:
        raise ValueError("Give me either 1 or 2 elements, not more!")
    if len(theta_range) == 1:
        pass
    if len(trans_range) == 1:
        trans_range = [trans_range[0]] * 2
    if len(scale_range) == 1:
        scale_range = [scale_range[0]] * 2

    batch_size, channel_dim, spatial_dims = shape[0], shape[1], shape[2:]
    # Define randomly sampled affine params
    theta_range = torch.tensor(theta_range)
    trans_range = torch.tensor(trans_range)
    scale_range = torch.tensor(scale_range)
    # rotation = theta_range[None]
    rotation = torch.rand((shape[0], 1)) * theta_range * 2 - theta_range
    # translation = trans_range[None]
    translation = torch.rand((shape[0], 2)) * trans_range * 2 - trans_range
    scaling = 1 + torch.rand((shape[0], 2)) * scale_range * 2 - scale_range
    # Generate affine matrices
    affines = params_to_mat_2d(rotation, translation, scaling)
    # Define and augment coordinates
    coord_start = torch.meshgrid([torch.linspace(-1., 1., i) for i in spatial_dims], indexing="ij")
    aug_dim = torch.ones_like(coord_start[0])
    coord_start_aug = torch.stack([*coord_start, aug_dim], dim=-1)
    coord_start_aug = coord_start_aug.tile((batch_size, *[1]*len(coord_start_aug.shape)))
    coord_start_aug_ = coord_start_aug.reshape((batch_size, -1, coord_start_aug.shape[-1]))
    # Transform coordinates
    coord_end_aug_ = torch.bmm(coord_start_aug_, affines.transpose(1, 2))
    # coord_end_aug_ = torch.bmm(coord_start_aug_, affines.transpose(1,2))
    coord_end_aug = coord_end_aug_.reshape(coord_start_aug.shape)
    # Calculate displacement
    displ_aug = coord_end_aug - coord_start_aug
    # Remove aug dimension
    displ = displ_aug[..., :len(spatial_dims)]
    return displ


def random_affine_displacements(num_spatial_dims, *args, **kwargs):
    if num_spatial_dims == 3:
        return random_affine_displacements_3d(*args, **kwargs)
    else:
        return random_affine_displacements_2d(*args, **kwargs)


def generate_perlin_noise_3d(shape, res):
    """
    3D Perlin implementation from: https://github.com/pvigier/perlin-numpy/blob/master/perlin3d.py
    Args:
        shape (numpy.ndarray, tuple): Shape of the 3D noise tensor.
        res (numpy.ndarray, tuple): Number of periods of noise to generate along each axis.
    Returns:
        (numpy.ndarray): Tensor filled with 3D Perlin noise.
    """
    if type(shape) == np.ndarray:
        pass
    elif type(shape) == tuple:
        shape = np.array(shape)
    else:
        print("Variable 'shape' should be of type tuple or numpy.array.")
        raise SystemExit
    if type(res) == np.ndarray:
        pass
    elif type(res) == tuple:
        res = np.array(res)
    else:
        print("Variable 'res' should be of type tuple or numpy.array.")
        raise SystemExit

    def f(a):
        return 6 * a ** 5 - 15 * a ** 4 + 10 * a ** 3

    mod = shape % res
    shape = shape - mod + res
    delta = (res[0] / shape[0], res[1] / shape[1], res[2] / shape[2])
    d = (shape[0] // res[0], shape[1] // res[1], shape[2] // res[2])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1], 0:res[2]:delta[2]]
    grid = grid.transpose(1, 2, 3, 0) % 1
    # Gradients
    theta = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1, res[2] + 1)
    phi = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1, res[2] + 1)
    gradients = np.stack((np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)), axis=3)
    g000 = gradients[0:-1, 0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g100 = gradients[1:, 0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g010 = gradients[0:-1, 1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g110 = gradients[1:, 1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g001 = gradients[0:-1, 0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g101 = gradients[1:, 0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g011 = gradients[0:-1, 1:, 1:].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g111 = gradients[1:, 1:, 1:].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    # Ramps
    n000 = np.sum(np.stack((grid[:, :, :, 0], grid[:, :, :, 1], grid[:, :, :, 2]), axis=3) * g000, 3)
    n100 = np.sum(np.stack((grid[:, :, :, 0] - 1, grid[:, :, :, 1], grid[:, :, :, 2]), axis=3) * g100, 3)
    n010 = np.sum(np.stack((grid[:, :, :, 0], grid[:, :, :, 1] - 1, grid[:, :, :, 2]), axis=3) * g010, 3)
    n110 = np.sum(np.stack((grid[:, :, :, 0] - 1, grid[:, :, :, 1] - 1, grid[:, :, :, 2]), axis=3) * g110, 3)
    n001 = np.sum(np.stack((grid[:, :, :, 0], grid[:, :, :, 1], grid[:, :, :, 2] - 1), axis=3) * g001, 3)
    n101 = np.sum(np.stack((grid[:, :, :, 0] - 1, grid[:, :, :, 1], grid[:, :, :, 2] - 1), axis=3) * g101, 3)
    n011 = np.sum(np.stack((grid[:, :, :, 0], grid[:, :, :, 1] - 1, grid[:, :, :, 2] - 1), axis=3) * g011, 3)
    n111 = np.sum(np.stack((grid[:, :, :, 0] - 1, grid[:, :, :, 1] - 1, grid[:, :, :, 2] - 1), axis=3) * g111, 3)
    # Interpolation
    t = f(grid)
    n00 = n000 * (1 - t[:, :, :, 0]) + t[:, :, :, 0] * n100
    n10 = n010 * (1 - t[:, :, :, 0]) + t[:, :, :, 0] * n110
    n01 = n001 * (1 - t[:, :, :, 0]) + t[:, :, :, 0] * n101
    n11 = n011 * (1 - t[:, :, :, 0]) + t[:, :, :, 0] * n111
    n0 = (1 - t[:, :, :, 1]) * n00 + t[:, :, :, 1] * n10
    n1 = (1 - t[:, :, :, 1]) * n01 + t[:, :, :, 1] * n11
    return (1 - t[:, :, :, 2]) * n0 + t[:, :, :, 2] * n1


def generate_gauss_noise_3d(shape, res):
    batch, channel, spatial_shape = shape[0], shape[1], shape[2:5]
    res = res[:len(spatial_shape)]
    d = [max(1, s//r) for s, r in zip(spatial_shape, res)]
    noise = torch.rand((batch, channel, *d), dtype=torch.float32) * 2 - 1.
    inter_mode = "trilinear" if len(spatial_shape) == 3 else "bilinear"
    noise = torch.nn.functional.interpolate(noise, spatial_shape.tolist(), mode=inter_mode)
    return noise


def generate_fractal_noise_3d(shape, res=(16,16,16), octaves=3, freq_multiplier=2, persistence=1.0, amplitude=.125):
    """
    Multiple resolutions of noise get averaged together
    Args:
        shape:
        res: Resolution of noise. Eg. 16 means a random sample is 16 pixels long
        octaves: How many resolutions of noise to generate.
        freq_multiplier: How much DECREASE each resolution by.
        persistence: How much to decrease in intensity each subsequent resolution.
        amplitude: The max amplitude of an individual resolution.
    """
    assert len(shape) == 5
    noise = torch.zeros(shape)
    shape = torch.tensor(shape)
    res = torch.tensor(res)
    frequency = 1
    for _ in range(octaves):
        octave_noise = generate_gauss_noise_3d(shape, frequency * res)[..., :shape[-3], :shape[-2], :shape[-1]]
        octave_noise *= amplitude
        noise += octave_noise / octaves
        frequency *= freq_multiplier
        amplitude *= persistence
    return noise


def random_fractal_deforms(shape, **kwargs):
    if len(shape) == 4:
        shape = (*shape, 1)
        x = generate_fractal_noise_3d(shape, **kwargs)[..., 0]
        y = generate_fractal_noise_3d(shape, **kwargs)[..., 0]
        stack = (x, y)
    else:
        x = generate_fractal_noise_3d(shape, **kwargs)
        y = generate_fractal_noise_3d(shape, **kwargs)
        z = generate_fractal_noise_3d(shape, **kwargs)
        stack = (x, y, z)
    disp = torch.concatenate(stack, dim=1).moveaxis(1, -1)
    return disp


if __name__ == '__main__':
    img = torch.rand((5, 1, 100, 100))
    random_affine_displacements_2d(img.shape)
    random_fractal_deforms(img.shape, torch.tensor((2, 2, 2)))
