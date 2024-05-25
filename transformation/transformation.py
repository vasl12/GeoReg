import torch
import numpy as np
import torch.nn.functional as F


def normalise_disp(disp):
    """
    Spatially normalise DVF to [-1, 1] coordinate system used by Pytorch `grid_sample()`
    Assumes disp size is the same as the corresponding image.

    Args:
        disp: (numpy.ndarray or torch.Tensor, shape (N, ndim, *size)) Displacement field

    Returns:
        disp: (normalised disp)
    """

    ndim = disp.ndim - 2

    if type(disp) is np.ndarray:
        norm_factors = 2. / np.array(disp.shape[1:-1])
        norm_factors = norm_factors.reshape(1, *(1,) * ndim, ndim)

    elif type(disp) is torch.Tensor:
        norm_factors = torch.tensor(2.) / torch.tensor(disp.size()[1:-1], dtype=disp.dtype, device=disp.device)
        norm_factors = norm_factors.view(1, *(1,)*ndim, ndim)

    else:
        raise RuntimeError("Input data type not recognised, expect numpy.ndarray or torch.Tensor")
    return disp * norm_factors


def unnormalise_disp(disp):
    """
    Spatially unnormalise DVF from [-1,1] to original image shape coordinate system
    Assumes disp size is the same as the corresponding image.

    Args:
        disp: (numpy.ndarray or torch.Tensor, shape (N, ndim, *size)) Displacement field

    Returns:
        disp: (normalised disp)
    """

    if type(disp) is np.ndarray:
        norm_factors = np.array(disp.shape[2:]) / 2
        # norm_factors = norm_factors.reshape(1, ndim, *(1,) * ndim)

    elif type(disp) is torch.Tensor:
        norm_factors = torch.tensor(disp.size()[2:], dtype=disp.dtype, device=disp.device) / 2
        # norm_factors = norm_factors.view(1, ndim, *(1,)*ndim)

    else:
        raise RuntimeError("Input data type not recognised, expect numpy.ndarray or torch.Tensor")
    return disp * norm_factors


def warp(x: torch.Tensor, disp: torch.Tensor, start_coords=None, interp_mode="bilinear", normalise=False):
    """
    Spatially transform an image by sampling at transformed locations (2D and 3D)

    Args:
        x: (Tensor float, shape (N, channels, *sizes)) input image
        disp: (Tensor float, shape (N, *sizes, ndim)) dense disp field in h-w-d order
        interp_mode: (string) mode of interpolation in grid_sample()
        normalise: (bool) whether to normalise disp by the spatial sizes of the array

    Returns:
        deformed x, Tensor of the same shape as input
    """
    assert disp.shape[-1] in {2, 3}
    ndim = x.ndim - 2
    size = disp.shape[1:-1]
    disp = disp.type_as(x)

    if normalise:
        # # normalise disp to [-1, 1]
        disp = normalise_disp(disp.moveaxis(-1, 1)).moveaxis(1, -1)

    # generate standard mesh grid
    if start_coords is None:
        start_coords = torch.meshgrid([torch.linspace(-1, 1, size[i], device=x.device, dtype=x.dtype)
                               for i in range(ndim)], indexing="ij")
        start_coords = torch.stack(start_coords, dim=-1).tile((disp.shape[0], 1, *([1]*len(size))))
    warped_grid = start_coords + disp

    # grid_sample takes in x-y-z ordering, so we need to flip the coordinate dim
    warped_grid = warped_grid.flip(dims=(-1,))
    return F.grid_sample(x, warped_grid, mode=interp_mode, align_corners=True, padding_mode="zeros")


def resample(x: torch.Tensor, coord_grid: torch.Tensor, interp_mode="nearest"):
    # grid_sample takes in x-y-z ordering, so we need to flip the coordinate dim
    coord_grid = coord_grid.flip(dims=(-1,))
    return F.grid_sample(x, coord_grid, mode=interp_mode, align_corners=True)


if __name__ == '__main__':
    x = torch.ones((8, 1, 200, 200), dtype=torch.float32, device="cuda")
    x[..., 100:150, 50:125] = 0.0
    a = torch.cat([x[0,0], x[1,0]]).cpu().numpy()
    d = torch.ones((x.shape[0], 2, *x.shape[2:]))*.5
    d[0, 0] = 0.0
    import time
    start = time.time()
    for _ in range(1000):
        out = warp2(x, d, "nearest")
    time1 = time.time()-start
    b = torch.cat([out[0,0], out[1,0]]).cpu().numpy()
    final = np.concatenate([a, b], 1)
    d = d.moveaxis(1, -1)
    start = time.time()
    for _ in range(1000):
        out = warp(x, d, "nearest")
    time2 = time.time()-start
    print(f"1: {time1}, 2:{time2}")
    c = torch.cat([out[0,0], out[1,0]]).cpu().numpy()
    final = np.concatenate([a, b, c], 1)
    final = np.concatenate([np.concatenate([a, b], 0), np.concatenate([a, c], 0)], 1)
    pass
