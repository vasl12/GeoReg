"""Generalization of unfold operation. Taken from https://github.com/f-dangel/unfoldNd """

from typing import Callable
import torch
from torch import nn

import numpy
from torch.nn.functional import (
    conv1d,
    conv2d,
    conv3d,
    conv_transpose1d,
    conv_transpose2d,
    conv_transpose3d,
)
from torch.nn.modules.utils import _pair, _single, _triple


def unfold3D(x: torch.Tensor, kernel_size: int, pad_value: int, mode: str = "constant"):
    b, d, h, w, c = x.shape
    pad_size = [kernel_size // 2, kernel_size // 2] * len(x.shape[1:-1])  # Spatial dimensions get padding depending on number of neighbours
    if mode == "constant":
        pad_size = [0, 0] + pad_size + [0, 0]  # Padding of batch and channel dims should be zero
        # Padding of batch and channel dims should be zero
        x_pad = nn.functional.pad(x, pad_size, value=pad_value, mode=mode)               # Pad the indexing array such that all points have the same number of neighbours
    else:
        x_pad = nn.functional.pad(x.moveaxis(-1, 1), pad_size, value=pad_value, mode=mode).moveaxis(1, -1)

    if kernel_size % 2 == 0:
        x_pad = x_pad[:, 1:, 1:, 1:]

    # Unfold each spatial dimension (tensor.unfold does not make copies of repeated elements)
    out = x_pad.unfold(1, size=kernel_size, step=1).unfold(2, size=kernel_size, step=1).unfold(3, size=kernel_size, step=1)  # (b, d, h, w, K, K, K, c)
    out = out.permute(0, 1, 2, 3, -3, -2, -1, 4).reshape((b, d, h, w, -1, c))  # Shape: (b, d, h, w, K*K*K, c)
    return out


def unfold2D(x: torch.Tensor, kernel_size: int, pad_value: int, mode: str = "constant"):
    b, h, w, c = x.shape
    pad_size = [kernel_size // 2, kernel_size // 2] * len(x.shape[1:-1])  # Spatial dimensions get padding depending on number of neighbours
    if mode == "constant":
        pad_size = [0, 0] + pad_size + [0, 0]
        # Padding of batch and channel dims should be zero
        x_pad = nn.functional.pad(x, pad_size, value=pad_value, mode=mode)               # Pad the indexing array such that all points have the same number of neighbours
    else:
        x_pad = nn.functional.pad(x.moveaxis(-1, 1), pad_size, mode=mode).moveaxis(1, -1)

    if kernel_size % 2 == 0:
        x_pad = x_pad[:, 1:, 1:]

    # Unfold each spatial dimension (tensor.unfold does not make copies of repeated elements)
    out = x_pad.unfold(1, size=kernel_size, step=1).unfold(2, size=kernel_size, step=1)  # (b, h, w, K, K, c)
    out = out.permute(0, 1, 2, -2, -1, 3).reshape((b, h, w, -1, c))  # Shape: (b, d, h, w, K*K*K, c)
    return out


def unfoldNd(input, kernel_size, dilation=1, padding=0, stride=1, pad_value=-1):
    """Extracts sliding local blocks from a batched input tensor. Also known as im2col.

    Pytorch functional that accepts 3d, 4d, and 5d tensors. Acts like
    ``torch.nn.functional.unfold`` for a 4d input. Uses one-hot convolution under the
    hood.

    See docs at https://pytorch.org/docs/stable/nn.functional.html#unfold.
    """
    batch_size, spatial_sizes, in_channels = input.shape[0], input.shape[1:-1], input.shape[-1]

    pad_size = [kernel_size // 2, kernel_size // 2] * len(input.shape[1:-1])
    input_p = nn.functional.pad(input.moveaxis(-1, 1), pad_size, value=pad_value)

    # get convolution operation
    batch_size_and_in_channels_dims = 2
    N = input.dim() - batch_size_and_in_channels_dims
    conv = _get_conv(N)

    # prepare one-hot convolution kernel
    kernel_size = _tuple(kernel_size, N)
    kernel_size_numel = _get_kernel_size_numel(kernel_size)
    weight = _make_weight(in_channels, kernel_size, input.device, input.dtype)

    unfold = conv(
        input_p,
        weight,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=in_channels,
    )

    # unfold = unfold.reshape(batch_size, in_channels * kernel_size_numel, -1)
    p = unfold.moveaxis(1, -1).reshape(batch_size, *spatial_sizes, in_channels, kernel_size_numel).transpose(-1,-2)
    return p


def _make_weight(in_channels, kernel_size, device, dtype):
    """Create one-hot convolution kernel. ``kernel_size`` must be an ``N``-tuple.

    Details:
        Let's assume 2d convolution. We are given an input of shape `[N,
        C_in, H, W]` and a kernel of shape `[*, *, K_H, K_W]`. We then want to
        produce an output with shape `[N, C_in * K_H * K_W, L]` with `L` the
        number of patches. We can run convolution with `groups=C_in`. This will
        treat each input channel independently with the same kernel `t` of shape
        `[K_H * K_W, 1, K_H, K_W]` that satisfies `t[h * w, 0, h, w] = δ_{h, w}`.
        We can run convolution with `groups=C_in` to achieve this independent
        treatment, but for that we must duplicate it `C_in` times along the leading
        dimension, because the kernel's output dimension must match that of the output
        for convolution in group mode (see its documentation).

        This yields a kernel `T` that satisfies `T[c * h * w, 0, h, w] = δ_{h,w}`.

        Such a kernel is formed by creating a `K_H * K_W` identity matrix,
        reshaping it into `[K_H * K_W, 1, K_H, K_W]` (`t`), and repeating it `C_in`
        times along the leading dimension (`T`).

    Returns:
        torch.Tensor : A tensor of shape ``[C_in * ∏ᵢ Kᵢ, 1, K]`` where
            ``K = (K₁, K₂, ..., Kₙ)`` is the kernel size. Filter groups are
            one-hot such that they effectively extract one element of the patch
            the kernel currently overlaps with.
    """
    kernel_size_numel = _get_kernel_size_numel(kernel_size)
    repeat = [in_channels, 1] + [1 for _ in kernel_size]

    return (
        torch.eye(kernel_size_numel, device=device, dtype=dtype)
        .reshape((kernel_size_numel, 1, *kernel_size))
        .repeat(*repeat)
    )


def _get_kernel_size_numel(kernel_size):
    """Determine number of pixels/voxels. ``kernel_size`` must be an ``N``-tuple."""
    if not isinstance(kernel_size, tuple):
        raise ValueError(f"kernel_size must be a tuple. Got {kernel_size}.")

    return _get_numel_from_shape(kernel_size)


def _get_numel_from_shape(shape_tuple):
    """Compute number of elements from shape."""
    return int(numpy.prod(shape_tuple))


def _tuple(kernel_size, N):
    """Turn ``kernel_size`` argument of ``N``d convolution into an ``N``-tuple."""
    if N == 1:
        return _single(kernel_size)
    elif N == 2:
        return _pair(kernel_size)
    elif N == 3:
        return _triple(kernel_size)
    else:
        _raise_dimension_error(N)


def _get_conv(N: int) -> Callable:
    """Return convolution operation used to perform unfolding."""
    if N == 1:
        return conv1d
    elif N == 2:
        return conv2d
    elif N == 3:
        return conv3d
    else:
        _raise_dimension_error(N)


def _get_conv_transpose(N: int) -> Callable:
    """Return transpose convolution operation used to perform unfolding."""
    if N == 1:
        return conv_transpose1d
    elif N == 2:
        return conv_transpose2d
    elif N == 3:
        return conv_transpose3d
    else:
        _raise_dimension_error(N)


def _raise_dimension_error(N):
    """Notify user that inferred input dimension is not supported."""
    raise ValueError(f"Only 1,2,3-dimensional unfold is supported. Got N={N}.")
