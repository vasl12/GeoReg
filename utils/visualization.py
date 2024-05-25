from typing import List, Optional, Union, Any, Dict, Tuple

import matplotlib

from utils.utils import normalize

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import numpy as np


def quiver_plot_2d(ax, coord, displ, max_size=24):
    assert len(coord.shape) == 3
    assert coord.shape[-1] == 2
    assert len(displ.shape) == 3
    assert displ.shape[-1] == 2
    assert coord.shape == displ.shape
    # If resolution is too high, we won't be able to see anything. Downsample!
    df0 = 1 if displ.shape[0] <= max_size else displ.shape[0] // max_size  # Make overly-fine grids coarser by this factor
    df1 = 1 if displ.shape[1] <= max_size else displ.shape[1] // max_size
    coord = coord[::df0, ::df1]
    displ = displ[::df0, ::df1]
    # displ = -displ  # For visualization, we plot arrows to where points are moving, not where they sample from
    displ[displ.norm() < 1e-3] = 1e-3  # If displ is too small, arrows won't be visible
    x, y = coord[..., 0].flatten(), coord[..., 1].flatten()
    u, v = displ[..., 0].flatten(), displ[..., 1].flatten()
    c = torch.arctan(u/v)  # Color based on angle
    ax.set_xlim([-.99, .99])
    ax.set_ylim([-.99, .99])
    ax.quiver(-y, x, -v, u, c, angles='xy', scale_units='xy', scale=1)
    return


def visualise_results_2d(fixed_image: torch.Tensor, moving_image: torch.Tensor,
                         transformed_image: torch.Tensor, field: torch.Tensor,
                         transformed_image_gt: Optional[torch.Tensor] = None,
                         field_gt: Optional[torch.Tensor] = None,
                         title: str = "",
                         metric_name: str = "",
                         error_mov_fix: Optional[float] = None,
                         error_warp_fix: Optional[float] = None,
                         error_warpgt_fix: Optional[float] = None,
                         error_warp_warpgt: Optional[float] = None,
                         vmin=0.0,
                         vmax=1.0,
                         show=False,
                         cmap='gray'):
    """input shape (H, W)"""
    cols = 5 if transformed_image_gt is not None or field_gt is not None else 3
    fig, ax = plt.subplots(2, cols, figsize=(6*cols, 6*2.5))
    if title:
        fig.suptitle(title, fontsize=16)

    s5 = ax[0, 0].imshow(transformed_image - fixed_image, cmap='seismic', vmin=-vmax, vmax=vmax)
    plt.colorbar(s5, ax=ax[0, 0], fraction=0.045, location='bottom')
    s5.set_clim(-1.0, 1.0)
    ax_title = "Error warped-fixed" + ("" if error_warp_fix is None else f"\n({metric_name}: {error_warp_fix})")
    ax[0, 0].set_title(ax_title)

    s4 = ax[1, 0].imshow(moving_image - fixed_image, cmap='seismic', vmin=-vmax, vmax=vmax)
    plt.colorbar(s4, ax=ax[1, 0], fraction=0.045, location='bottom')
    s4.set_clim(-1.0, 1.0)
    ax_title = "Error moving-fixed" + ("" if error_mov_fix is None else f"\n({metric_name}: {error_mov_fix})")
    ax[1, 0].set_title(ax_title)

    s1 = ax[0, 1].imshow(fixed_image, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(s1, ax=ax[0, 1], fraction=0.045, location='bottom')
    ax[0, 1].set_yticks([])
    ax[0, 1].set_title('Fixed')
    s2 = ax[1, 1].imshow(moving_image, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(s2, ax=ax[1, 1], fraction=0.045, location='bottom')
    ax[1, 1].set_yticks([])
    ax[1, 1].set_title('Moving')
    s3 = ax[0, 2].imshow(transformed_image, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(s3, ax=ax[0, 2], fraction=0.045, location='bottom')
    ax[0, 2].set_yticks([])
    ax[0, 2].set_title('Warped')

    plot_warped_grid(ax[1, 2], field.numpy(), None, interval=4, title=f"axis {0}", fontsize=20)
    if field_gt is not None:
        plot_warped_grid(ax[1, 3], field_gt.numpy(), None, interval=4, title=f"GT axis {0}", fontsize=20)

        s4 = ax[0, 3].imshow(transformed_image_gt, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(s4, ax=ax[0, 3], fraction=0.045, location='bottom')
        ax[0, 3].set_yticks([])
        ax[0, 3].set_title('Warped with GT Affine disp')

        s5 = ax[0, 4].imshow(transformed_image - transformed_image_gt, cmap='seismic', vmin=-vmax, vmax=vmax)
        plt.colorbar(s5, ax=ax[0, 4], fraction=0.045, location='bottom')
        s5.set_clim(-1.0, 1.0)
        ax[0, 4].set_yticks([])
        ax_title = "Error warped-GT_Affine" + ("" if error_warp_fix is None else f"\n({metric_name}: {error_warp_warpgt})")
        ax[0, 4].set_title(ax_title)
        s4 = ax[1, 4].imshow(transformed_image_gt - fixed_image, cmap='seismic', vmin=-vmax, vmax=vmax)
        plt.colorbar(s4, ax=ax[1, 4], fraction=0.045, location='bottom')
        s4.set_clim(-1.0, 1.0)
        ax[1, 4].set_yticks([])
        ax_title = "Error GT_Affine-fixed" + ("" if error_warp_fix is None else f"\n({metric_name}: {error_warpgt_fix})")
        ax[1, 4].set_title(ax_title)
    if show:
        plt.show()

    # Converting pyplot figs to RGB takes longer but significantly reduces WandB logging
    canvas = fig.canvas
    canvas.draw()  # Draw the canvas, cache the renderer
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(*reversed(canvas.get_width_height()), 3)
    plt.close(fig)
    return image


def visualise_features(fixed_image: List[List[torch.Tensor]],
                       moving_image: List[List[torch.Tensor]],
                       title="Validation intensities", show=False, cmap='viridis'):
    """input shape (H, W)"""
    num_grids = 5
    fig, ax = plt.subplots(2*len(fixed_image), num_grids, figsize=(3*num_grids, 3*2*len(fixed_image)))

    for j, (f, m) in enumerate(zip(fixed_image, moving_image)):
        for i in range(num_grids):
            f_f = normalize(f[..., i*3: (i+1)*3].detach().cpu().numpy())
            m_f = normalize(m[..., i * 3: (i + 1) * 3].detach().cpu().numpy())
            f_ax = ax[j*2, i].imshow(f_f, cmap=cmap)
            if i > 0:
                ax[j*2, i].set_yticks([])
            m_ax = ax[(j*2)+1, i].imshow(m_f, cmap=cmap)
            if i > 0:
                ax[(j*2)+1, i].set_yticks([])
        ax[j*2, 0].set_ylabel(f"Fixed\n({f.shape[0]} x {f.shape[1]})")
        ax[(j*2)+1, 0].set_ylabel(f"Moving\n({m.shape[0]} x {m.shape[1]})")

    plt.tight_layout()
    plt.close(fig)
    plt.clf()
    return fig


def visualise_displ_components(displ: Dict[str, Tuple[List[torch.Tensor], List[torch.Tensor]]],
                               ims: Optional[Dict[str, List[torch.Tensor]]] = None,
                               cmap='gray'):
    assert isinstance(displ, dict)
    assert isinstance(list(displ.values())[0], tuple)
    assert isinstance(list(displ.values())[0][0], list)
    sample_disp_per_res = list(displ.values())[0][0]
    # Number of columns is number of displacement arrays + number of images
    col_num = len(displ) + (0 if ims is None else len(ims))
    fig, ax = plt.subplots(len(sample_disp_per_res), col_num, figsize=(5*col_num, 5*len(sample_disp_per_res)))

    # We use this variable to find the number of resolutions and the shape of the array at each resolution
    for res in range(len(sample_disp_per_res)):  # For each resolution
        axres = (ax[res] if len(sample_disp_per_res) > 1 else ax)  # Pyplot will remove a dimension if the size is 1
        # Write on the left axis what shape the current resolution has
        axres[0].set_ylabel(f'Res: ({sample_disp_per_res[res].shape[0]} x {sample_disp_per_res[res].shape[1]})')

        count = 0
        # Iterate over all displacement elements in the displ dictionary.
        # The key of the dict is the title of the given column and the value is a (coord, displ) tuple
        for col_idx, (displ_title, (coord_list, displ_list)) in enumerate(displ.items()):
            # Create quiver plot on given subplot axis
            quiver_plot_2d(axres[count], coord_list[res], displ_list[res]*torch.tensor(np.array(coord_list[-1].shape[:2]) / np.array(displ_list[res].shape[:2])))
            # plot_warped_grid(axres[count], displ_list[res], None, interval=3, title=f"axis {0}", fontsize=20)

            if res == 0:
                # If we're at the top-most axis of the plot, display the title on this axis
                axres[count].set_title(displ_title)
            if col_idx > 0:
                # If we're not on a left-most axis, hide the y-axis ticks
                axres[count].set_yticks([])
            if res < len(sample_disp_per_res) - 1:
                # If we're not on a bottom-most axis, hide the x-axis ticks
                axres[count].set_xticks([])
            count += 1

        # Iterate over all image elements in the ims dictionary.
        # The key of the dict is the title of the given column and the value is the image array
        for im_title, im_list in ims.items():
            # Display image
            axres[count].imshow(im_list[res], cmap=cmap)

            if res == 0:
                # If we're at the top-most axis of the plot, display the title on this axis
                axres[count].set_title(im_title)
            axres[count].set_yticks([])
            if res < len(sample_disp_per_res) - 1:
                # If we're not on a bottom-most axis, hide the x-axis ticks
                axres[count].set_xticks([])
            count += 1

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.close(fig)
    plt.clf()
    return fig


def plot_warped_grid(ax, disp, bg_img=None, interval=5, title="$\mathcal{T}_\phi$", fontsize=30, color='c'):
    """disp shape (H, W, 2)"""
    h, w = disp.shape[:2]
    # We want to trim disp such that there is no space after the nth line at idx n*interval+1
    disp = disp[:interval * ((h-1)//interval) + 1, :interval * ((w-1)//interval) + 1]
    if bg_img is not None:
        background = bg_img[:disp.shape[0], :disp.shape[1]]
    else:
        background = np.zeros(disp.shape[:2])  # black background

    id_grid_H, id_grid_W = np.meshgrid(range(0, background.shape[0]),
                                       range(0, background.shape[1]),
                                       indexing='ij')

    new_grid_H = id_grid_H + disp[id_grid_H, id_grid_W, 0]
    new_grid_W = id_grid_W + disp[id_grid_H, id_grid_W, 1]

    kwargs = {"linewidth": 1.5, "color": color}
    # matplotlib.plot() uses x-y(-z) indexing
    for i in range(0, new_grid_H.shape[0], interval):
        ax.plot(new_grid_W[i, :], new_grid_H[i, :], **kwargs)  # each draws a horizontal line
    for i in range(0, new_grid_H.shape[1], interval):
        ax.plot(new_grid_W[:, i], new_grid_H[:, i], **kwargs)  # each draws a vertical line

    ax.set_title(title, fontsize=fontsize)
    ax.imshow(background, cmap='gray')
    ax.grid(False)
    # ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


def plot_warped_grid_old(ax, disp, bg_img=None, interval=2, title="$\mathcal{T}_\phi$", fontsize=30, color='c'):
    """disp shape (H, W, 2)"""
    if bg_img is not None:
        background = bg_img
    else:
        background = np.zeros(disp.shape[:2])

    id_grid_H, id_grid_W = np.meshgrid(range(0, background.shape[0] - 1, interval),
                                       range(0, background.shape[1] - 1, interval),
                                       indexing='ij')
    new_grid_H = id_grid_H + disp[id_grid_H, id_grid_W, 0]
    new_grid_W = id_grid_W + disp[id_grid_H, id_grid_W, 1]

    kwargs = {"linewidth": 1.5, "color": color}
    # matplotlib.plot() uses CV x-y indexing
    for i in range(new_grid_H.shape[0]):
        ax.plot(new_grid_W[i, :], new_grid_H[i, :], **kwargs)  # each draws a horizontal line
    for i in range(new_grid_H.shape[1]):
        ax.plot(new_grid_W[:, i], new_grid_H[:, i], **kwargs)  # each draws a vertical line

    ax.set_title(title, fontsize=fontsize)
    ax.imshow(background, cmap='gray')
    # ax.axis('off')
    ax.grid(False)
    # ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
