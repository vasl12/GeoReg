import numpy as np
import torch
import os
import random
from matplotlib import pyplot as plt
# import neurite as ne

def plot_warped_grid(ax, disp, bg_img=None, interval=3, title="$\mathcal{T}_\phi$", fontsize=30, color='c'):
    """disp shape (2, H, W)"""
    if bg_img is not None:
        background = bg_img
    else:
        background = np.zeros(disp.shape[1:])

    id_grid_H, id_grid_W = np.meshgrid(range(0, background.shape[0] - 1, interval),
                                       range(0, background.shape[1] - 1, interval),
                                       indexing='ij')

    new_grid_H = id_grid_H + disp[0, id_grid_H, id_grid_W]
    new_grid_W = id_grid_W + disp[1, id_grid_H, id_grid_W]

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
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


def plot_result_fig(vis_data_dict, save_path=None, title_font_size=20, dpi=100, show=False, close=False):
    """Plot visual results in a single figure/subplots.
    Images should be shaped (*sizes)
    Disp should be shaped (ndim, *sizes)

    vis_data_dict.keys() = ['fixed', 'moving', 'fixed_original',
                            'fixed_pred', 'warped_moving',
                            'disp_gt', 'disp_pred']
    """
    fig = plt.figure(figsize=(30, 18))
    title_pad = 10

    ax = plt.subplot(2, 6, 1)
    plt.imshow(vis_data_dict["fixed"], cmap='gray')
    plt.axis('off')
    ax.set_title('fixed', fontsize=title_font_size, pad=title_pad)

    ax = plt.subplot(2, 6, 2)
    plt.imshow(vis_data_dict["fixed_original"], cmap='gray')
    plt.axis('off')
    ax.set_title('fixed original', fontsize=title_font_size, pad=title_pad)

    # calculate the error before and after reg
    error_before = vis_data_dict["fixed"] - vis_data_dict["moving"]
    error_after = vis_data_dict["fixed"] - vis_data_dict["warped"]


    # error before
    ax = plt.subplot(2, 6, 3)
    plt.imshow(error_before, vmin=-2, vmax=2, cmap='seismic')  # assuming images were normalised to [0, 1]
    plt.axis('off')
    ax.set_title('Error before', fontsize=title_font_size, pad=title_pad)

    # error after
    ax = plt.subplot(2, 6, 4)
    plt.imshow(error_after, vmin=-2, vmax=2, cmap='seismic')  # assuming images were normalised to [0, 1]
    plt.axis('off')
    ax.set_title('Error after', fontsize=title_font_size, pad=title_pad)

    # predicted fixed image
    ax = plt.subplot(2, 6, 5)
    plt.imshow(vis_data_dict["moving"], cmap='gray')
    plt.axis('off')
    ax.set_title('Moving', fontsize=title_font_size, pad=title_pad)

    # warped moving image
    ax = plt.subplot(2, 6, 6)
    plt.imshow(vis_data_dict["warped_moving"], cmap='gray')
    plt.axis('off')
    ax.set_title('Warped moving', fontsize=title_font_size, pad=title_pad)

    # warped grid: ground truth
    ax = plt.subplot(2, 6, 7)
    bg_img = np.zeros_like(vis_data_dict["fixed"])
    plot_warped_grid(ax, vis_data_dict["disp_gt"], bg_img, interval=3, title="$\phi_{GT}$", fontsize=title_font_size)

    # warped grid: prediction
    ax = plt.subplot(2, 6, 8)
    plot_warped_grid(ax, vis_data_dict["disp_pred"], bg_img, interval=3, title="$\phi_{pred}$", fontsize=title_font_size)

    ax = plt.subplot(2, 6, 9)
    plt.imshow(vis_data_dict["fixed_seg"], cmap='gray')
    plt.axis('off')
    ax.set_title('fixed seg', fontsize=title_font_size, pad=title_pad)

    ax = plt.subplot(2, 6, 10)
    plt.imshow(vis_data_dict["fixed_seg"] - vis_data_dict["warped_seg"], cmap='gray')
    plt.axis('off')
    ax.set_title('diff seg', fontsize=title_font_size, pad=title_pad)

    ax = plt.subplot(2, 6, 11)
    plt.imshow(vis_data_dict["moving_seg"], cmap='gray')
    plt.axis('off')
    ax.set_title('moving seg', fontsize=title_font_size, pad=title_pad)

    ax = plt.subplot(2, 6, 12)
    plt.imshow(vis_data_dict["warped_seg"], cmap='gray')
    plt.axis('off')
    ax.set_title('warped seg', fontsize=title_font_size, pad=title_pad)


    # adjust subplot placements and spacing
    plt.subplots_adjust(left=0.0001, right=0.99, top=0.9, bottom=0.1, wspace=0.001, hspace=0.1)

    # saving
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', dpi=dpi)

    if show:
        plt.show()

    if close:
        plt.close()
    return fig


def visualise_result(data_dict, axis=0, save_result_dir=None, epoch=None, dpi=50):
    """
    Save one validation visualisation figure for each epoch.
    - 2D: 1 random slice from N-slice stack (not a sequence)
    - 3D: the middle slice on the chosen axis

    Args:
        data_dict: (dict) images shape (N, 1, *sizes), disp shape (N, ndim, *sizes)
        save_result_dir: (string) Path to visualisation result directory
        epoch: (int) Epoch number (for naming when saving)
        axis: (int) For 3D only, choose the 2D plane orthogonal to this axis in 3D volume
        dpi: (int) Image resolution of saved figure
    """
    # check cast to Numpy array
    for n, d in data_dict.items():
        if isinstance(d, torch.Tensor):
            data_dict[n] = d.cpu().numpy()

    ndim = data_dict["fixed"].ndim - 2
    sizes = data_dict["fixed"].shape[2:]

    # put 2D slices into visualisation data dict
    vis_data_dict = {}
    if ndim == 2:
        # randomly choose a slice for 2D
        z = random.randint(0, data_dict["fixed"].shape[0]-1)
        for name, d in data_dict.items():
            # if name in ["disp_pred", "disp_gt"]:
            #     vis_data_dict[name] = data_dict[name].squeeze()[z, ...]  # (H, W) or (2, H, W)
            # else:
            vis_data_dict[name] = data_dict[name][z, ...].squeeze()

    else:  # 3D
        # visualise the middle slice of the chosen axis
        z = int(sizes[axis] // 2)
        for name, d in data_dict.items():
            if name in ["disp_pred", "disp_gt"]:
                # dvf.yaml: choose the two axes/directions to visualise
                axes = [0, 1, 2]
                axes.remove(axis)
                vis_data_dict[name] = d[0, axes, ...].take(z, axis=axis+1)  # (2, X, X)
            else:
                # images
                vis_data_dict[name] = d[0, 0, ...].take(z, axis=axis)  # (X, X)

    # housekeeping: dummy dvf_gt for inter-subject case
    if not "disp_gt" in data_dict.keys():
        vis_data_dict["disp_gt"] = np.zeros_like(vis_data_dict["disp_pred"])

    # set up figure saving path
    if save_result_dir is not None:
        fig_save_path = os.path.join(save_result_dir, f'epoch{epoch}_axis_{axis}_slice_{z}.png')
    else:
        fig_save_path = None

    fig = plot_result_fig(vis_data_dict, save_path=fig_save_path, dpi=dpi)
    return fig

def visualise_results(fixed_image, moving_image, transformed_image, field, title="Validation intensities"):

    assert field.shape[0] == 2 or field.shape[0] == 3
    image_size = fixed_image.shape
    # field = def_coords.reshape(*image_size, 3).cpu() * image_size[0]/2
    # field = np.transpose(field)
    # field = field.permute(0, 3, 2, 1)

    fig, ax = plt.subplots(3, 6, figsize=(15, 15))

    slices = None
    if slices is None:
        slices = [int(image_size[0] / 2), int(image_size[1] / 2), int(image_size[2] / 2)]

    ne_disps = []
    for a in range(0, 3):
        axes = [0, 1, 2]
        axes.remove(a)
        # z = 64
        z = slices[a]

        fixedAx = torch.index_select(fixed_image.cpu().clone().reshape(image_size), dim=a, index=torch.tensor([z])).squeeze().numpy()
        movingAx = torch.index_select(moving_image.clone().cpu().reshape(image_size), dim=a, index=torch.tensor([z])).squeeze().numpy()
        warpedAx = torch.index_select(transformed_image.detach().cpu().clone().reshape(image_size), dim=a, index=torch.tensor([z])).squeeze().numpy()

        s1 = ax[a, 0].imshow(fixedAx, cmap='gray')
        plt.colorbar(s1, ax=ax[a, 0], fraction=0.045)
        ax[a, 0].set_title('Fixed')
        s2 = ax[a, 1].imshow(movingAx, cmap='gray')
        ax[a, 1].set_title('Moving')
        plt.colorbar(s2, ax=ax[a, 1], fraction=0.045)
        s3 = ax[a, 2].imshow(warpedAx, cmap='gray')
        ax[a, 2].set_title('Warped')
        plt.colorbar(s3, ax=ax[a, 2], fraction=0.045)
        s4 = ax[a, 3].imshow(movingAx - fixedAx, cmap='seismic')
        plt.colorbar(s4, ax=ax[a, 3], fraction=0.045)
        ax[a, 3].set_title('Error before')
        s4.set_clim(-1.0, 1.0)
        s5 = ax[a, 4].imshow(warpedAx - fixedAx, cmap='seismic')
        plt.colorbar(s5, ax=ax[a, 4], fraction=0.045)
        ax[a, 4].set_title('Error after')
        s5.set_clim(-1.0, 1.0)


        fieldAx = torch.index_select(field[axes, ...], dim=a+1, index=torch.tensor([z])).squeeze().numpy()
        plot_warped_grid(ax[a, 5], fieldAx, None, interval=5, title=f"axis {a}", fontsize=20)

    #     ne_disps.append(torch.tensor(fieldAx).permute(1, 2, 0).numpy()[::4, ::4])
    #
    # fig1, ax1 = ne.plot.flow(ne_disps, show=False, img_indexing=False)

    return fig






