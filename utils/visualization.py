import matplotlib.pyplot as plt
import torch
import numpy as np
import wandb
import neurite as ne

def visualise_results(fixed_image, moving_image, transformed_image, field, title="Validation intensities", show=False, dim=2, cmap='viridis'):
    """input shape (H, W)"""
    fig, ax = plt.subplots(2, 6, figsize=(15, 15))

    fixedAx = fixed_image
    movingAx = moving_image
    warpedAx = transformed_image
    fieldAx = field.numpy()
    ne_disps = []

    for a in range(0, dim):
        s1 = ax[a, 0].imshow(fixedAx, cmap=cmap)
        plt.colorbar(s1, ax=ax[a, 0], fraction=0.045)
        ax[a, 0].set_title('Fixed')
        s2 = ax[a, 1].imshow(movingAx, cmap=cmap)
        ax[a, 1].set_title('Moving')
        plt.colorbar(s2, ax=ax[a, 1], fraction=0.045)
        s3 = ax[a, 2].imshow(warpedAx, cmap=cmap)
        ax[a, 2].set_title('Warped')
        plt.colorbar(s3, ax=ax[a, 2], fraction=0.045)
        s4 = ax[a, 3].imshow(movingAx - fixedAx, cmap='seismic')
        plt.colorbar(s4, ax=ax[a, 3], fraction=0.045)
        s4.set_clim(-1.0, 1.0)
        ax[a, 3].set_title('Error before')
        s5 = ax[a, 4].imshow(warpedAx - fixedAx, cmap='seismic')
        plt.colorbar(s5, ax=ax[a, 4], fraction=0.045)
        s5.set_clim(-1.0, 1.0)
        ax[a, 4].set_title('Error after')

        plot_warped_grid(ax[a, 5], fieldAx, None, interval=5, title=f"axis {a}", fontsize=20)
        ne_disps.append(torch.tensor(fieldAx).permute(1, 2, 0).numpy()[::3, ::3])

    fig1, ax1 = ne.plot.flow(np.rot90(np.rot90(ne_disps, k=1)), show=show, img_indexing=False)

    if show:
        plt.show()
    plt.close(fig)
    plt.close(fig1)
    return fig, fig1



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
