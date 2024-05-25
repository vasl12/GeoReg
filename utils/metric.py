import numpy as np
import pandas as pd
import torch
import cv2
from scipy.spatial.distance import directed_hausdorff
import SimpleITK as sitk

from transformation.transformation import unnormalise_disp
from utils.image import bbox_from_mask, bbox_crop
from monai.metrics import HausdorffDistanceMetric, SurfaceDistanceMetric, SSIMMetric, PSNRMetric
from utils.misc import save_dict_to_csv
import torch.nn.functional as F


def measure_metrics(metric_data, metric_groups, return_tensor=False, **kwargs):
    """
    Wrapper function for calculating all metrics
    Args:
        metric_data: (dict) data used for calculation of metrics, could be Tensor or Numpy Array
        metric_groups: (list of strings) name of metric groups
        return_tensor: (bool) return Torch Tensor if True

    Returns:
        metrics_results: (dict) {metric_name: metric_value}
    """

    # keys must match metric_groups and params.metric_groups
    # (using groups to share pre-scripts)
    metric_group_fns = {'disp_metrics': measure_disp_metrics,
                        'image_metrics': measure_image_metrics,
                        'seg_metrics': measure_seg_metrics,
                        'landmark_metrics': measure_landmark_metrics}

    metric_results = dict()
    for group in metric_groups:
        metric_results.update(metric_group_fns[group](metric_data, **kwargs))

    # cast Numpy array to Tensor if needed
    if return_tensor:
        for k, x in metric_results.items():
            metric_results[k] = torch.tensor(x)

    return metric_results


"""
Functions calculating groups of metrics
"""


def measure_disp_metrics(metric_data, **kwargs):
    """
    Calculate DVF-related metrics.
    If roi_mask is given, the disp is masked and only evaluate in the bounding box of the mask.

    Args:
        metric_data: (dict)

    Returns:
        metric_results: (dict)
    """
    # new object to avoid changing data in metric_data
    disp_pred = unnormalise_disp(metric_data['disp_norm']).detach().cpu().numpy()
    disp_gt = unnormalise_disp(metric_data['disp']).detach().cpu().numpy() if 'disp' in metric_data else None
    disp_affine = unnormalise_disp(metric_data['disp_affine']).detach().cpu().numpy() if 'disp_affine' in metric_data else None

    # mask the disp with roi mask if given
    if 'roi_mask' in metric_data.keys():
        roi_mask = metric_data['roi_mask']  # (N, 1, *(sizes))

        # find roi mask bbox mask
        mask_bbox, mask_bbox_mask = bbox_from_mask(roi_mask[:, 0, ...])

        # mask and bbox crop dvf gt and pred by roi_mask
        disp_pred = disp_pred * roi_mask
        disp_pred = bbox_crop(disp_pred, mask_bbox)

        if disp_gt is not None:
            disp_gt = disp_gt * roi_mask
            disp_gt = bbox_crop(disp_gt, mask_bbox)
        if disp_affine is not None:
            disp_affine = disp_affine * roi_mask
            disp_affine = bbox_crop(disp_affine, mask_bbox)

    # Regularity (Jacobian) metrics
    folding_ratio, mag_det_jac_det = calculate_jacobian_metrics(disp_pred)

    disp_metric_results = {'folding_ratio': folding_ratio, 'mag_det_jac_det': mag_det_jac_det}

    # DVF accuracy metrics if ground truth is available
    if disp_gt is not None:
        disp_metric_results['aee_disp'] = calculate_aee(disp_pred, disp_gt)
        disp_metric_results['rmse_disp'] = calculate_rmse_disp(disp_pred, disp_gt)
    if disp_affine is not None:
        disp_metric_results['aee_disp_affine'] = calculate_aee(disp_pred, disp_affine)
        disp_metric_results['rmse_disp_affine'] = calculate_rmse_disp(disp_pred, disp_affine)
    return disp_metric_results


def measure_image_metrics(metric_data, **kwargs):

    result = dict()
    # unpack metric data, keys must match metric_data input
    img = metric_data['fixed']
    img_pred = metric_data['warped_moving']  # (N, 1, *sizes)

    # crop out image by the roi mask bounding box if given
    if 'roi_mask' in metric_data.keys():
        roi_mask = metric_data['roi_mask']
        mask_bbox, mask_bbox_mask = bbox_from_mask(roi_mask[:, 0, ...])
        img = bbox_crop(img, mask_bbox)
        img_pred = bbox_crop(img_pred, mask_bbox)

    result['rmse'] = calculate_rmse(img, img_pred)

    # SSIM and PSNR
    ssim_metric = SSIMMetric(spatial_dims=len(img.shape) - 2)
    ssim_metric.reset()
    ssim_metric(img_pred, img)
    result['ssim'] = ssim_metric.aggregate('mean')

    psnr_metric = PSNRMetric(max_val=1.0)
    psnr_metric.reset()
    psnr_metric(img_pred, img)
    result['psnr'] = psnr_metric.aggregate('mean')

    # try:
    #     # Perceptual loss
    #     with torch.no_grad():
    #         # TODO: when monai resolves the medical model issue we can set is_fake_3d to False when doing 3D
    #         perceptual_loss = PerceptualLoss(spatial_dims=len(img.shape) - 2)
    #         lpips = perceptual_loss(torch.tensor(img_pred), torch.tensor(img))
    #         result['lpips'] = lpips
    # except Exception:
    #     pass  # Not so important if we can't get MONAI to run this

    return result


def measure_landmark_metrics(metric_data, spacing=(1.5, 1.5, 1.5), **kwargs):
    """
    Calculate landmark metrics
    """
    device = metric_data['fixed_landmarks'].device
    vx = np.array(spacing) / (np.array(metric_data['fixed'].shape[2:]) / np.array([224, 192, 224]))
    vx = torch.tensor(vx, device=device)
    batch_size = metric_data['fixed'].shape[0]
    # we use the landmark_shape to identify the non-padded landmarks
    results = dict()
    fixed_landmarks = metric_data['fixed_landmarks']
    moving_landmarks = metric_data['moving_landmarks']

    # land_disp = F.grid_sample(torch.tensor(metric_data['disp_norm'], device=device).moveaxis(-1, 1),
    #                           fixed_landmarks.view(1, -1, 1, 1, 3).float(),
    #                           align_corners=True).squeeze().t()[None, None]
    #
    offset = torch.tensor(metric_data['fixed'].shape[2:], device=device) / 2
    offset = offset[None][None]

    land_disp = torch.flip((fixed_landmarks - offset) / offset, (-1,)).float()
    land_disp = F.grid_sample(metric_data['disp_pred'].moveaxis(-1, 1),
                              land_disp.view(batch_size, -1, 1, 1, 3),
                              align_corners=True).view(batch_size, 3, -1).permute((0, 2, 1))

    warped_fixed_landmarks = land_disp + fixed_landmarks.float()

    tre_before = ((fixed_landmarks - moving_landmarks).mul(vx) ** 2).sum(-1).sqrt()
    results['tre_before'] = tre_before.nanmean()

    tre_after = ((warped_fixed_landmarks - moving_landmarks).mul(vx)  ** 2).sum(-1).sqrt()
    results['tre_after'] = tre_after.nanmean()

    return results


def measure_seg_metrics(metric_data, seg_class_max=None, **kwargs):
    """ Calculate segmentation """
    seg_gt = metric_data['fixed_seg'].int()
    seg_pred = metric_data['warped_moving_seg'].int()
    assert seg_gt.ndim == seg_pred.ndim

    if seg_class_max is None:
        seg_class_range = torch.unique(seg_gt)
    else:
        seg_class_range = range(seg_class_max)
    results = dict()
    results_before = dict()
    for label_cls in seg_class_range:
        # calculate DICE score for each class
        # if label_cls == 0 or label_cls == 5:
        if label_cls == 0:
            # skip background
            continue
        results[f'dice_class_{label_cls}'] = calculate_dice(seg_gt, seg_pred, label_class=label_cls)
        results_before[f'dice_class_{label_cls}'] = calculate_dice(seg_gt, metric_data['moving_seg'].int(),
                                                                   label_class=label_cls)

    # calculate mean dice
    results['mean_dice'] = np.mean([dice.cpu() for k, dice in results.items()])
    results['mean_dice_before'] = np.mean([dice.cpu() for k, dice in results_before.items()])

    # calculate haudorff distance
    hausdorf_metric = HausdorffDistanceMetric(percentile=95)
    hausdorf_metric.reset()
    hausdorf_metric(seg_pred, seg_gt)
    results['hausdorf_distance'] = hausdorf_metric.aggregate('mean')

    # calculate surface distance
    surface_metric = SurfaceDistanceMetric()
    surface_metric.reset()
    surface_metric(seg_pred, seg_gt)
    results['surface_distance'] = surface_metric.aggregate('mean')

    if 'warped_moving_seg_disp_affine' in metric_data:
        results_disp_affine = dict()
        seg_disp_affine = metric_data['warped_moving_seg_disp_affine'].int()
        for label_cls in seg_class_range:
            if label_cls == 0:
                # skip background
                continue
            results_disp_affine[f'dice_class_{label_cls}'] = calculate_dice(seg_gt, seg_disp_affine,
                                                                            label_class=label_cls)
        results['mean_dice_disp_affine'] = np.mean([dice.cpu() for k, dice in results_disp_affine.items()])
        hausdorf_metric.reset()
        hausdorf_metric(seg_disp_affine, seg_gt)
        results['hausdorf_distance_disp_affine'] = hausdorf_metric.aggregate('mean')
    return results


"""
Functions calculating individual metrics
"""


def calculate_aee(x, y):
    """
    Average End point Error (AEE, mean over point-wise L2 norm)
    Input DVF shape: (N, dim, *(sizes))
    """
    return np.sqrt(((x - y) ** 2).sum(axis=-1)).mean()


def calculate_rmse_disp(x, y):
    """
    RMSE of DVF (square root over mean of sum squared)
    Input DVF shape: (N, dim, *(sizes))
    """
    return np.sqrt(((x - y) ** 2).sum(axis=-1).mean())


def calculate_rmse(x, y):
    """Standard RMSE formula, square root over mean
    (https://wikimedia.org/api/rest_v1/media/math/render/svg/6d689379d70cd119e3a9ed3c8ae306cafa5d516d)
    """
    return torch.sqrt(((x - y) ** 2).mean())


def calculate_jacobian_metrics(disp: np.ndarray):
    """
    Calculate Jacobian related regularity metrics.

    Args:
        disp: (numpy.ndarray, shape (N, *sizes, ndim) Displacement field

    Returns:
        folding_ratio: (scalar) Folding ratio (ratio of Jacobian determinant < 0 points)
        mag_grad_jac_det: (scalar) Mean magnitude of the spatial gradient of Jacobian determinant
    """
    folding_ratio = []
    mag_grad_jac_det = []
    for n in range(disp.shape[0]):
        # disp_n = np.moveaxis(disp[n, ...], 0, -1)  # (*sizes, ndim)
        disp_n = disp[n]
        jac_det_n = calculate_jacobian_det(disp_n)
        folding_ratio += [(jac_det_n < 0).sum() / np.prod(jac_det_n.shape)]
        mag_grad_jac_det += [np.abs(np.gradient(jac_det_n)).mean()]
    return np.mean(folding_ratio), np.mean(mag_grad_jac_det)


def calculate_jacobian_det(disp):
    """
    Calculate Jacobian determinant of displacement field of one image/volume (2D/3D)

    Args:
        disp: (numpy.ndarray, shape (*sizes, ndim)) Displacement field

    Returns:
        jac_det: (numpy.adarray, shape (*sizes) Point-wise Jacobian determinant
    """
    disp_img = sitk.GetImageFromArray(disp, isVector=True)
    jac_det_img = sitk.DisplacementFieldJacobianDeterminant(disp_img)
    jac_det = sitk.GetArrayFromImage(jac_det_img)
    return jac_det


def calculate_dice_no_background(labels1, labels2):
    dice = np.mean([calculate_dice(labels1, labels2, i).detach().cpu().numpy() for i in torch.unique(labels2) if i > 0])
    return dice


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
    mask1_pos = (mask1 == label_class).float()
    mask2_pos = (mask2 == label_class).float()

    assert mask1.ndim == mask2.ndim
    axes = tuple(range(2, mask1.ndim))
    pos1and2 = torch.sum(mask1_pos * mask2_pos, dim=axes)
    pos1 = torch.sum(mask1_pos, dim=axes)
    pos2 = torch.sum(mask2_pos, dim=axes)
    return torch.mean(2 * pos1and2 / (pos1 + pos2 + 1e-7))


def contour_distances_2d(image1, image2, dx=1):
    """
    Calculate contour distances between binary masks.
    The region of interest must be encoded by 1

    Args:
        image1: 2D binary mask 1
        image2: 2D binary mask 2
        dx: physical size of a pixel (e.g. 1.8 (mm) for UKBB)

    Returns:
        mean_hausdorff_dist: Hausdorff distance (mean if input are 2D stacks) in pixels
    """

    # Retrieve contours as list of the coordinates of the points for each contour
    # convert to contiguous array and data type uint8 as required by the cv2 function
    image1 = np.ascontiguousarray(image1, dtype=np.uint8)
    image2 = np.ascontiguousarray(image2, dtype=np.uint8)

    # extract contour points and stack the contour points into (N, 2)
    contours1, _ = cv2.findContours(image1.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour1_pts = np.array(contours1[0])[:, 0, :]
    for i in range(1, len(contours1)):
        cont1_arr = np.array(contours1[i])[:, 0, :]
        contour1_pts = np.vstack([contour1_pts, cont1_arr])

    contours2, _ = cv2.findContours(image2.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour2_pts = np.array(contours2[0])[:, 0, :]
    for i in range(1, len(contours2)):
        cont2_arr = np.array(contours2[i])[:, 0, :]
        contour2_pts = np.vstack([contour2_pts, cont2_arr])

    # distance matrix between two point sets
    dist_matrix = np.zeros((contour1_pts.shape[0], contour2_pts.shape[0]))
    for i in range(contour1_pts.shape[0]):
        for j in range(contour2_pts.shape[0]):
            dist_matrix[i, j] = np.linalg.norm(contour1_pts[i, :] - contour2_pts[j, :])

    # symmetrical mean contour distance
    mean_contour_dist = 0.5 * (np.mean(np.min(dist_matrix, axis=0)) + np.mean(np.min(dist_matrix, axis=1)))

    # calculate Hausdorff distance using the accelerated method
    # (doesn't really save computation since pair-wise distance matrix has to be computed for MCD anyways)
    hausdorff_dist = directed_hausdorff(contour1_pts, contour2_pts)[0]

    return mean_contour_dist * dx, hausdorff_dist * dx


def contour_distances_stack(stack1, stack2, label_class, dx=1):
    """
    Measure mean contour distance metrics between two 2D stacks

    Args:
        stack1: stack of binary 2D images, shape format (W, H, N)
        stack2: stack of binary 2D images, shape format (W, H, N)
        label_class: class of which to calculate distance
        dx: physical size of a pixel (e.g. 1.8 (mm) for UKBB)

    Return:
        mean_mcd: mean contour distance averaged over non-empty slices
        mean_hd: Hausdorff distance averaged over non-empty slices
    """

    # assert the two stacks has the same number of slices
    assert stack1.shape[-1] == stack2.shape[-1], 'Contour dist error: two stacks has different number of slices'

    # mask by class
    stack1 = (stack1 == label_class).astype('uint8')
    stack2 = (stack2 == label_class).astype('uint8')

    mcd_buffer = []
    hd_buffer = []
    for slice_idx in range(stack1.shape[-1]):
        # ignore empty masks
        if np.sum(stack1[:, :, slice_idx]) > 0 and np.sum(stack2[:, :, slice_idx]) > 0:
            slice1 = stack1[:, :, slice_idx]
            slice2 = stack2[:, :, slice_idx]
            mcd, hd = contour_distances_2d(slice1, slice2, dx=dx)

            mcd_buffer += [mcd]
            hd_buffer += [hd]

    return np.mean(mcd_buffer), np.mean(hd_buffer)


class MetricReporter(object):
    """
    Collect and report values
        self.collect_value() collects value in `report_data_dict`, which is structured as:
            self.report_data_dict = {'value_name_A': [A1, A2, ...], ... }

        self.summarise() construct the report dictionary if called, which is structured as:
            self.report = {'value_name_A': {'mean': A_mean,
                                            'std': A_std,
                                            'list': [A1, A2, ...]}
                            }
    """

    def __init__(self, id_list, save_dir, save_name='analysis_results'):
        self.id_list = id_list
        self.save_dir = save_dir
        self.save_name = save_name

        self.report_data_dict = {}
        self.report = {}

    def reset(self):
        self.report_data_dict = {}
        self.report = {}

    def collect(self, x):
        for name, value in x.items():
            if name not in self.report_data_dict.keys():
                self.report_data_dict[name] = []
            self.report_data_dict[name].append(value)

    def summarise(self):
        # summarise aggregated results to form the report dict
        for name in self.report_data_dict:
            self.report[name] = {
                'mean': np.mean(self.report_data_dict[name]),
                'std': np.std(self.report_data_dict[name]),
                'list': self.report_data_dict[name]
            }

    def save_mean_std(self):
        report_mean_std = {}
        for metric_name in self.report:
            report_mean_std[metric_name + '_mean'] = self.report[metric_name]['mean']
            report_mean_std[metric_name + '_std'] = self.report[metric_name]['std']
        # save to CSV
        csv_path = self.save_dir + f'/{self.save_name}.csv'
        save_dict_to_csv(report_mean_std, csv_path)

    def save_df(self):
        # method_column = [str(model_name)] * len(self.id_list)
        # df_dict = {'Method': method_column, 'ID': self.id_list}
        # df_dict = {'ID': self.id_list}
        df_dict = {}
        for metric_name in self.report:
            df_dict[metric_name] = self.report[metric_name]['list']

        df = pd.DataFrame(data=df_dict)
        df.to_pickle(self.save_dir + f'/{self.save_name}_df.pkl')
        df.to_csv(self.save_dir + f'/{self.save_name}_df.csv')
