""" Dataset helper functions """
import os
import random
import numpy as np
from omegaconf.listconfig import ListConfig
import torch
from utils.image import crop_and_pad, normalise_intensity, apply_randconv
from utils.image_io import load_nifti, load_nifti_center_slice
from data.transforms import random_fractal_deforms, random_affine_displacements_2d, random_affine_displacements_3d, \
    random_affine_displacements
from transformation.transformation import warp
import SimpleITK as sitk


def _to_tensor(data_dict):
    # cast to Pytorch Tensor
    for name, data in data_dict.items():
        data_dict[name] = torch.from_numpy(data).float()
    return data_dict


def _crop_and_pad(data_dict, crop_size):
    # cropping and padding
    for name, data in data_dict.items():
        data_dict[name] = crop_and_pad(data, new_size=crop_size)
    return data_dict


def _create_masks(data_dict, keys=None):
    for key, value in data_dict.items():
        if key in keys:
            data_dict[key] = create_mask(data_dict[key])

    return data_dict


def create_mask(img):
    mask = img
    mask[0][np.where(img[0] >= 0.01)] = 1
    mask[0][np.where(img[0] < 0.01)] = 0
    return mask


def _normalise_intensity(data_dict, keys=None, vmin=0., vmax=1.):
    """ Normalise intensity of data in `data_dict` with `keys` """
    if keys is None:
        keys = {'fixed', 'moving', 'fixed_original'}

    # images in one pairing should be normalised using the same scaling
    vmin_in = np.amin(np.array([data_dict[k] for k in keys]))
    vmax_in = np.amax(np.array([data_dict[k] for k in keys]))

    for k, x in data_dict.items():
        if k in keys:
            data_dict[k] = normalise_intensity(x,
                                               min_in=vmin_in, max_in=vmax_in,
                                               min_out=vmin, max_out=vmax,
                                               mode="minmax", clip=True)
    return data_dict


def _shape_checker(data_dict):
    """Check if all data points have the same shape
    if so return the common shape, if not print data type"""
    data_shapes_dict = {n: x.shape for n, x in data_dict.items()}
    shapes = [x for _, x in data_shapes_dict.items()]
    if all([s == shapes[0] for s in shapes]):
        common_shape = shapes[0]
        return common_shape
    else:
        raise AssertionError(f'Not all data points have the same shape, {data_shapes_dict}')


def _magic_slicer(data_dict, slice_range=None, slicing=None):
    """Select all slices, one random slice, or some slices
    within `slice_range`, according to `slicing`
    """
    # slice selection
    num_slices = _shape_checker(data_dict)[0]

    # set range for slicing
    if slice_range is None:
        # all slices if not specified
        slice_range = (0, num_slices)
    else:
        # check slice_range
        assert isinstance(slice_range, (tuple, list, ListConfig))
        assert len(slice_range) == 2
        assert all(isinstance(s, int) for s in slice_range)
        assert slice_range[0] < slice_range[1]
        assert all(0 <= s <= num_slices for s in slice_range)

    # select slice(s)
    if slicing is None:
        # all slices within slice_range
        slicer = slice(slice_range[0], slice_range[1])

    elif slicing == 'random':
        # randomly choose one slice within range
        z = random.randint(slice_range[0], slice_range[1]-1)
        slicer = slice(z, z + 1)  # use slicer to keep dim

    elif isinstance(slicing, (list, tuple, ListConfig)):
        # slice several slices specified by slicing
        assert all(0 <= i <= 1 for i in slicing), f'Relative slice positions {slicing} need to be within [0, 1]'
        slicer = tuple(int(i * (slice_range[1] - slice_range[0])) + slice_range[0] for i in slicing)

    else:
        raise ValueError(f'Slicing mode {slicing} not recognised.')

    # slicing
    for name, data in data_dict.items():
        data_dict[name] = data[slicer, ...]  # (N, H, W)

    return data_dict


def _load2d(data_path_dict):
    data_dict = dict()
    for name, data_path in data_path_dict.items():
        # image is saved in shape (H, W, N) ->  (N, H, W)
        data_dict[name] = load_nifti(data_path).transpose(2, 0, 1)
    return data_dict


def _load3d(data_path_dict):
    data_dict = dict()
    for name, data_path in data_path_dict.items():
        # image is saved in shape (H, W, D) -> (ch=1, H, W, D)
        data_dict[name] = load_nifti(data_path)[np.newaxis, ...]
    return data_dict


def _load3d_center_slice(data_path_dict):
    data_dict = dict()
    for name, data_path in data_path_dict.items():
        # image is saved in shape (H, W, D) -> (ch=1, H, W, D)
        data_dict[name] = load_nifti_center_slice(data_path)[np.newaxis, ...]
    return data_dict


def _apply_randconv(data_dict, dim=2):
    mix = np.random.randint(2)
    data_dict['moving'], rand_conv_mov = apply_randconv(data_dict['moving'], mask=data_dict['moving_mask'], K=0, mix=mix, p=0.1, dim=dim, random_convolution=None)
    # data_dict['fix_mov'], _ = apply_randconv(data_dict['fix_mov'], mask=data_dict['fixed_mask'], K=0, mix=mix, p=0.1, dim=dim, random_convolution=rand_conv_mov)
    data_dict['fixed_original'], _ = apply_randconv(data_dict['fixed_original'], mask=data_dict['fixed_mask'], K=0, mix=mix, p=0.1, dim=dim, random_convolution=rand_conv_mov)
    data_dict['fixed'], rand_conv_fix = apply_randconv(data_dict['fixed'], mask=data_dict['fixed_mask'], K=0, mix=mix, p=0.1, dim=dim, random_convolution=None)
    data_dict['moving_original'], _ = apply_randconv(data_dict['moving_original'], mask=data_dict['moving_mask'], K=0, mix=mix, p=0.1, dim=dim, random_convolution=rand_conv_fix)

    return data_dict


def _create_tensor_from_torchio_subject(subject, image_names=('Fixed', 'Moving'), mask_names=('FixedMask', 'MovingMask')):
    images = []
    masks = []
    image_count = 0
    mask_count = 0
    for idx, (image_name, image) in enumerate(subject.items()):

        if image_name in image_names:
            if image_count == 0:
                images = image.tensor
                image_count += 1
            else:
                images = torch.cat((images, image.tensor), 0)

        if image_name in mask_names:
            if mask_count == 0:
                masks = image.tensor
                mask_count += 1
            else:
                masks = torch.cat((masks, image.tensor), 0)

    return images, masks


def _apply_augs(data_dict, angles=(0., ), translation=(0.,), scale=(0.,), do_affine=True, do_deform=False, intersubject=False):
    # Since we can't easily get the inverse of the displacement field, the deformed image becomes the
    # fixed image and we must learn to recreate the displacement field to be applied to untransformed moving image.
    batch_size, channels, *spatial_dims = data_dict['moving'][None].shape
    num_spatial_dims = len(spatial_dims)
    disp = torch.zeros((batch_size, *spatial_dims, len(spatial_dims)))
    if do_affine:
        disp_affine = random_affine_displacements(num_spatial_dims, (batch_size, channels, *spatial_dims),
                                                  theta_range=angles, trans_range=translation, scale_range=scale)
        disp += disp_affine
        data_dict['disp_affine'] = disp_affine[0]
    if do_deform:
        disp += random_fractal_deforms((1, *data_dict['moving'].shape))
    if not intersubject:
        data_dict['fixed'] = warp(data_dict['moving'][None].clone(), disp, interp_mode='bilinear', normalise=False)[0]
        if "fixed_mask" in data_dict:
            data_dict['fixed_mask'] = warp(data_dict['moving_mask'][None].clone(), disp, interp_mode='nearest', normalise=False)[0]
        if "fixed_seg" in data_dict:
            data_dict['fixed_seg'] = warp(data_dict['moving_seg'][None].clone(), disp, interp_mode='nearest', normalise=False)[0]
    else:
        data_dict['fixed'] = warp(data_dict['fixed'][None].clone(), disp, interp_mode='bilinear', normalise=False)[0]
        if "fixed_mask" in data_dict:
            data_dict['fixed_mask'] = warp(data_dict['fixed_mask'][None].clone(), disp, interp_mode='nearest', normalise=False)[0]
        if "fixed_seg" in data_dict:
            data_dict['fixed_seg'] = warp(data_dict['fixed_seg'][None].clone(), disp, interp_mode='nearest', normalise=False)[0]
    data_dict['disp'] = disp[0]
    return data_dict

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)


def augment(img_list, hflip=True, rot=True, split='val'):
    # horizontal flip OR rotate
    hflip = hflip and (split == 'train' and random.random() < 0.5)

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        return img

    return [_augment(img) for img in img_list]


def transform2numpy(img):
    img = np.array(img)
    img = img.astype(np.float32)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return img


def transform2tensor(img, min_max=(0, 1)):
    # HWC to CHW
    img = torch.from_numpy(np.ascontiguousarray(
        np.transpose(img, (2, 0, 1)))).float()
    # to range min_max
    img = img*(min_max[1] - min_max[0]) + min_max[0]
    return img


def transform_augment(img_list, split='val', min_max=(0, 1)):
    ret_img = []
    img_list = augment(img_list, split=split)
    for img in img_list:
        img = transform2numpy(img)
        img = transform2tensor(img, min_max)
        ret_img.append(img)
    return ret_img


def _load3d_abd(data_path_dict, transpose=False):
    data_dict = dict()
    for name, data_path in data_path_dict.items():
        sitk_img = sitk.DICOMOrient(sitk.ReadImage(data_path), 'LPI')
        sitk1 = sitk_img
        data_dict[name], info = sitk_to_torch(sitk1, transpose=transpose)
        data_dict[name] = data_dict[name][np.newaxis, :, :, :]


    return data_dict


def sitk_to_torch(sitk_image, transpose=True):
    np_image = sitk.GetArrayFromImage(sitk_image)  # need to permute indices for consistency
    image_centre = np.array(sitk_image.GetOrigin()) + np.array(sitk_image.GetSpacing()) * np.array(
        sitk_image.GetSize()) / 2.0
    if transpose is True:
        np_image = np.transpose(np_image)
        tensor_info = {'spacing': sitk_image.GetSpacing(), 'origin': sitk_image.GetOrigin(),
                       'size': sitk_image.GetSize(), 'centre': image_centre}
    else:
        tensor_info = {'spacing': sitk_image.GetSpacing()[::-1], 'origin': sitk_image.GetOrigin()[::-1],
                       'size': sitk_image.GetSize()[::-1], 'centre': image_centre}

    image_tensor = torch.from_numpy(np_image)
    return image_tensor, tensor_info

