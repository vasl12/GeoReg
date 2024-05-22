import lightning.pytorch as pl
from torch.utils.data import random_split, DataLoader
import torch
# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import Dataset
from typing import Dict
import SimpleITK as sitk
import numpy as np
import omegaconf


# TODO: messy dataloader, clean this up!
class RegMNIST(Dataset):
    def __init__(self, data_dir="./", train=True):
        self.data_dir = data_dir
        self.train = train
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
        self.mnist_train, self.mnist_val = random_split(
            mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
        )

        # build just a pair of images with the corresponding segmentation maps
        self.data_dict = {}

        # if self.train:
        #     self.data_dict['fixed'] = torch.tensor(self.mnist_train[0][0])
        #     self.data_dict['fixed_seg'] = torch.where(self.mnist_train[0][0] > 0, 1., 0.)
        # else:
        #     self.data_dict['fixed'] = torch.tensor(self.mnist_train[0][0])
        #     self.data_dict['fixed_seg'] = torch.where(self.mnist_train[0][0] > 0, 1., 0.)
        #
        # # create the moving image as a translated version of the fixed
        # moving = torch.zeros_like(self.data_dict['fixed'])
        # moving[:, 0:-3] = self.mnist_train[8][0][:, 3:].clone()
        # self.data_dict['moving'] = moving
        #
        # self.data_dict['moving_seg'] = torch.where(self.data_dict['moving'] > 0, 1., 0.)

        self.data_dict = self.load_image_Brain(folder=r"/home/pti/Documents/datasets/camcan_80_10_10/one/2/1/", size=(192, 192, 192))

    def __len__(self):
        length = 1
        return length

    def __getitem__(self, index: int) -> Dict:
        return self.data_dict

    def load_image_Brain(self, folder=r"/home/pti/Documents/datasets/camcan_80_10_10/one/1/1/", size=(192, 192, 192)):
        # Size of data, per image pair
        image_sizes = [
            0,
            [256, 256, 256],
            # [192, 192, 192],
        ]

        # Scale of data, per image pair
        voxel_sizes = [
            0,
            [1, 1, 1],
        ]

        shape = size

        batch = {}

        fixed = load_nifty(f'{folder}/2.nii.gz', shape)[None, 192//2]
        fixed = normalise_intensity(fixed, min_in=0.0, max_in=1.0, min_out=0.0, max_out=1.0)
        batch['fixed_seg'] = load_nifty(f'{folder}/2_seg.nii.gz', shape)[None, 192//2]
        batch['fixed_mask'] = load_nifty(f'{folder}/2_mask.nii.gz', shape)[None, 192//2]

        moving = load_nifty(f'{folder}/1.nii.gz', shape)[None, 192//2]
        moving = normalise_intensity(moving, min_in=0.0, max_in=1.0, min_out=0.0, max_out=1.0)
        batch['moving_seg'] = load_nifty(f'{folder}/1_seg.nii.gz', shape)[None, 192//2]
        batch['moving_mask'] = load_nifty(f'{folder}/1_mask.nii.gz', shape)[None, 192//2]

        batch['fixed'] = torch.FloatTensor(fixed)
        batch['moving'] = torch.FloatTensor(moving)

        angle = (torch.rand(1) * 90 - 45).item()
        t = torch.rand(2) * 50 - 25
        scale = (torch.rand(1) * (1.2 - 0.8) + 0.8).item()
        shear = torch.rand(2) * 10 - 5

        with open('/home/pti/Documents/git/IDIR/output.txt', 'a') as f:
            f.write(f'Angle: {angle}\ntranslation: {t}\nscale: {scale}\nshear: {shear}\n')

        # batch['moving'] = torchvision.transforms.functional.affine(batch['moving'], angle=angle, translate=[t[0], t[1]], scale=scale, shear=[shear[0], shear[1]])
        # batch['moving_mask'] = torchvision.transforms.functional.affine(batch['moving_mask'], angle=angle, translate=[t[0], t[1]], scale=scale, shear=[shear[0], shear[1]])
        # batch['moving_seg'] = torchvision.transforms.functional.affine(batch['moving_seg'], angle=angle, translate=[t[0], t[1]], scale=scale, shear=[shear[0], shear[1]])

        batch['voxel_size'] = [1., 1., 1.]

        return batch

def load_nifty(image_path, image_size):

    image = sitk.ReadImage(image_path)
    image = sitk.GetArrayFromImage(image)
    image = crop_and_pad(image.reshape(1, *image.shape), new_size=image_size)

    image = torch.tensor(image).squeeze(0)

    return image


def normalise_intensity(x,
                        mode="minmax",
                        min_in=0.0,
                        max_in=255.0,
                        min_out=0.0,
                        max_out=1.0,
                        clip=False,
                        clip_range_percentile=(0.05, 99.95),
                        ):
    """
    Intensity normalisation (& optional percentile clipping)
    for both Numpy Array and Pytorch Tensor of arbitrary dimensions.

    The "mode" of normalisation indicates different ways to normalise the intensities, including:
    1) "meanstd": normalise to 0 mean 1 std;
    2) "minmax": normalise to specified (min, max) range;
    3) "fixed": normalise with a fixed ratio

    Args:
        x: (ndarray / Tensor, shape (N, *size))
        mode: (str) indicate normalisation mode
        min_in: (float) minimum value of the input (assumed value for fixed mode)
        max_in: (float) maximum value of the input (assumed value for fixed mode)
        min_out: (float) minimum value of the output
        max_out: (float) maximum value of the output
        clip: (boolean) value clipping if True
        clip_range_percentile: (tuple of floats) percentiles (min, max) to determine the thresholds for clipping

    Returns:
        x: (same as input) in-place op on input x
    """

    # determine data dimension
    dim = x.ndim - 1
    image_axes = tuple(range(1, 1 + dim))  # (1,2) for 2D; (1,2,3) for 3D

    # for numpy.ndarray
    if type(x) is np.ndarray:
        # Clipping
        if clip:
            # intensity clipping
            clip_min, clip_max = np.percentile(x, clip_range_percentile, axis=image_axes, keepdims=True)
            x = np.clip(x, clip_min, clip_max)

        # Normalise meanstd
        if mode == "meanstd":
            mean = np.mean(x, axis=image_axes, keepdims=True)  # (N, *range(dim))
            std = np.std(x, axis=image_axes, keepdims=True)  # (N, *range(dim))
            x = (x - mean) / std  # axis should match & broadcast

        # Normalise minmax
        elif mode == "minmax":
            min_in = np.amin(x, axis=image_axes, keepdims=True)  # (N, *range(dim))
            max_in = np.amax(x, axis=image_axes, keepdims=True)  # (N, *range(dim)))
            x = (x - min_in) * (max_out - min_out) / (max_in - min_in + 1e-12) + min_out # (!) multiple broadcasting)

        # Fixed ratio
        elif mode == "fixed":
            x = (x - min_in) * (max_out - min_out) / (max_in - min_in + 1e-12)

        else:
            raise ValueError("Intensity normalisation mode not understood."
                             "Expect either one of: 'meanstd', 'minmax', 'fixed'")

        # cast to float 32
        x = x.astype(np.float32)

    # for torch.Tensor
    elif type(x) is torch.Tensor:
        # todo: clipping not supported at the moment (requires Pytorch version of the np.percentile()

        # Normalise meanstd
        if mode == "meanstd":
            mean = torch.mean(x, dim=image_axes, keepdim=True)  # (N, *range(dim))
            std = torch.std(x, dim=image_axes, keepdim=True)  # (N, *range(dim))
            x = (x - mean) / std  # axis should match & broadcast

        # Normalise minmax
        elif mode == "minmax":
            # get min/max across dims by flattening first
            min_in = x.flatten(start_dim=1, end_dim=-1).min(dim=1)[0].view(-1, *(1,)*dim)  # (N, (1,)*dim)
            max_in = x.flatten(start_dim=1, end_dim=-1).max(dim=1)[0].view(-1, *(1,)*dim)  # (N, (1,)*dim)
            x = (x - min_in) * (max_out - min_out) / (max_in - min_in + 1e-12) + min_out  # (!) multiple broadcasting)

        # Fixed ratio
        elif mode == "fixed":
            x = (x - min_in) * (max_out - min_out) / (max_in - min_in + 1e-12)

        else:
            raise ValueError("Intensity normalisation mode not recognised."
                             "Expect: 'meanstd', 'minmax', 'fixed'")

        # cast to float32
        x = x.float()

    else:
        raise TypeError("Input data type not recognised, support numpy.ndarray or torch.Tensor")
    return x




def crop_and_pad(x, new_size=192, mode="constant", **kwargs):
    """
    Crop and/or pad input to new size.
    (Adapted from DLTK: https://github.com/DLTK/DLTK/blob/master/dltk/io/preprocessing.py)

    Args:
        x: (np.ndarray) input array, shape (N, H, W) or (N, H, W, D)
        new_size: (int or tuple/list) new size excluding the batch size
        mode: (string) padding value filling mode for numpy.pad() (compulsory in Numpy v1.18)
        kwargs: additional arguments to be passed to np.pad

    Returns:
        (np.ndarray) cropped and/or padded input array
    """
    assert isinstance(x, (np.ndarray, np.generic))
    new_size = param_ndim_setup(new_size, ndim=x.ndim - 1)

    dim = x.ndim - 1
    sizes = x.shape[1:]

    # Initialise padding and slicers
    to_padding = [[0, 0] for i in range(x.ndim)]
    slicer = [slice(0, x.shape[i]) for i in range(x.ndim)]

    # For each dimensions except the dim 0, set crop slicers or paddings
    for i in range(dim):
        if sizes[i] < new_size[i]:
            to_padding[i+1][0] = (new_size[i] - sizes[i]) // 2
            to_padding[i+1][1] = new_size[i] - sizes[i] - to_padding[i+1][0]
        else:
            # Create slicer object to crop each dimension
            crop_start = int(np.floor((sizes[i] - new_size[i]) / 2.))
            crop_end = crop_start + new_size[i]
            slicer[i+1] = slice(crop_start, crop_end)

    return np.pad(x[tuple(slicer)], to_padding, mode=mode, **kwargs)

def param_ndim_setup(param, ndim):
    """
    Check dimensions of paramters and extend dimension if needed.

    Args:
        param: (int/float, tuple or list) check dimension match if tuple or list is given,
                expand to `dim` by repeating if a single integer/float number is given.
        ndim: (int) data/model dimension

    Returns:
        param: (tuple)
    """
    if isinstance(param, (int, float)):
        param = (param,) * ndim
    elif isinstance(param, (tuple, list, omegaconf.listconfig.ListConfig)):
        assert len(param) == ndim, \
            f"Dimension ({ndim}) mismatch with data"
        param = tuple(param)
    else:
        raise TypeError("Parameter type not int, tuple or list")
    return param



class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=False)
        MNIST(self.data_dir, train=False, download=False)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            # mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            # self.mnist_train, self.mnist_val = random_split(
            #     mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
            # )
            self.mnist_train = RegMNIST(self.data_dir, train=True)
            self.mnist_val = RegMNIST(self.data_dir, train=False)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=32)