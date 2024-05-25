import os
from typing import List, Tuple, Optional
import lightning as pl
import torch
import numpy as np
import json
from data.utils import _create_tensor_from_torchio_subject, _to_tensor, _apply_augs
from utils.utils import make_coordinate_tensor_like
from monai.transforms import (
    Compose,
    LoadImaged,
    MapTransform,
    RandAffined,
    Resized,
    ScaleIntensityRanged,
)
import pandas as pd
from monai.data import Dataset, CacheDataset, DataLoader



class NLSTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str,
                 batch_size=1,
                 crop_size=[224, 192, 224],
                 with_affine=False,
                 angle=0,
                 translation=0,
                 scale=0.,
                 num_workers=32,
                 full_res_training=False,
                 **kwargs,
                 ):

        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_shape = crop_size
        self.with_affine = with_affine
        self.angle = angle
        self.translation = translation
        self.scale = scale
        self.num_workers = num_workers
        self.num_seg_classes = 2

        self.train_files, self.val_files, self.test_files = self.get_files(self.data_dir)

        if full_res_training:
            target_res = crop_size
            spatial_size = [
                -1,
                -1,
                -1,
            ]  # for Resized transform, [-1, -1, -1] means no resizing, use this when training challenge model
        else:
            target_res = crop_size
            spatial_size = target_res  # downsample to 96^3 voxels for faster training on resized data (good for testing)

        self.train_transforms = Compose(
            [
                LoadImaged(keys=["fixed", "moving", "fixed_seg", "moving_seg"], ensure_channel_first=True),
                LoadKeypointsd(
                    keys=["fixed_landmarks", "moving_landmarks"],
                ),
                ScaleIntensityRanged(
                    keys=["fixed", "moving"],
                    a_min=-1200,
                    a_max=400,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                Resized(
                    keys=["fixed", "moving", "fixed_seg", "moving_seg"],
                    mode=("trilinear", "trilinear", "nearest", "nearest"),
                    align_corners=(True, True, None, None),
                    spatial_size=spatial_size,
                ),
                # RandAffined(
                #     keys=["fixed", "moving", "fixed_seg", "moving_seg"],
                #     mode=("bilinear", "bilinear", "nearest", "nearest"),
                #     prob=0.8,
                #     shear_range=0.2,
                #     translate_range=int(25 * target_res[0] / 224),
                #     rotate_range=np.pi / 180 * 15,
                #     scale_range=0.2,
                #     padding_mode=("zeros", "zeros", "zeros", "zeros"),
                # ),
                TransformKeypointsd(
                    keys_keypoints=["fixed_landmarks", "moving_landmarks"],
                    keys_images=["fixed", "moving"],
                ),
            ]
        )

        self.val_transforms = Compose(
            [
                LoadImaged(keys=["fixed", "moving", "fixed_seg", "moving_seg"], ensure_channel_first=True),
                ScaleIntensityRanged(
                    keys=["fixed", "moving"],
                    a_min=-1200,
                    a_max=400,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                Resized(
                    keys=["fixed", "moving", "fixed_seg", "moving_seg"],
                    mode=("trilinear", "trilinear", "nearest", "nearest"),
                    align_corners=(True, True, None, None),
                    spatial_size=spatial_size,
                ),
                LoadKeypointsd(
                    keys=["fixed_landmarks", "moving_landmarks"],
                ),
                TransformKeypointsd(
                    keys_keypoints=["fixed_landmarks", "moving_landmarks"],
                    keys_images=["fixed", "moving"],
                ),
            ]
        )
        self.test_transforms = Compose(
            [
                LoadImaged(keys=["fixed", "moving", "fixed_seg", "moving_seg"],
                           ensure_channel_first=True),
                ScaleIntensityRanged(
                    keys=["fixed", "moving"],
                    a_min=-1200,
                    a_max=400,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                Resized(
                    keys=["fixed", "moving", "fixed_seg", "moving_seg"],
                    mode=("trilinear", "trilinear", "nearest", "nearest"),
                    align_corners=(True, True, None, None),
                    spatial_size=spatial_size,
                ),
                LoadKeypointsd(
                    keys=["fixed_landmarks", "moving_landmarks"],
                ),
                TransformKeypointsd(
                    keys_keypoints=["fixed_landmarks", "moving_landmarks"],
                    keys_images=["fixed", "moving"],
                ),
            ]
        )


    def setup(self, stage: str = None):
        # Assign Train/val split(s) for use in Dataloaders
        if stage == "fit":
            self.train_dataset = Dataset(data=self.train_files, transform=self.train_transforms)
            self.val_dataset = Dataset(data=self.val_files, transform=self.val_transforms)
        # Assign Test split(s) for use in Dataloaders
        if stage == "test":
            self.test_dataset = Dataset(data=self.test_files, transform=self.test_transforms)


    def get_files(self, data_dir):
        """
        Get L2R train/val files from NLST challenge
        """
        data_json = os.path.join(data_dir, "NLST_dataset.json")

        with open(data_json) as file:
            data = json.load(file)

        train_files = []
        for pair in data["training_paired_images"]:
            nam_fixed = os.path.basename(pair["fixed"]).split(".")[0]
            nam_moving = os.path.basename(pair["moving"]).split(".")[0]
            train_files.append(
                {
                    "fixed": os.path.join(data_dir, "imagesTr", nam_fixed + ".nii.gz"),
                    "moving": os.path.join(data_dir, "imagesTr", nam_moving + ".nii.gz"),
                    "fixed_seg": os.path.join(data_dir, "masksTr", nam_fixed + ".nii.gz"),
                    "moving_seg": os.path.join(data_dir, "masksTr", nam_moving + ".nii.gz"),
                    "fixed_landmarks": os.path.join(data_dir, "keypointsTr", nam_fixed + ".csv"),
                    "moving_landmarks": os.path.join(data_dir, "keypointsTr", nam_moving + ".csv"),
                }
            )

        val_files = []
        for pair in data["registration_val"]:
            nam_fixed = os.path.basename(pair["fixed"]).split(".")[0]
            nam_moving = os.path.basename(pair["moving"]).split(".")[0]
            val_files.append(
                {
                    "fixed": os.path.join(data_dir, "imagesTr", nam_fixed + ".nii.gz"),
                    "moving": os.path.join(data_dir, "imagesTr", nam_moving + ".nii.gz"),
                    "fixed_seg": os.path.join(data_dir, "masksTr", nam_fixed + ".nii.gz"),
                    "moving_seg": os.path.join(data_dir, "masksTr", nam_moving + ".nii.gz"),
                    "fixed_landmarks": os.path.join(data_dir, "keypointsTr", nam_fixed + ".csv"),
                    "moving_landmarks": os.path.join(data_dir, "keypointsTr", nam_moving + ".csv"),
                }
            )

        test_files = []
        for pair in data["test_paired_images"]:
            nam_fixed = os.path.basename(pair["fixed"]).split(".")[0]
            nam_moving = os.path.basename(pair["moving"]).split(".")[0]
            test_files.append(
                {
                    "fixed": os.path.join(data_dir, "imagesTr", nam_fixed + ".nii.gz"),
                    "moving": os.path.join(data_dir, "imagesTr", nam_moving + ".nii.gz"),
                    "fixed_seg": os.path.join(data_dir, "masksTr", nam_fixed + ".nii.gz"),
                    "moving_seg": os.path.join(data_dir, "masksTr", nam_moving + ".nii.gz"),
                    "fixed_landmarks": os.path.join(data_dir, "keypointsTr", nam_fixed + ".csv"),
                    "moving_landmarks": os.path.join(data_dir, "keypointsTr", nam_moving + ".csv"),
                }
            )

        return train_files, val_files, test_files

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)





# class NLSTDataset(Dataset):
#     def __init__(self, data, transforms=None, with_affine=False, angle=0, translation=0, scale=0.):
#         self.data_dir = data
#         self.transforms = transforms
#         self.with_affine = with_affine
#         self.angle = angle
#         self.translation = translation
#         self.scale = scale
#
#     @property
#     def num_seg_classes(self) -> Optional[int]:
#         return 2
#
#     def __getitem__(self, index):
#
#         # return images, masks#, landmarks, case_id
#         data_dict = {'fixed': images[0][None],
#                      'moving': images[1][None],
#                      'fixed_seg': masks[0][None].float(),
#                      'moving_seg': masks[1][None].float(),
#                      'fixed_mask': masks[0][None].float(),
#                      'moving_mask': masks[1][None].float(),
#                      'fixed_landmarks': fixed_landmarks[None],
#                      'moving_landmarks': moving_landmarks[None],
#                      'landmark_size': torch.tensor(landmark_size)[None]
#                     }
#
#         # apply geometric augmentation
#         if self.with_affine:
#             data_dict = _apply_augs(data_dict, angle=self.angle, translation=self.translation, scale=self.scale)
#         return data_dict

    # def __len__(self):
    #     return len(self.samples)

class LoadKeypointsd(MapTransform):
    """
    Load keypoints from csv file
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            keypoints = d[key]
            keypoints = pd.read_csv(keypoints, header=None)
            keypoints = keypoints.to_numpy()
            keypoints = torch.as_tensor(keypoints)
            d[key] = keypoints  # [N, 3]
        return d


class TransformKeypointsd(MapTransform):
    """
    Applies any potential linear image transformation to keypoint values
    """

    def __init__(self, keys_keypoints, keys_images, ras=False):
        # super.__init__(self)
        self.keys_keypoints = keys_keypoints
        self.keys_images = keys_images
        self.ras = ras

    def __call__(self, data):
        d = dict(data)
        for kp, ki in zip(self.keys_keypoints, self.keys_images):
            # Get image meta data
            image = d[ki]
            meta = image.meta
            # Get keypoints
            keypoints_ijk = d[kp]
            # Get transformation (in voxel space)
            affine = meta["affine"]
            original_affine = torch.as_tensor(meta["original_affine"], dtype=affine.dtype, device=affine.device)
            transforms_affine = (
                original_affine.inverse() @ affine
            )  # Assumes: affine = original_affine @ transforms_affine
            transforms_affine = transforms_affine.inverse()
            if self.ras:
                # RAS space
                transforms_affine = original_affine @ transforms_affine
            # Apply transformation to keypoints
            keypoints_ijk_moved = torch.cat((keypoints_ijk, torch.ones((keypoints_ijk.shape[0]), 1)), dim=1)
            keypoints_ijk_moved = (transforms_affine @ keypoints_ijk_moved.T).T
            keypoints_ijk_moved = keypoints_ijk_moved[:, :3]
            keypoints_ijk_moved = keypoints_ijk_moved.float()

            d[kp] = keypoints_ijk_moved  # [N, 3]

        return d

