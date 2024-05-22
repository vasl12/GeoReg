import torch
from lightning import LightningDataModule
import os
import random
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from data.utils import _load3d, _crop_and_pad, _normalise_intensity, _to_tensor, _load2d, _magic_slicer
import matplotlib.pyplot as plt

class _BaseDataset(Dataset):
    """Base dataset class"""
    def __init__(self, data_dir_path):
        super(_BaseDataset, self).__init__()
        self.data_dir = data_dir_path
        assert os.path.exists(data_dir_path), f"Data dir does not exist: {data_dir_path}"
        self.subject_list = sorted(os.listdir(self.data_dir))
        self.data_path_dict = dict()

    def _set_path(self, index):
        """ Set the paths of data files to load and the keys in data_dict"""
        raise NotImplementedError

    def __getitem__(self, index):
        """ Load data and pre-process """
        raise NotImplementedError

    def __len__(self):
        return len(self.subject_list)


class BrainMRInterSubj3D(_BaseDataset):
    def __init__(self,
                 data_dir_path,
                 crop_size,
                 evaluate=False,
                 modality='t1t1',
                 atlas_path=None):
        super(BrainMRInterSubj3D, self).__init__(data_dir_path)
        self.evaluate = evaluate
        self.crop_size = crop_size
        self.img_keys = ['fixed', 'moving']
        self.modality = modality
        self.atlas_path = atlas_path

    def _set_path(self, index):
        # choose the fixed and moving subjects/paths
        if self.atlas_path is None:
            self.tar_subj_id = self.subject_list[index]
            self.tar_subj_path = f'{self.data_dir}/{self.tar_subj_id}'
        else:
            self.tar_subj_path = self.atlas_path

        self.src_subj_id = random.choice(self.subject_list)
        self.src_subj_path = f'{self.data_dir}/{self.src_subj_id}'

        self.data_path_dict['fixed'] = f'{self.tar_subj_path}/T1_brain_norm.nii.gz'

        # modality
        if self.modality == 't1t1':
            self.data_path_dict['moving'] = f'{self.src_subj_path}/T1_brain_norm.nii.gz'
        elif self.modality == 't1t2':
            self.data_path_dict['moving'] = f'{self.src_subj_path}/T2_brain_norm.nii.gz'
        else:
            raise ValueError(f'Modality ({self.modality}) not recognised.')

        if self.modality == 't1t1':
            self.data_path_dict['moving_mask'] = f'{self.tar_subj_path}/T1_brain_mask.nii.gz'
            self.data_path_dict['fixed_mask'] = f'{self.src_subj_path}/T1_brain_mask.nii.gz'
        elif self.modality == 't1t2':
            self.data_path_dict['fixed_mask'] = f'{self.tar_subj_path}/T1_brain_mask.nii.gz'
            self.data_path_dict['moving_mask'] = f'{self.src_subj_path}/T2_brain_mask.nii.gz'

        else:
            raise ValueError(f'Modality ({self.modality}) not recognised.')

        self.data_path_dict['fixed_seg'] = f'{self.tar_subj_path}/T1_brain_MALPEM_tissues.nii.gz'
        self.data_path_dict['moving_seg'] = f'{self.src_subj_path}/T1_brain_MALPEM_tissues.nii.gz'

    def __getitem__(self, index):
        self._set_path(index)
        data_dict = _load3d(self.data_path_dict)
        data_dict = _crop_and_pad(data_dict, self.crop_size)
        data_dict = _normalise_intensity(data_dict, self.img_keys)
        data_dict = {key: value[:, :, :, 192//2] for key, value in data_dict.items()}
        return _to_tensor(data_dict)



class CamCANDataModule(LightningDataModule):
    def __init__(self, data_dir: str = '/home/pti/Documents/datasets/camcan_80_10_10', batch_size: int = 32, num_workers: int = 8, crop_size: list = [192, 192, 192], modality: str = 't1t1', atlas_path=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.modality = modality
        self.atlas_path = atlas_path
        self.num_workers = num_workers

    def setup(self, stage: str):
        self.train_dataset = BrainMRInterSubj3D(f'{self.data_dir}/train',
                                           crop_size=self.crop_size,
                                           modality=self.modality,
                                           atlas_path=self.atlas_path)

        self.val_dataset = BrainMRInterSubj3D(f'{self.data_dir}/val',
                                         crop_size=self.crop_size,
                                         evaluate=True,
                                         modality=self.modality,
                                         atlas_path=self.atlas_path)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          )

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          )

    def test_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          )


