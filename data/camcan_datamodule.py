from typing import Optional

from lightning import LightningDataModule
import os
import random
random.seed(42)
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from data.utils import _load3d, _crop_and_pad, _normalise_intensity, _to_tensor, _load2d, _magic_slicer, _apply_augs, \
    _load3d_center_slice


class _BaseDataset(Dataset):
    """Base dataset class"""
    def __init__(self, data_dir_path, num_subjects=None, **kwargs):
        super(_BaseDataset, self).__init__()
        self.data_dir = data_dir_path
        assert os.path.exists(data_dir_path), f"Data dir does not exist: {data_dir_path}"
        self.subject_list = sorted(os.listdir(self.data_dir))
        if isinstance(num_subjects,  int):
            self.subject_list = self.subject_list[:num_subjects]
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
                 atlas_path=None,
                 with_affine=False,
                 with_deform=False,
                 angle=0.,
                 translation=0.,
                 scale=0.,
                 intersubject=False,
                 **kwargs):
        super(BrainMRInterSubj3D, self).__init__(data_dir_path, **kwargs)
        self.evaluate = evaluate
        self.crop_size = crop_size
        self.img_keys = ['fixed', 'moving']
        self.modality = modality
        self.atlas_path = atlas_path
        self.with_affine = with_affine
        self.with_deform = with_deform
        self.angle = angle
        self.translation = translation
        self.scale = scale
        self.intersubject = intersubject
        self.cache = []
        # self.cache_data()

    def cache_data(self):
        print("CACHING...")
        for i in range(len(self.subject_list)):
            self._set_path(i)
            data_dict = self.load_func(self.data_path_dict)
            self.cache.append(data_dict)
        print("Finished CACHING.")

    @property
    def load_func(self):
        return _load3d

    @property
    def num_seg_classes(self) -> Optional[int]:
        return 5

    def _set_path(self, index):
        # choose the fixed and moving subjects/paths
        if self.atlas_path is None:
            self.tar_subj_id = self.subject_list[index]
            self.tar_subj_path = f'{self.data_dir}/{self.tar_subj_id}'
        else:
            self.tar_subj_path = self.atlas_path

        self.data_path_dict['fixed'] = f'{self.tar_subj_path}/T1_brain_norm.nii.gz'
        if self.intersubject:
            self.src_subj_id = random.choice(self.subject_list)
        else:
            self.src_subj_id = self.tar_subj_id
        self.src_subj_path = f'{self.data_dir}/{self.src_subj_id}'

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
        if self.cache:
            data_dict = self.cache[index]
        else:
            self._set_path(index)
            data_dict = self.load_func(self.data_path_dict)
        data_dict = _crop_and_pad(data_dict, self.crop_size)
        data_dict = _normalise_intensity(data_dict, self.img_keys)
        data_dict = {key: value for key, value in data_dict.items()}
        data_dict = _to_tensor(data_dict)

        # apply geometric augmentation
        if self.with_affine or self.with_deform:
            data_dict = _apply_augs(data_dict, angles=self.angle, translation=self.translation, scale=self.scale,
                                    do_affine=self.with_affine, do_deform=self.with_deform,
                                    intersubject=self.intersubject or (self.modality == 't1t2'))
        return data_dict


class BrainMRInterSubj2D(BrainMRInterSubj3D):
    @property
    def load_func(self):
        return _load3d_center_slice


class CamCANDataModule(LightningDataModule):
    def __init__(self, data_dir: str,
                 num_train: Optional[int] = None,
                 num_val: Optional[int] = None,
                 num_test: Optional[int] = None,
                 batch_size: int = 32,
                 num_workers: int = 8,
                 crop_size: list = [192]*3,
                 same_expression: bool = True,
                 atlas_path=None,
                 with_deform: bool = False,
                 with_affine: bool = False,
                 angle: float = 0,
                 translation: float = 0,
                 scale: float = 0.,
                 intersubject: bool = False,
                 **kwargs):
        super().__init__()
        self.crop_size = crop_size
        self.modality = 't1t1' if same_expression else 't1t2'
        self.atlas_path = atlas_path
        self.with_deform = with_deform
        self.with_affine = with_affine
        self.angle = angle
        self.translation = translation
        self.scale = scale
        self.intersubject = intersubject

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    @property
    def dataset_class(self):
        return BrainMRInterSubj3D

    def setup(self, stage: str):
        if stage == 'fit':
            self.train_dataset = self.dataset_class(f'{self.data_dir}/train',
                                                    num_subjects=self.num_train,
                                                    crop_size=self.crop_size,
                                                    modality=self.modality,
                                                    atlas_path=self.atlas_path,
                                                    with_deform=self.with_deform,
                                                    with_affine=self.with_affine,
                                                    angle=self.angle,
                                                    translation=self.translation,
                                                    scale=self.scale,
                                                    intersubject=self.intersubject)

            self.val_dataset = self.dataset_class(f'{self.data_dir}/val',
                                                  num_subjects=self.num_val,
                                                  crop_size=self.crop_size,
                                                  evaluate=True,
                                                  modality=self.modality,
                                                  atlas_path=self.atlas_path,
                                                  with_deform=self.with_deform,
                                                  with_affine=self.with_affine,
                                                  angle=self.angle,
                                                  translation=self.translation,
                                                  scale=self.scale,
                                                  intersubject=self.intersubject)
        elif stage == 'test':
            self.test_dataset = self.dataset_class(f'{self.data_dir}/test',
                                                   num_subjects=self.num_test,
                                                   crop_size=self.crop_size,
                                                   evaluate=True,
                                                   modality=self.modality,
                                                   atlas_path=self.atlas_path,
                                                   with_deform=self.with_deform,
                                                   with_affine=self.with_affine,
                                                   angle=self.angle,
                                                   translation=self.translation,
                                                   scale=self.scale,
                                                   intersubject=self.intersubject)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          persistent_workers=True,
                          pin_memory=True,
                          )

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          persistent_workers=True,
                          pin_memory=True,
                          )

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          persistent_workers=True,
                          pin_memory=True,
                          )


class CamCANDataModule2D(CamCANDataModule):
    @property
    def dataset_class(self):
        return BrainMRInterSubj2D
