from .timeseries_utils import Normalize
import os
from typing import Optional, Sequence
from warnings import warn

import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader, random_split

from .dataset_wrapper import DataSetWrapper
import pdb 



class ECGDataModule(LightningDataModule):

    name = 'ecg_dataset'
    extra_args = {}

    def __init__(
            self,
            batch_size,
            target_folders,
            data_modifiers,
            data_dir: str = None,
            num_workers: int = 8,
            seed: int = 42,
            data_input_size=250,
            lbl_itos=None,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.dims = (12, data_input_size)
        # self.val_split = val_split
        self.target_folders = target_folders
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.data_input_size = data_input_size
        self.data_modifiers = data_modifiers
        self.lbl_itos = lbl_itos
        self.num_classes = len(self.lbl_itos)
        self.set_params()

    def set_params(self):
        dataset, _, _ = get_dataset(self.batch_size, self.num_workers, self.target_folders, self.data_modifiers, input_size=self.data_input_size)
        self.num_samples = dataset.train_ds_size
        self.lbl_itos = dataset.lbl_itos
        self.num_classes = len(self.lbl_itos)
        print("Number of classes", self.num_classes)
        self.val_idmap = dataset.val_ds_idmap

    def prepare_data(self):
        pass

    def train_dataloader(self):
        _, train_loader, _ = get_dataset(self.batch_size, self.num_workers, self.target_folders, self.data_modifiers, input_size=self.data_input_size)
        return train_loader

    def val_dataloader(self):
        _, _, valid_loader = get_dataset(self.batch_size, self.num_workers, self.target_folders, self.data_modifiers, input_size=self.data_input_size)
        return valid_loader

    def test_dataloader(self):
        dataset, _, test_loader = get_dataset(self.batch_size, self.num_workers, self.target_folders, self.data_modifiers, test=True, input_size=self.data_input_size)
        self.test_idmap = dataset.val_ds_idmap
        return test_loader

    def default_transforms(self):
        pass


def get_dataset(batch_size, num_workers, target_folders, data_modifiers, test=False, input_size=250):
    dataset = DataSetWrapper(batch_size, num_workers, target_folders, data_modifiers, test=test, input_size=input_size)
    train_loader, valid_loader = dataset.get_data_loaders()
    return dataset, train_loader, valid_loader
