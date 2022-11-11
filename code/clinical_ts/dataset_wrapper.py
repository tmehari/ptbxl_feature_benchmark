from .ecg_utils import *
from .timeseries_utils import TimeseriesDatasetCrops, reformat_as_memmap, load_dataset
import pdb

from .create_logger import create_logger
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pathlib import Path
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
try:
    import pickle5 as pickle
except ImportError as e:
    import pickle

logger = create_logger(__name__)

class Transformation:
    def __init__(self, *args, **kwargs):
        self.params = kwargs

    def get_params(self):
        return self.params

class ToTensor(Transformation):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, transpose_data=True, transpose_label=False):
        super(ToTensor, self).__init__(
            transpose_data=transpose_data, transpose_label=transpose_label)
        # swap channel and time axis for direct application of pytorch's convs
        self.transpose_data = transpose_data
        self.transpose_label = transpose_label

    def __call__(self, sample):

        def _to_tensor(data, transpose=False):
            if(isinstance(data, np.ndarray)):
                if(transpose):  # seq,[x,y,]ch
                    return torch.from_numpy(np.moveaxis(data, -1, 0))
                else:
                    return torch.from_numpy(data)
            else:  # default_collate will take care of it
                return data

        data, label = sample

        if not isinstance(data, tuple):
            data = _to_tensor(data, self.transpose_data)
        else:
            data = tuple(_to_tensor(x, self.transpose_data) for x in data)

        if not isinstance(label, tuple):
            label = _to_tensor(label, self.transpose_label)
        else:
            label = tuple(_to_tensor(x, self.transpose_label) for x in label)

        return data, label  # returning as a tuple (potentially of lists)

    def __str__(self):
        return "ToTensor"
    
    

class DataSetWrapper(object):

    def __init__(self, batch_size, num_workers, target_folders, data_modifiers, input_size=250, transformations=None, t_params=None, test=False, stability_training=False,
                 domain_adaptation=False, shuffle_train=True, drop_last=True, dnl_training=False, second_val_ds=False, match_label_distributions=False):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_folders = [Path(target_folder)
                               for target_folder in target_folders]
        self.input_size = input_size
        self.val_ds_idmap = None
        self.train_ds_size = 0
        self.val_ds_size = 0
        self.stability_training = stability_training
        self.test = test
        self.domain_adaptation = domain_adaptation
        self.second_ds = False
        self.shuffle_train = shuffle_train
        self.drop_last = drop_last
        self.dnl_training = dnl_training
        self.second_val_ds = second_val_ds
        self.match_label_distributions = match_label_distributions
        # self.filter_labels = ['IAVB', 'AF', 'AFL', 'Brady', 'CRBBB', 'IRBBB', 'LAnFB', 'LAD', 'LBBB', 'LQRSV', 'NSIVCB',
        #                       'PR', 'PAC', 'PVC', 'LPR', 'LQT', 'QAb', 'RAD', 'RBBB', 'SA', 'SB', 'NSR', 'STach', 'SVPB', 'TAb', 'TInv', 'VPB'] 3kg

        self.data_modifiers = data_modifiers
        self.lbl_itos=None

    def get_data_loaders(self):
        trainSampler = None
        validSampler = None
        train_ds, val_ds = self._get_datasets(
                self.target_folders[0], transforms=ToTensor())
        self.val_ds_idmap = val_ds.get_id_mapping()
       
        train_loader = DataLoader(train_ds, batch_size=self.batch_size,
                                      num_workers=self.num_workers, pin_memory=True, shuffle=self.shuffle_train, drop_last=self.drop_last, timeout=1)
        valid_loader = DataLoader(val_ds, batch_size=self.batch_size,
                                      shuffle=False, num_workers=self.num_workers, pin_memory=True)

        self.train_ds_size = len(train_ds)
        self.val_ds_size = len(val_ds)
        return train_loader, valid_loader

    def get_training_params(self):
        chunkify_train = False
        chunkify_valid = True
        chunk_length_train = self.input_size  # target_fs*6
        chunk_length_valid = self.input_size
        min_chunk_length = self.input_size  # chunk_length
        stride_length_train = chunk_length_train//4  # chunk_length_train//8
        stride_length_valid = self.input_size//2  # chunk_length_valid

        copies_valid = 0  # >0 should only be used with chunkify_valid=False
        return chunkify_train, chunkify_valid, chunk_length_train, chunk_length_valid, min_chunk_length, stride_length_train, stride_length_valid, copies_valid

    def get_folds(self, target_folder):
        if self.test:
            valid_fold = 10
            test_fold = 9
        else:
            valid_fold = 9
            test_fold = 10
        if "thew" in str(target_folder) or "chap" in str(target_folder) or "cinc" in str(target_folder):
            valid_fold -= 1
            test_fold -= 1

        train_folds = []
        train_folds = list(range(1, 11))
        train_folds.remove(test_fold)
        train_folds.remove(valid_fold)
        train_folds = np.array(train_folds)
        train_folds = train_folds - \
            1 if "thew" in str(target_folder) or "zheng" in str(
                target_folder) else train_folds
        return train_folds, valid_fold, test_fold

    def get_dfs(self, df_mapped, target_folder, second_ds):
        train_folds, valid_fold, test_fold = self.get_folds(target_folder)
        self.train_folds = train_folds
        self.valid_fold = valid_fold
        self.test_fold = test_fold
        # & (df_mapped.label.apply(lambda x: np.sum(x) > 0))
        df_train = df_mapped[(df_mapped.strat_fold.apply(
            lambda x: x in train_folds))]
        df_valid = df_mapped[(df_mapped.strat_fold == valid_fold)]  # & (
        # df_mapped.label.apply(lambda x: np.sum(x) > 0))]
        df_test = df_mapped[(df_mapped.strat_fold == test_fold)]  # & (
        # df_mapped.label.apply(lambda x: np.sum(x) > 0))]

        
        for data_modifier in self.data_modifiers:
            df_train, df_valid, df_test = data_modifier.modify_dfs(
                    self, df_train, df_valid, df_test)
                
        self.df_train = df_train
        self.df_valid = df_valid
        self.df_test = df_test

        return df_train, df_valid, df_test

    def _get_datasets(self, target_folder, transforms=None, second_ds=False):
        if second_ds:
            logger.info("load second dataset for auxiliary task")

        logger.info("get dataset from " + str(target_folder))
        chunkify_train, chunkify_valid, chunk_length_train, chunk_length_valid, min_chunk_length, stride_length_train, stride_length_valid, copies_valid = self.get_training_params()

        ############### Load dataframe with memmap indices ##################

        df_mapped, lbl_itos, mean, std = load_dataset(target_folder)

        df_train, df_valid, df_test = self.get_dfs(
            df_mapped, target_folder, second_ds)

        logger.info("num samples train: {}".format(len(df_train)))
        ################## create datasets ########################
        train_ds = TimeseriesDatasetCrops(df_train, self.input_size, num_classes=self.num_classes, data_folder=target_folder, chunk_length=chunk_length_train if chunkify_train else 0,
                                          min_chunk_length=min_chunk_length, stride=stride_length_train, transforms=transforms, annotation=False, col_lbl="label", memmap_filename=target_folder/("memmap.npy"))
        val_ds = TimeseriesDatasetCrops(df_valid, self.input_size, num_classes=self.num_classes, data_folder=target_folder, chunk_length=chunk_length_valid if chunkify_valid else 0,
                                        min_chunk_length=min_chunk_length, stride=stride_length_valid, transforms=transforms, annotation=False, col_lbl="label", memmap_filename=target_folder/("memmap.npy"))

        self.df_train = df_train
        self.df_valid = df_valid
        self.df_test = df_test
        return train_ds, val_ds
