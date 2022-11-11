from abc import ABC, abstractmethod
from .create_logger import create_logger
import numpy as np
import pandas as pd
from os.path import join, dirname

logger = create_logger(__name__)


class DataModifier(ABC):

    @abstractmethod
    def modify_dfs(self, dataset_wrapper, df_train, df_valid, df_test):
        pass

    @abstractmethod
    def modify_dfs_of_second_ds(self, dataset_wrapper, df_train, df_valid, df_test):
        pass


class ToOneHot(DataModifier):
    def __init__(self, target_folders, label_class, lbl_itos):
        self.target_folders = target_folders
        self.label_class = label_class
        self.lbl_itos = lbl_itos

    def modify_dfs(self, dataset_wrapper, df_train, df_valid, df_test):
        if dataset_wrapper.lbl_itos is None:
            # ATTENTION SIDE EFFECT
            dataset_wrapper.lbl_itos = self.lbl_itos
        else:
            # take lbl_itos from dataset wrapper if it exists
            self.lbl_itos = dataset_wrapper.lbl_itos
        dataset_wrapper.num_classes = len(self.lbl_itos)
        numeric_label_column = self.label_class + "_filtered_numeric"
        self.add_numeric_label_column(
            df_train, df_valid, df_test, numeric_label_column, self.label_class)
        ############### map to multihot encoding ###################

        df_train = df_train.assign(label=df_train[numeric_label_column].apply(
            lambda x: multihot_encode(x, len(self.lbl_itos))))
        df_valid = df_valid.assign(label=df_valid[numeric_label_column].apply(
            lambda x: multihot_encode(x, len(self.lbl_itos))))
        df_test = df_test.assign(label=df_test[numeric_label_column].apply(
            lambda x: multihot_encode(x, len(self.lbl_itos))))

        return df_train, df_valid, df_test

    def modify_dfs_of_second_ds(self, dataset_wrapper, df_train, df_valid, df_test):
        ################# adds labels that have the same dimension as the labels in the first dataset - labels are useless here though, but must have the same dimension for #######################
        ################# further processing, in dnl_training labels are picked randomly, in stability_training we use the same label set, in domain adaptation labels are not used ################
        label_class = 'label'
        numeric_label_column = label_class + "_filtered_numeric"

        self.add_numeric_label_column(
            df_train, df_valid, df_test, numeric_label_column, label_class)
        ############### map to multihot encoding ###################

        # fill second dataset with dummy labels
        df_train = df_train.assign(label=df_train[numeric_label_column].apply(
            lambda x: np.zeros(len(self.lbl_itos), dtype=np.float32)))
        df_valid = df_valid.assign(label=df_valid[numeric_label_column].apply(
            lambda x: np.zeros(len(self.lbl_itos), dtype=np.float32)))
        df_test = df_test.assign(label=df_test[numeric_label_column].apply(
            lambda x: np.zeros(len(self.lbl_itos), dtype=np.float32)))

        return df_train, df_valid, df_test

    def add_numeric_label_column(self, df_train, df_valid, df_test, numeric_label_column, label_class):
        if numeric_label_column in df_train.columns:
            return
        elif type(df_train[label_class].iloc[0][0]) == int:
            df_train.loc[:, numeric_label_column] = df_train.loc[:,
                                                                 label_class]
            df_valid.loc[:, numeric_label_column] = df_valid.loc[:,
                                                                 label_class]
            df_test.loc[:, numeric_label_column] = df_test.loc[:,
                                                               label_class]
            return

        def to_numeric(label):
            return np.where(self.lbl_itos == label)[0][0]

        df_train.loc[:, numeric_label_column] = df_train.loc[:, label_class].apply(
            lambda x: [to_numeric(label) for label in x])
        df_valid.loc[:, numeric_label_column] = df_valid.loc[:, label_class].apply(
            lambda x: [to_numeric(label) for label in x])
        df_test.loc[:, numeric_label_column] = df_test.loc[:, label_class].apply(
            lambda x: [to_numeric(label) for label in x])

    def __str__(self):
        return "ToOneHot"


class StatementsToLabels(DataModifier):
    def __init__(self, data_dir, lbl_itos, label_class, label_threshold=None):
        self.data_dir = data_dir
        self.lbl_itos = lbl_itos
        self.label_class = label_class
        self.label_threshold = 100 if label_threshold is None else label_threshold
        self.remove_noshows=True

    def start(self):
        label_class_to_key = {'label_all_12sl': 'statements', 'label_all_12sl_ext': 'statements_ext',
                              'label_all_12sl_ext_snomed': 'statements_ext_snomed',
                              'label_all_12sl_ext_snomed_union': 'statements_ext_snomed',
                              'label_all_ptb_ext_snomed': 'scp_codes_ext_snomed',
                              'label_all_ptb_ext_snomed_union': 'scp_codes_ext_snomed'}
        
        df_12sl = pd.read_csv(
            join(dirname(self.data_dir), "statements/12sl_statements.csv"))  # load 12sl df
        df_ptbxl = pd.read_csv(
            join(dirname(self.data_dir), "statements/ptbxl_statements.csv"))  # load 12sl df
        
        key = label_class_to_key[self.label_class]
        if '12sl' in self.label_class:
            key2 = label_class_to_key[self.label_class.replace("12sl", "ptb")]
            df_12sl[key] = df_12sl[key].apply(lambda x: eval(x))
            df_ptbxl[key2] = df_ptbxl[key2].apply(lambda x: eval(x)) 
            df_labels = df_12sl.copy()
            df_labels2 = df_ptbxl.copy()
        else:
            key2 = label_class_to_key[self.label_class.replace("ptb", "12sl")]
            df_12sl[key2] = df_12sl[key2].apply(lambda x: eval(x)) 
            df_ptbxl[key] = df_ptbxl[key].apply(lambda x: eval(x)) 
            df_labels = df_ptbxl.copy()
            df_labels2 = df_12sl.copy()
        
        
        if 'ext' in key:
            df_labels[key] = df_labels[key].apply(
                lambda x: [i[0] for i in x if int(i[1]) >= self.label_threshold])
            # makes sure that list contains strings, not tuples
            df_labels[key] = df_labels[key].apply(lambda x: [str(i) for i in x])

            df_labels2[key2] = df_labels2[key2].apply(
                lambda x: [i[0] for i in x if int(i[1]) >= self.label_threshold])
            # makes sure that list contains strings, not tuples
            df_labels2[key2] = df_labels2[key2].apply(lambda x: [str(i) for i in x])

        return df_labels, df_labels2, key, key2

    def get_intersected_ids(self, df_labels, df_train, df_valid, df_test):
        ecg_ids = set(sorted(df_labels.ecg_id))  # get ecg ids
        train_ids = sorted(ecg_ids.intersection(set(df_train.index)))
        valid_ids = sorted(ecg_ids.intersection(set(df_valid.index)))
        test_ids = sorted(ecg_ids.intersection(set(df_test.index)))
        return train_ids, valid_ids, test_ids

    def remove_nans_and_empty_labels(self, df_labels, key):
        # drop rows that have empty entries or labels that are not valid
        num_samples_old = len(df_labels)
        df_labels = df_labels.drop(df_labels[df_labels[key].apply(
            lambda x: len(x) == 0 or not (set(x) <= set(self.lbl_itos)))].index)
        df_labels[key] = df_labels[key].dropna()
        df_labels = df_labels.sort_values('ecg_id')
        logger.info("reduce num samples from {} -> {} because of label set".format(num_samples_old, len(df_labels)))
        return df_labels

    def check_if_all_labels_are_present(self, list_of_labels, labelset):
        flatten_labels = [x for xi in list_of_labels for x in xi]
        flatten_labels_set = set(flatten_labels)
        labelsetset = set(labelset)
        return str(flatten_labels_set == labelsetset)

    def update_label_set(self, df_labels, key, train_ids, valid_ids, test_ids, label_set, ignore_train=False):
        def get_discardable(ids):
            flatten_statements_valid = [x for xi in df_labels[df_labels['ecg_id'].apply(
                lambda x: x in ids)][key] for x in xi]  # get flatten statements
            statements_unique_valid = set(flatten_statements_valid)
            discardable_statements = list(set(label_set) - statements_unique_valid)
            return discardable_statements
        discardable_statements = get_discardable(valid_ids)+get_discardable(test_ids)
        if not ignore_train:
            discardable_statements += get_discardable(train_ids) 
        new_label_set = np.array(
            sorted([x for x in label_set if x not in discardable_statements]))
        return new_label_set

    def filter_df(self, df_labels, reduced_labelset, key):
        df_labels[key] = df_labels[key].apply(
            lambda x: [label for label in x if label in reduced_labelset])
        return df_labels

    def assign_labels(self, df_labels, df_data, ids, key):
        df_data = df_data.loc[ids]
        data_labels = df_labels[df_labels['ecg_id'].apply(
            lambda x: x in ids)][key].values
        df_data[self.label_class] = data_labels
        return df_data

    def modify_dfs(self, dataset_wrapper, df_train, df_valid, df_test):
        df_labels, df_labels2, key, key2 = self.start()

        # df_12sl = self.remove_nans_and_empty_labels(df_12sl, key)

        train_ids, valid_ids, test_ids = self.get_intersected_ids(
            df_labels, df_train, df_valid, df_test)

        # in case we want to compare ptb to 12sl snomeds by evaluating on the union
        # we might want to refrain from discarding labels as the labelset must be 
        # the same for both labelsets due to comparability (labelset is the same but labelling 
        # per sample might be different)
        if self.remove_noshows:
            # update lbl_itos by discarding statements that are in lbl_itos but do not occur in valid or test set fold
            if 'union' in self.label_class:
                new_lbl_itos = self.update_label_set(
                df_labels, key, train_ids, valid_ids, test_ids, self.lbl_itos)
                # # dont check for train_ids in second label set
                new_lbl_itos = self.update_label_set(
                    df_labels2, key2, train_ids, valid_ids, test_ids, new_lbl_itos, ignore_train=False) 
            else:
                new_lbl_itos = self.update_label_set(
                df_labels, key, train_ids, valid_ids, test_ids, self.lbl_itos) 

            logger.info("reduce num labels from {} -> {}, due to absence in either val or test set".format(
                len(self.lbl_itos), len(new_lbl_itos)))
            
            dataset_wrapper.lbl_itos = new_lbl_itos  # ATTENTION; SIDE EFFECT
            
            # remove labels that do not occur in the valid or test fold, because of the label_threshold
        
            df_labels = self.filter_df(df_labels, new_lbl_itos, key)
        else:
            dataset_wrapper.lbl_itos = self.lbl_itos  # ATTENTION; SIDE EFFECT

        # drop rows that have empty entries or labels that are not valid
        df_labels = self.remove_nans_and_empty_labels(df_labels, key)

        train_ids, valid_ids, test_ids = self.get_intersected_ids(
            df_labels, df_train, df_valid, df_test)
        df_train = self.assign_labels(df_labels, df_train, train_ids, key)
        df_valid = self.assign_labels(df_labels, df_valid, valid_ids, key)
        df_test = self.assign_labels(df_labels, df_test, test_ids, key)

        # check label availability
        logger.info("label availability in train: "+ self.check_if_all_labels_are_present(list(df_train[self.label_class]), dataset_wrapper.lbl_itos))
        logger.info("label availability in valid: "+ self.check_if_all_labels_are_present(list(df_valid[self.label_class]), dataset_wrapper.lbl_itos))
        logger.info("label availability in test: " +
                    self.check_if_all_labels_are_present(list(df_test[self.label_class]), dataset_wrapper.lbl_itos))
        
        return df_train, df_valid, df_test

    def modify_dfs_of_second_ds(self, dataset_wrapper, df_train, df_valid, df_test):
        return df_train, df_valid, df_test

    def __str__(self):
        return "StatementsToLabels"

    def multihot_encode(x, num_classes):
        res = np.zeros(num_classes)
        res[x] = 1
        return res


def filter_multihot_encode(x, filter_labels, numeric_mapping):
    res = res = np.zeros(len(filter_labels), dtype=np.float32)
    for label in x:
        if label in filter_labels:
            res[numeric_mapping[label]] = 1
    return res


def multihot_encode(x, num_classes):
    res = np.zeros(num_classes, dtype=np.float32)
    res[x] = 1
    return res


def get_label_distances(labels):
    def find_leave(label):
        return sorted(findall(root, filter_=lambda node: node.name == label), key=lambda x: x.depth, reverse=True)[0]

    def distance_between_two_nodes(node1, node2):
        depth1 = node1.depth
        depth2 = node2.depth
        lca = sorted(list(set(node1.path).intersection(node2.path)),
                     key=lambda x: x.depth, reverse=True)[0]
        lca_depth = lca.depth
        return depth1 + depth2 - 2*lca_depth

    nodes = [find_leave(label) for label in labels]
    distances = np.zeros(((len(nodes), len(nodes))))
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            distances[i, j] = distance_between_two_nodes(node1, node2)
    return distances


def get_data_modifiers(args, label_class, lbl_itos):
    data_modifiers = []

    if '12sl' in label_class or 'ext' in label_class:
        data_modifiers.append(
            StatementsToLabels(args.target_folders[0], lbl_itos, label_class, label_threshold=args.label_threshold))
    # One hot encoding in case of normal training
    data_modifiers.append(
        ToOneHot(args.target_folders, label_class, lbl_itos))
    logger.info('Data modifiers: ' + str([str(d) for d in data_modifiers]))
    return data_modifiers
