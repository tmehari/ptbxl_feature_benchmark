import pandas as pd
import numpy as np 
from os.path import join 
import pdb
import re
import time
import pickle
from os import makedirs
import json 
feature_col_map = {'unig':'unig_feature', 'ecgdeli':'ecgdeli_feature', '12sl':'12sl_feature'}

def replace_nans(df):
    hasnancols = list(df.columns[df.isnull().any()])
    # replace NaNs with median of respective column
    for col in hasnancols:
        df[col]=df[col].fillna((df[col].median()))
    return df

def load_dataset(datasetname, data_dir):
    print("load data..")
    if datasetname=='unig':
        df = pd.read_csv(join(data_dir, 'features/unig_features.csv')) # glasgow_features_final.csv
    elif datasetname=='ecgdeli':
        df = pd.read_csv(join(data_dir, 'features/ecgdeli_features.csv')) # kit_features_final.csv
    elif datasetname == '12sl':
        df = pd.read_csv(join(data_dir, 'features/12sl_features.csv')) # '12sl_features_final.csv'
    else:
        raise Exception("specified dataset: {} is unknown".format(datasetname))    

    return df

def save_model(model, results   , dataset='glasgow', name='rf', label_class='label_all', common_with=None):
    def make_pretty(text):
        rx = re.compile('([{}\'])')
        text = rx.sub(r'', text)
        text = text.replace(" ", "")
        text = text.replace(":","_")
        return text

    logdir = join("./output", dataset+("_w_"+common_with if common_with is not None else ''), name+"_"+label_class + "_"+make_pretty(str(results ['params'])), str(time.time()))
    makedirs(logdir)
    pickle.dump(model, open(join(logdir, 'model.sav'), 'wb'))
    with open(join(logdir, 'log.txt'), 'w') as file:
        file.write(json.dumps(results))

def to_one_hot(arr, num_classes):
    res = np.zeros((len(arr), num_classes))
    for i, elem in enumerate(arr):
        res[i][elem] = 1
    return res

def remove_area_features(feats):
    return [feat for feat in feats if 'area' not in feat.lower()]

def select_features(datasetname, data, common_with, data_dir):
    if common_with is None:
        return data
    # data = data[features]
    common_features = get_common_features(data, datasetname, common_with, data_dir)
    # common_features = remove_area_features(common_features)
    expanded_features = expand_features(common_features)
    expanded_features = [x for x in expanded_features if x in data.columns] # should be unnecessary
    return data[expanded_features]

def expand_features(features):
    expanded_features = []
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    for feature in features:
        if feature.endswith("_X"):
            for lead in leads:
                expanded_features.append(feature[:-1]+lead)
        else:
            expanded_features.append(feature)
    return expanded_features

def get_common_features(data, datasetname, common_with, data_dir):
    print("select {} features that are also {}".format(datasetname, common_with))
    fm = pd.read_excel(join(data_dir, "./features/ge12sl_glasgow_kit_ECGFeaturesMapToOMOP_draft1.xlsx"))
    feat_col1 = feature_col_map[datasetname]
    feat_col2 = feature_col_map[common_with]
    fm = fm[['id', feat_col1, feat_col2]]
    common_features = fm[~fm[feat_col1].isna() & ~fm[feat_col2].isna()]
    final_features = common_features['id'].values
    print('selected features: {}'.format(final_features))
    return final_features

def get_data_from_ids(datasetname, df, labels, ids, num_classes, common_with, common_with_ids, data_dir):
    common_ids = np.intersect1d(ids, common_with_ids) if common_with_ids is not None else ids
    data = df.loc[df['ecg_id'].isin(common_ids)].sort_values("ecg_id")
    valid_ids = data['ecg_id'].values
    data.drop(["ecg_id"], axis=1)
    data = select_features(datasetname, data, common_with, data_dir)
    data = replace_nans(data)
    X = data.values
    y = labels.loc[valid_ids].values 
    y = to_one_hot(y, num_classes)
    return X, y, data.columns

def get_split(datasetname, label_class, common_with, data_dir='./data'):
    df = load_dataset(datasetname, data_dir)

    print("get split..")
    # PTB-XL Labels:
    lbl_itos = pd.read_pickle(join(data_dir, "ptb_xl_fs100/lbl_itos.pkl"))
    lbl_itos = lbl_itos[label_class]
    num_classes = len(lbl_itos)
    df_labels = pd.read_pickle(join(data_dir, "ptb_xl_fs100/df.pkl"))
    label = label_class + "_filtered_numeric"
    labels = df_labels[label]

    # Get Ids for train and test set
    train_folds = list(range(1, 11))
    valid_folds = [9]
    test_folds = [10]
    for fold in valid_folds + test_folds:
        train_folds.remove(fold)
    
    train_set_ids = df_labels[df_labels['strat_fold'].apply(lambda x: x in train_folds)].index.values
    valid_set_ids  = df_labels[df_labels['strat_fold'].apply(lambda x: x in valid_folds)].index.values
    test_set_ids  = df_labels[df_labels['strat_fold'].apply(lambda x: x in test_folds)].index.values
    
    if common_with is None:
        common_with_ids = None
    else:
        common_with_df = load_dataset(common_with, data_dir)
        common_with_ids = common_with_df["ecg_id"]

    # get train and test data based on ids
    X_train, y_train, features = get_data_from_ids(datasetname, df, labels, train_set_ids, num_classes, common_with, common_with_ids, data_dir)
    X_valid, y_valid, _ = get_data_from_ids(datasetname, df, labels, valid_set_ids, num_classes, common_with, common_with_ids, data_dir)
    X_test, y_test, _ = get_data_from_ids(datasetname, df, labels, test_set_ids, num_classes, common_with, common_with_ids, data_dir)

    return X_train, X_valid, X_test, y_train, y_valid, y_test, features