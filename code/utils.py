import pandas as pd
import numpy as np 
from os.path import join 
import pdb

feature_col_map = {'glasgow':'Glasgow_feature', 'kit':'KIT_feature', 'ge':'GE_feature'}



def replace_nans(df):
    hasnancols = list(df.columns[df.isnull().any()])
    # replace NaNs with median of respective column
    for col in hasnancols:
        df[col]=df[col].fillna((df[col].median()))
    return df

def load_dataset(datasetname):
    print("load data..")
    if datasetname=='glasgow':
        df = pd.read_csv(join("data", 'glasgow_features_final.csv'))
    elif datasetname=='kit':
        df = pd.read_csv(join("data", 'kit_features_final.csv'))
    elif datasetname == 'ge':
        df = pd.read_csv(join("data", '12sl_features_final.csv'))
    else:
        raise Exception("specified dataset: {} is unknown".format(datasetname))    

    return df

def to_one_hot(arr, num_classes):
    res = np.zeros((len(arr), num_classes))
    for i, elem in enumerate(arr):
        res[i][elem] = 1
    return res

def select_features(datasetname, data, common_with):
    if common_with is None:
        return data
    # data = data[features]
    common_features = get_common_features(data, datasetname, common_with)
    expanded_features = expand_features(common_features)
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

def get_common_features(data, datasetname, common_with):
    print("select {} features that are also {}".format(datasetname, common_with))
    fm = pd.read_excel("./data/ge12sl_glasgow_kit_ECGFeaturesMapToOMOP_draft1.xlsx")
    feat_col1 = feature_col_map[datasetname]
    feat_col2 = feature_col_map[common_with]
    fm = fm[['id', feat_col1, feat_col2]]
    common_features = fm[~fm[feat_col1].isna() & ~fm[feat_col2].isna()]
    final_features = common_features['id'].values
    print('selected features: {}'.format(final_features))
    return final_features

def get_data_from_ids(datasetname, df, labels, ids, num_classes, common_with):
    data = df.loc[df['ecg_id'].isin(ids)].sort_values("ecg_id")
    valid_ids = data['ecg_id'].values
    data.drop(["ecg_id"], axis=1)
    data = select_features(datasetname, data, common_with)
    data = replace_nans(data)
    X = data.values
    y = labels.loc[valid_ids].values 
    y = to_one_hot(y, num_classes)
    return X, y

def get_split(datasetname, label_class, common_with, test_folds=[9, 10]):
    df = load_dataset(datasetname)
    print("get split..")
    # PTB-XL Labels:
    lbl_itos = pd.read_pickle("./data/lbl_itos.pkl")
    lbl_itos = lbl_itos[label_class]
    num_classes = len(lbl_itos)
    df_labels = pd.read_pickle("./data/df.pkl")
    label = label_class + "_filtered_numeric"
    labels = df_labels[label]

    # Get Ids for train and test set
    train_folds = list(range(1, 11))
    for fold in test_folds:
        train_folds.remove(fold)
    
    train_set_ids = df_labels[df_labels['strat_fold'].apply(lambda x: x in train_folds)].index.values
    test_set_ids  = df_labels[df_labels['strat_fold'].apply(lambda x: x in test_folds)].index.values

    # get train and test data based on ids
    X_train, y_train = get_data_from_ids(datasetname, df, labels, train_set_ids, num_classes, common_with)
    X_test, y_test = get_data_from_ids(datasetname, df, labels, test_set_ids, num_classes, common_with)

    return X_train, X_test,  y_train,  y_test