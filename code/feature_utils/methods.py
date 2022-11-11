from random import random
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np 
import pdb
from os.path import join 
import time 
import pickle 
from os import makedirs
import json
import pandas as pd

   
def cal_metrics(y_true, y_pred):
    metrics = {}
    metrics['accuracy'] = 1-np.mean(abs(y_pred - y_true))
    metrics['f1_score_macro'] = sklearn.metrics.f1_score(y_true, y_pred, average='macro')
    metrics['f1_score_weighted'] = sklearn.metrics.f1_score(y_true, y_pred, average='weighted')
    metrics['precision_macro'] = sklearn.metrics.average_precision_score(y_true, y_pred, average='macro')
    metrics['precision_weighted'] = sklearn.metrics.average_precision_score(y_true, y_pred, average='weighted')
    metrics['recall_macro'] = sklearn.metrics.recall_score(y_true, y_pred, average='macro')
    return metrics

def get_model(modelname, model_path, rf_max_depth):
    if model_path is not None:
        print("load model from", model_path)
        model = pickle.load(open(model_path, 'rb'))
    elif modelname == 'rf':
        print("create random forest..")
        model = RandomForestClassifier(max_depth=rf_max_depth)
    elif modelname == "ridge":
        print("create ridge classifier..")
        model = RidgeClassifier()
    elif modelname == "nn":
        print("fit neural network..")
        model = MLPClassifier(random_state=1, max_iter=300, hidden_layer_sizes=(256, 256, 256))
    else:
        raise Exception("'{}' modelname is unknown".format(modelname))

    return model

def evaluate_model(model, X_test, y_test, label_class, modelname, dataset,save=True, common_with=None, modelparams={}):
    y_pred = model.predict(X_test)
    labels = pd.read_pickle("./data/ptb_xl_fs100/lbl_itos.pkl")[label_class]
    
    if hasattr(model, 'predict_proba'):
        y_pred = model.predict_proba(X_test)
        if type(y_pred) == list:
            aucs={}
            y_score = np.zeros((len(y_test), len(labels)))
            for i in range(len(labels)):
                y_score[:, i] = y_pred[i][:, 1]
            auc = sklearn.metrics.roc_auc_score(y_test, y_score, labels=labels)
            
        else:
            auc = sklearn.metrics.roc_auc_score(y_test, y_pred, labels=labels)
        return auc
    return None