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


def save_model(model, metrics, name='rf', label_class='label_all'):
    logdir = join("./output", name+"_"+label_class, str(time.time()))
    makedirs(logdir)
    pickle.dump(model, open(join(logdir, 'model.sav'), 'wb'))
    with open(join(logdir, 'log.txt'), 'w') as file:
        file.write(json.dumps(metrics))
   

def cal_metrics(y_true, y_pred):
    metrics = {}
    metrics['accuracy'] = 1-np.mean(abs(y_pred - y_true))
    metrics['f1_score_macro'] = sklearn.metrics.f1_score(y_true, y_pred, average='macro')
    metrics['f1_score_weighted'] = sklearn.metrics.f1_score(y_true, y_pred, average='weighted')
    metrics['precision_macro'] = sklearn.metrics.average_precision_score(y_true, y_pred, average='macro')
    metrics['precision_weighted'] = sklearn.metrics.average_precision_score(y_true, y_pred, average='weighted')
    metrics['recall_macro'] = sklearn.metrics.recall_score(y_true, y_pred, average='macro')
    return metrics

def get_model(modelname, model_path):
    if model_path is not None:
        print("load model from", model_path)
        model = pickle.load(open(model_path, 'rb'))
    elif modelname == 'rf':
        print("create random forest..")
        model = RandomForestClassifier()
    elif modelname == "ridge":
        print("create ridge classifier..")
        model = RidgeClassifier()
    elif modelname == "nn":
        print("fit random forest..")
        model = MLPClassifier(random_state=1, max_iter=300, hidden_layer_sizes=(256, 256, 256))
    else:
        raise Exception("'{}' modelname is unknown".format(modelname))

    return model

def evaluate_model(model, X_test, y_test, label_class, modelname, save=True):
    y_pred = model.predict(X_test)
    metrics = cal_metrics(y_test, y_pred)
    metrics['modelname'] = modelname
    metrics['label_class'] = label_class
    print(metrics)
    if save:
        save_model(model, metrics, name=modelname, label_class=label_class)
        