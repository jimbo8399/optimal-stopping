# -*- coding: utf-8 -*-
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np

def get_svr_rbf_model(temp_hum,sensor):
    svr_rbf = SVR(kernel='poly', gamma='scale')
    X = temp_hum
    r_y = sensor.reshape(sensor.shape[0],)
    return svr_rbf.fit(X,r_y)

def evaluate_svr_rbf_model(model, X, y):
    seq_errors = []
    for point in X:
        seq_errors = [(model.predict(X)-y)**2]
    mean_sq_error = np.mean(seq_errors)
    return mean_sq_error

def k_fold_cv(model, X, y, K=6):
    r_y = y.reshape(y.shape[0],)
    scores = []
    for k in range(K):
        _, X_test, _, y_test = \
            train_test_split(X, y, test_size=0.1, random_state=None)
        scores += [evaluate_svr_rbf_model(model, X_test, y_test)]
    return np.mean(scores)