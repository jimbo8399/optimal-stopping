# -*- coding: utf-8 -*-
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np
import sys

sys.path.append("/home/katya/optimal-stopping/")

from bin.data_import import import_dataset_2 as im


def get_linear_regression_model(temperature, humidity):
    reg = linear_model.LinearRegression()
    return reg.fit(temperature,humidity)

def evaluate_lin_reg_model(model, X, y):
    sq_errors = []
    for point in X:
        sq_errors = [(model.predict(X)-y)**2]
    mean_sq_error = np.mean(sq_errors)
    return mean_sq_error

def k_fold_cv(X, y, K=6):
    scores = []
    for k in range(K):
        X_train, _, y_train, _ = \
            train_test_split(X, y, test_size=0.1, random_state=None)
        model = get_linear_regression_model(X_train, y_train)
        scores += [evaluate_lin_reg_model(model, X, y)]
    return np.mean(scores)

if __name__ == "__main__":
    print("Model accuracy")
    for suv in im():
        _id = suv.device.values[0]
        data = suv
        X = data.temperature.values.reshape(-1,1)
        y = data.humidity.values.reshape(-1,1)
        print("Sensor " + str(_id) + " : " + str(k_fold_cv(X,y)))