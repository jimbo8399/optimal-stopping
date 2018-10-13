# -*- coding: utf-8 -*-
from sklearn import linear_model
from sklearn.model_selection import cross_validate
import numpy as np
from bin.data_import import import_dataset_2 as im

data = im()['temperature', 'humidity']


def get_linear_regression_model(df):
    reg = linear_model.LinearRegression()
    X = df.temperature.values.reshape(-1, 1)
    y = df.humidity.values.reshape(-1, 1)
    mean_score = np.mean(cross_validate(reg, X, y, cv=6)["test_score"])
    print("Acc: {}".format(mean_score))
    return reg.fit(X, y)


# if __name__ == "__main__":
#     get_linear_regression_model(data)
