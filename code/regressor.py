from pcp import pcp
import numpy as np

class Regressor(object):

    regressorA = None
    regressorB = None
    baseRegressor = None

    def __init__(self, regressorA, regressorB, baseRegressor):
        self.regressorA = regressorA
        self.regressorB = regressorB
        self.baseRegressor = baseRegressor

    def fit(self, X, y):
        L, S, (u, s, v) = pcp(np.array(y), maxiter=30, verbose=False, svd_method="approximate")
        L = np.ravel(L)
        S = np.ravel(S)
        self.regressorA.fit(X, S)
        self.regressorB.fit(X, L)
        self.baseRegressor.fit(X, y)

    def predict(self, X):
        predict_base = self.regressorB.predict(X)
        predict_anomaly = self.regressorA.predict(X)
        predict_total = [x_i+y_i for x_i, y_i in zip(predict_base, predict_anomaly)]
        predict_dummy = self.baseRegressor.predict(X)
        return predict_base, predict_anomaly, predict_total, predict_dummy