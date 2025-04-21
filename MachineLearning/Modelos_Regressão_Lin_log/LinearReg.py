import numpy as np
import scipy as sp


class LinearReg:
    def __init__(self):
        self.X = None
        self.y = None
        self.alpha = None
        
    def fit(self, X, y):
        self.X = X
        self.y = y
        n = self.X.shape[1]
        X_m = np.zeros(n)
        for j in range(n):
            X_m[j] = np.mean(self.X[:,j])
        for j in range(n):
            self.X[:,j] = self.X[:, j] - X_m[j]
        y_m = np.mean(self.y)
        self.y = self.y - y_m
        self.alpha = sp.linalg.inv(self.X.T@self.X)@self.X.T@self.y
        self.alpha_0 = y_m - X_m@self.alpha

    def predict_real(self, Xp):

        return Xp@self.alpha + self.alpha_0
    
    def predict(self, Xp):
        
        return np.sign(self.predict_real(Xp))