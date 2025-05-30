import numpy as np
import pandas as pd


class NaiveBayes:
    def __init__(self, classes):
        self.classes = classes
        self.X_train = None
        self.y_train = None

    def treino(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def pred(self, X_test):
        def likelihood(y, Z):
            def gaussian(x, mu, sig):
                return np.exp(-np.power(x - mu, 2.)/(2*np.power(sig, 2.)))
            prob = 1
            for j in np.arange(0, Z.shape[1]):
                m = np.mean(Z[:,j])
                s = np.std(Z[:,j])      
                prob = prob*gaussian(y[j], m, s)
            return prob
        
        valores = pd.DataFrame(data=np.zeros((X_test.shape[0], len(self.classes))), columns=self.classes) 
        for i in np.arange(0, len(self.classes)):
            elements = tuple(np.where(self.y_train == self.classes[i]))
            Z = self.X_train[elements,:][0]
            for j in np.arange(0,X_test.shape[0]):
                x = X_test[j,:]
                pj = likelihood(x, Z)
                valores[self.classes[i]][j] = pj*len(elements)/self.X_train.shape[0]
        return valores