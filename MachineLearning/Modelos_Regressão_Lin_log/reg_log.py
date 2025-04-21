import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
import pandas as pd

class LogisticRegression:
    def __init__(self, lr=0.01, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None


    def _sigmoid(self, y):
        return 1 / (1 + np.exp(-y))

    def fit(self, x, y):

        self.w = np.zeros((1, 1))
        self.b = 0
        m = len(y)

        # Gradiente descendente
        for epoch in range(self.epochs):
            z = np.dot(x, self.w) + self.b
            y_hat = self._sigmoid(z)
            cost = -np.mean(y * np.log(y_hat + 1e-15) + (1 - y) * np.log(1 - y_hat + 1e-15))
            dw = np.dot(x.T, (y_hat - y)) / m
            db = np.sum(y_hat - y) / m
            self.w -= self.lr * dw
            self.b -= self.lr * db

            if epoch % (self.epochs // 5 if self.epochs >= 5 else 1) == 0:
                print(f"Epoch {epoch}: Custo = {cost:.4f}")

        z_final = np.dot(x, self.w) + self.b
        y_prob = self._sigmoid(z_final)
        y_pred = (y_prob >= 0.5).astype(int)
        acc = np.mean(y_pred == y)
        print(f"Acurácia: {acc * 100:.2f}%")

        return y_pred

    def predict_proba(self, x_new):
        z = np.dot(x_new, self.w) + self.b
        return self._sigmoid(z)

    def predict(self, x_new, threshold=0.5):
        y_prob = self.predict_proba(x_new)
        return (y_prob >= threshold).astype(int)
