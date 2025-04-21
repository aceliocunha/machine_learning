import numpy as np



def normaliza(X_train, X_test):
    from sklearn.preprocessing import StandardScaler
    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)

    return X_train_std, X_test_std


def get_xor():
    X = np.zeros((200, 2))
    X[:50] = np.random.random((50, 2)) / 2 + 0.5  # (0.5-1, 0.5-1)
    X[50:100] = np.random.random((50, 2)) / 2  # (0-0.5, 0-0.5)
    X[100:150] = np.random.random((50, 2)) / 2 + np.array([[0, 0.5]])  # (0-0.5, 0.5-1)
    X[150:] = np.random.random((50, 2)) / 2 + np.array([[0.5, 0]])  # (0.5-1, 0-0.5)
    y = np.array([0] * 100 + [1] * 100)
    return X, y


def get_random_noral():
    X = np.vstack((np.random.normal(loc=1, scale=0.5, size=(100, 2)),
                   np.random.normal(loc=-1, scale=0.5, size=(100, 2))))
    y = np.concatenate((np.ones(100), -np.ones(100)))
    return X, y


def get_donut():
    N = 200
    R_inner = 5
    R_outer = 10

    # distance from origin is radius + random normal
    # angle theta is uniformly distributed between (0, 2pi)
    R1 = np.random.randn(N // 2) + R_inner
    theta = 2 * np.pi * np.random.random(N // 2)
    X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

    R2 = np.random.randn(N // 2) + R_outer
    theta = 2 * np.pi * np.random.random(N // 2)
    X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

    X = np.concatenate([X_inner, X_outer])
    y = np.array([0] * (N // 2) + [1] * (N // 2))
    return X, y


def makeregression():
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=100, n_features=1, noise=10.0)
    # X_binary = (X > X.mean()).astype(int)
    return X, y


def diabetes():
    from sklearn.datasets import make_regression, load_diabetes
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target
    return X, y


def iris():
    from sklearn.datasets import load_iris

    iris = load_iris()
    X = iris.data[:, 2:]
    y = (iris.target == 2).astype(np.int64)
    return X, y




