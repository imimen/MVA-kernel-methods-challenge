import numpy as np


def kernel_linear(x1, x2):
    return np.dot(x1, x2.T)


def kernel_polynomial(x1, x2, d=2):
    return np.dot(x1, x2.T) ** d


def kernel_gaussian(x1, x2, t=1):
    d = x1.shape[0]
    return np.exp(-np.dot(x1 - x2, (x1 - x2).T) / (4 * t)) / (
        (4 * np.pi * t) ** (d / 2)
    )


kernels = {
    "linear": kernel_linear,
    "polynomial": kernel_polynomial,
    "gaussian": kernel_gaussian,
}

def compute_kernel_matrix(kernel, X1, X2):
    n1, n2 = X1.shape[0], X2.shape[0]
    K = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            K[i, j] = kernel(X1[i], X2[j])
    return K
    
def threshold(ytest):
    mask = ytest > 0.5
    ytest[mask] = 1
    ytest[~mask] = 0
    ytest = ytest.astype(dtype=np.uint8)
    return ytest


class kernelRidge:
    def __init__(self, kernel_type="linear", c=0.0001):
        self.kernel_type = kernel_type
        self.kernel = kernels[self.kernel_type]
        self.c = c

    def fit(self, X, y):
        self.K = compute_kernel_matrix(self.kernel, X, X)
        self.weights = np.dot(
            np.linalg.inv(self.K + self.c * np.eye(self.K.shape[0])), y
        )
        return self.weights

    def predict(self, X, Xtest):
        k = compute_kernel_matrix(self.kernel, X, Xtest)
        ytest = np.dot(self.weights, k)
        return ytest.transpose()
    
    
class KernelLogistic():
    def __init__(self):
        super().__init__()

    def fit(self, x, y):
        return 1
    
    def predict(self, x):
        pred = 0
        return pred
    
class KernelSvm():
    def __init__(self):
        super().__init__()

    def fit(self, x, y):
        return 1
    
    def predict(self, x):
        pred = 0
        return pred
