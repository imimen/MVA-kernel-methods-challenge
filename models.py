import numpy as np
import scipy.sparse as sp
import cvxopt as co


# UTILS
def compute_kernel_matrix(kernel, X1, X2):
    n1, n2 = X1.shape[0], X2.shape[0]
    K = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            K[i, j] = kernel(X1[i], X2[j])
    K = sp.csr_matrix(K)
    return K


def threshold(ytest):
    mask = ytest > 0
    ytest[mask] = 1
    ytest[~mask] = 0
    ytest = ytest.astype(dtype=np.uint8)
    return ytest


# KERNELS
class kernel_linear:
    def __init__(self):
        self.type = "_linear"

    def __call__(self, x1, x2):
        return np.dot(x1, x2.T)


class kernel_polynomial:
    def __init__(self, d=2):
        self.type = "_polynomial_d_" + str(d)
        self.d = d

    def __call__(self, x1, x2):
        return np.dot(x1, x2.T) ** self.d


class kernel_gaussian:
    def __init__(self, sigma=1):
        self.type = "_gaussian_sigma_" + str(sigma)
        self.sigma = sigma

    def __call__(self, x1, x2):
        return np.exp(-np.linalg.norm(x1 - x2) / (2 * self.sigma ** 2))


# BASELINES
class kernelRidge:
    def __init__(self, kernel_type="linear", d=2, sigma=1, c=0.01):
        if kernel_type == "linear":
            self.kernel = kernel_linear()
        elif kernel_type == "gaussian":
            self.kernel = kernel_gaussian(sigma)
        elif kernel_type == "polynomial":
            self.kernel = kernel_polynomial(d)
        self.type = "_ridge_c_" + str(c) + self.kernel.type
        self.c = c

    def fit(self, X, y):
        self.K = compute_kernel_matrix(self.kernel, X, X)
        A = self.K + sp.csr_matrix(self.c * np.eye(self.K.todense().shape[0]))
        self.weights = sp.csr_matrix(X).transpose()
        self.weights = self.weights.dot(sp.linalg.spsolve(A, y))
        return self.weights

    def predict(self, X, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest = threshold(ytest)
        return ytest


class kernelSVM:
    def __init__(self, kernel_type="gaussian", d=2, sigma=1, c=0.01):
        if kernel_type == "linear":
            self.kernel = kernel_linear()
        elif kernel_type == "gaussian":
            self.kernel = kernel_gaussian(sigma)
        elif kernel_type == "polynomial":
            self.kernel = kernel_polynomial(d)
        self.c = c
        self.type = "_svm_c_" + str(c) + self.kernel.type

    def fit(self, X, y):
        # convert labels {0,1} to {-1, 1}
        y[y == 0] = -1

        K = compute_kernel_matrix(self.kernel, X, X).todense()
        n = X.shape[0]
        y = y.astype("float")

        # solve the dual problem
        P = co.matrix(np.diag(y) @ K @ np.diag(y))
        q = co.matrix(np.ones(n) * -1)
        A = co.matrix(y, (1, n))
        b = co.matrix(0.0)
        Id = np.identity(n)
        G = co.matrix(np.vstack((-1 * Id, Id)))
        h1 = np.zeros(n)
        h2 = np.ones(n) * self.c
        h = co.matrix(np.hstack((h1, h2)))
        sol = co.solvers.qp(P, q, G, h, A, b)

        # get primal solution
        self.weights = np.diag(y) @ np.ravel(sol["x"]) / self.c
        return self.weights

    def predict(self, X, Xtest):
        k = compute_kernel_matrix(self.kernel, X, Xtest).todense()
        preds = np.dot(self.weights, k)
        preds = threshold(preds)  # labels are {0, 1}
        ytest = list(np.array(preds).flat)
        return ytest


class KernelLogistic:
    def __init__(self):
        super().__init__()

    def fit(self, x, y):
        return 1

    def predict(self, x):
        pred = 0
        return pred