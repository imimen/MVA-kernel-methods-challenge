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
    

class logistic_regression:
    def __init__(self, tol=1-3, maxit=1000):
        self.tol = tol
        self.maxit = maxit
        self.reg = None  # regression coefficients
    
    def L(self, phi, t):
        y = sigmoid(phi@self.reg)
        if np.any(y==0) or (np.any(1-y)==0):
            return None
        return (t.T@np.log(y)+(1-t).T@np.log(1-y))    
        
    def fit(self, x, t, alpha=0.01, beta=0.5):
        phi = np.concatenate((np.ones((x.shape[0],1)), x), axis=1) 
        self.reg= np.zeros(phi.shape[1])   # initialization of the regression weights 
        L = self.L(phi, t)                 # initial loglikelihhod
        it = 0
        while (it<self.maxit):
            y = sigmoid(phi@self.reg)
            R = np.diag(np.diag(y*(1-y)[:,np.newaxis]))   # weighting matrix
            hess = phi.T @ R @ phi                        # hessian matrix
            if (np.linalg.det(hess)==0):
                print("singular hessian matrix encountered")
                break;
            inv_hess = np.linalg.inv(hess)                      # inverse of the hessian matrix
            self.reg = self.reg - inv_hess @ phi.T @ (y-t)      # update regression weights
            L_new = self.L(phi, t)       # stopping criterion
            if (L_new==None):
                break;    
            if (abs(L_new-L)<self.tol):
                break;
            L = L_new 
            it = it+1  
        return self.reg
        
    def predict(self, x):
        X = np.concatenate((np.ones((x.shape[0],1)), x), axis=1)  # we add the intercept
        pred = sigmoid(X@self.reg)
        return np.array([ 0 if v<0.5 else 1 for v in pred])
    
    
    
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
