# -------------------------------------------------------------------------
'''
    Problem 2: Compute the objective function and decision function of dual SVM.

'''
from problem1 import *

import numpy as np

# -------------------------------------------------------------------------
def dual_objective_function(alpha, train_y, train_X, kernel_function, sigma):
    """
    Compute the dual objective function value.

    alpha: 1 x m learned Lagrangian multipliers (the dual variables).
    train_y: 1 x m labels (-1 or 1) of training data.
    train_X: n x m training feature matrix. n: number of features; m: number training examples.
    kernel_function: a kernel function implemented in problem1 (Python treats functions as objects).
    sigma: need to be provided when Gaussian kernel is used.
    :return: a scalar representing the dual objective function value at alpha
    Hint: refer to the objective function of Eq. (47).
          You can try to call kernel_function.__name__ to figure out which kernel are used.
    """
    #########################################
    if kernel_function.__name__ == 'linear_kernel':
    #    Kernels = linear_kernel(train_X, train_X)
    #    K = np.zeros((Kernels.shape[0], Kernels.shape[1]))
    #    for i in range(Kernels.shape[0]):
    #            for j in range(Kernels.shape[1]):
    #                K[i,j] = alpha[0][i]*alpha[0][j]*train_y[0][i]*train_y[0][j]*Kernels[i][j]
    #    
    #    return (np.sum(alpha) - 0.5*np.sum(K))
        K = linear_kernel(train_X, train_X)
        front = np.dot(np.dot(np.multiply(alpha, train_y), K), np.multiply(alpha, train_y).T)
        Lin_dual = np.sum(alpha) - 0.5*np.sum(front)
        return Lin_dual
    elif kernel_function.__name__ == 'Gaussian_kernel':
    #    Kernels = Gaussian_kernel(train_X, train_X)
    #    K = np.zeros((Kernels.shape[0], Kernels.shape[0]))
    #    for i in range(Kernels.shape[0]):
    #            for j in range(Kernels.shape[1]):
    #                K[i,j] = alpha[0][i]*alpha[0][j]*train_y[0][i]*train_y[0][j]*Kernels[i][j]
    #    
    #    return (np.sum(alpha) - 0.5*np.sum(K))
    #########################################
        K = Gaussian_kernel(train_X, train_X)
        front = np.dot(np.dot(np.multiply(alpha, train_y), K), np.multiply(alpha, train_y).T)
        Gauss_dual = np.sum(alpha) - 0.5*np.sum(front)
        return Gauss_dual

# -------------------------------------------------------------------------
def primal_objective_function(alpha, train_y, train_X, b, C, kernel_function, sigma):
    """
    Compute the primal objective function value.
    When with linear kernel:
        The primal parameter w is recovered from the dual variable alpha.
    When with Gaussian kernel:
        Can't recover the primal parameter and kernel trick needs to be used to compute the primal objective function.

    alpha: 1 x m learned Lagrangian multipliers (the dual variables).
    train_y: 1 x m labels (-1 or 1) of training data.
    train_X: n x m training feature matrix.
    b: bias term
    C: regularization parameter of soft-SVM
    kernel_function: a kernel function implemented in problem1 (Python treats functions as objects).
    sigma: need to be provided when Gaussian kernel is used.

    :return: a scalar representing the primal objective function value at alpha
    Hint: you need to use kernel trick when come to Gaussian kernel. Refer to the derivation of the dual objective function Eq. (47) to check how to find
            1/2 ||w||^2 and the decision_function with kernel trick.
    """
    #########################################
    if kernel_function.__name__ == 'linear_kernel':
        K = linear_kernel(train_X, train_X)

        # ||w||^2 
        front = np.dot(np.dot(np.multiply(alpha, train_y), K), np.multiply(alpha, train_y).T) 
        # y(wTx+b)
        z = sum(np.dot(np.multiply(alpha,train_y),K))+b
        # loss
        loss = hinge_loss(z, train_y)

        Lin = 0.5*front + C*np.sum(loss)

        return Lin

    elif kernel_function.__name__ == 'Gaussian_kernel':
        K = Gaussian_kernel(train_X, train_X, sigma)

        # ||w||^2 
        front = np.dot(np.dot(np.multiply(alpha, train_y), K), np.multiply(alpha, train_y).T)  
        # y(wTx+b)
        z = sum(np.dot(np.multiply(alpha,train_y),K))+b
        # loss
        loss = hinge_loss(z, train_y)

        Gauss = 0.5*front + C*np.sum(loss)

        return Gauss
    #########################################


def decision_function(alpha, train_y, train_X, b, kernel_function, sigma, test_X):
    """
    Compute the linear function <w, x> + b on examples in test_X, using the current SVM.

    alpha: 1 x m learned Lagrangian multipliers (the dual variables).
    train_y: 1 x m labels (-1 or 1) of training data.
    train_X: n x m training feature matrix.
    test_X: n x m2 test feature matrix.
    b: scalar, the bias term in SVM <w, x> + b.
    kernel_function: a kernel function implemented in problem1 (Python treats functions as objects).
    sigma: need to be provided when Gaussian kernel is used.

    :return: 1 x m2 vector <w, x> + b
    """
    #########################################
    if kernel_function.__name__ == 'linear_kernel':
        return np.dot(np.multiply(alpha,train_y), linear_kernel(test_X, train_X).T)+b
    elif kernel_function.__name__ == 'Gaussian_kernel':
        return np.dot(np.multiply(alpha,train_y), Gaussian_kernel(test_X.T, train_X, sigma).T)+b
    #########################################
