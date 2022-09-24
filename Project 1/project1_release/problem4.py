# -------------------------------------------------------------------------
'''
    Problem 4: compute sigmoid(Z), the loss function, and the gradient.
    This is the vectorized version that handle multiple training examples X.

    20/100 points
'''

import numpy as np # linear algebra
from scipy.sparse import diags
from scipy.sparse import csr_matrix

def linear(theta, X):
    '''
    theta: (n+1) x 1 column vector of model parameters
    x: (n+1) x m matrix of m training examples, each with (n+1) features.
    :return: inner product between theta and x
    '''
    #########################################
    return np.dot(theta.T, X)
    #########################################

def sigmoid(Z):
    '''
    Z: 1 x m vector. <theta, X>
    :return: A = sigmoid(Z)
    '''
    #########################################
    return 1/(1+np.exp(-Z))
    #########################################

def loss(A, Y):
    '''
    A: 1 x m, sigmoid output on m training examples
    Y: 1 x m, labels of the m training examples

    You must use the sigmoid function you defined in *this* file.

    :return: mean negative log-likelihood loss on m training examples.
    '''
    #########################################
    # l = -(np.sum(np.isfinite((Y*np.log(A)+(1-Y)*(np.log(1-A))))))/len(A)
    # l = -np.mean(np.isfinite((Y*np.log(A)+(1-Y)*(np.log(1-A)))))
    # l = -(np.sum(np.isfinite(np.dot(Y, np.log(A)) + np.dot(1-Y,np.log(1-A)))))/(len(A))
    # l = -(np.sum(np.dot(Y, np.log1p(A)) + np.dot(1-Y,np.log1p(1-A))))/(len(A))
    # l = -np.mean((Y*np.log1p(A)+(1-Y)*(np.log1p(1-A))))
    # inside = Y*np.log(A)+(1-Y)*(np.log(1-A))
    
    
    
    # l = -np.mean((np.dot(Y,np.log(A))+np.dot(1-Y,np.where(1-A>0, np.log(1-A),0))))
    # l = - np.sum(Y*np.log(A)+ ((1-Y)*np.log(1-A)))/(len(A))
    l = -np.mean((Y*np.log(A) + (1-Y)*np.log(1-A)))
    return l 
    #########################################

def dZ(Z, Y):
    '''
    Z: 1 x m vector. <theta, X>
    Y: 1 x m, label of X

    You must use the sigmoid function you defined in *this* file.

    :return: 1 x m, the gradient of the negative log-likelihood loss on all samples wrt z.
    '''
    #########################################
    return -(Y*(-sigmoid(Z)+1)+(1-Y)*(-sigmoid(Z)))
    #########################################

def dtheta(Z, X, Y):
    '''
    Z: 1 x m vector. <theta, X>
    X: (n+1) x m, m example feature vectors.
    Y: 1 x m, label of X
    :return: n x 1, the gradient of the negative log-likelihood loss on all samples wrt w.
    '''
    #########################################
    return np.atleast_2d(np.mean((sigmoid(Z)-Y)*X, axis=1)).T
    #########################################

def Hessian(Z, X):
    '''
    Compute the Hessian matrix on m training examples.
    Z: 1 x m vector. <theta, X>
    X: (n+1) x m, m example feature vectors.
    :return: the Hessian matrix of the negative log-likelihood loss wrt theta
    '''
    #########################################
    sigma_m = np.diagflat(np.diag(sigmoid(Z)*sigmoid(-Z).T))



    return np.dot(np.dot(X,sigma_m),X.T)
    #########################################
