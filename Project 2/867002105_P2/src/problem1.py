'''
    Problem 1: Implement linear and Gaussian kernels and hinge loss
'''

import numpy as np
from sklearn.metrics.pairwise  import euclidean_distances


def linear_kernel(X1, X2):
    
    """
    Compute linear kernel between two set of feature vectors.
    The constant 1 is not appended to the x's.

    X1: n x m1 matrix, each of the m1 column is an n-dim feature vector.
    X2: n x m2 matrix, each of the m2 column is an n-dim feature vector.
    
    Note that m1 may not equal m2

    :return: if both m1 and m2 are 1, return linear kernel on the two vectors; else return a m1 x m2 kernel matrix K,
            where K(i,j)=linear kernel evaluated on column i from X1 and column j from X2.
    """
    #########################################
    '''
    shape_1 = X1.shape
    shape_2 = X2.shape
 
    if shape_1[0] == 1 and shape_2[0] == 1:
        return np.dot(X1.T, X2)

        #Kernel = np.trace(K)

    else:
        K = np.zeros((shape_1[0], shape_2[1]))
        for i in range(shape_1[0]):
            for j in range(shape_2[1]):
                    K[i,j] = np.dot(X1[:,i], X2[:,j])
        return K
    '''
    #if X1.shape[1] == 1 and X2.shape[1] == 1:
    #    K = np.dot(X1.T, X2)
    #elif (X1.shape == X2.shape) and (X1.shape[0] == X1.shape[1]) and (X2.shape[0] == X2.shape[1]):
    #    K = np.inner(X1, X2) # np.dot(X1.T, X2) != (X1^T)(X2) when matrices are square and equal to each other. I may have found a bug in numpy.
    #elif :
    #    K = np.dot(X1.T, X2)


    K = np.dot(X1.T, X2)
    return K
    #return np.inner(X1.T, X2.T)
    #########################################



def Gaussian_kernel(X1, X2, sigma=1):
    """
    Compute Gaussian kernel between two set of feature vectors.
    
    The constant 1 is not appended to the x's.
    
    For your convenience, please use euclidean_distances.

    X1: n x m1 matrix, each of the m1 column is an n-dim feature vector.
    X2: n x m2 matrix, each of the m2 column is an n-dim feature vector.
    sigma: Gaussian variance (called bandwidth)

    Note that m1 may not equal m2

    :return: if both m1 and m2 are 1, return Gaussian kernel on the two vectors; else return a m1 x m2 kernel matrix K,
            where K(i,j)=Gaussian kernel evaluated on column i from X1 and column j from X2

    """
    #########################################
    if X1.shape[1] == 1 and X2.shape[1] == 1:
        return np.exp(-(euclidean_distances(X1.T,X2.T)**2)/(2*sigma**2))

        #Kernel = np.trace(K)

    elif X1.shape[1] !=  X2.shape[1]: 
        return np.exp(-(euclidean_distances(X1,X2.T)**2)/(2*(sigma**2)))

    elif X1.shape[1] ==  X2.shape[1]: 
        K = np.zeros((X1.shape[1], X2.shape[1]))
        for i in range(X2.shape[1]):
            for j in range(X2.shape[1]):
                    K[i,j] = np.exp(-(euclidean_distances(np.array([X2[:,i]]),np.array([X2[:,j]]))**2)/(2*sigma**2))
        return K
    #########################################


def hinge_loss(z, y):
    """
    Compute the hinge loss on a set of training examples
    z: 1 x m vector, each entry is <w, x> + b (may be calculated using a kernel function)
    y: 1 x m label vector. Each entry is -1 or 1
    :return: 1 x m hinge losses over the m examples
    """
    #########################################
 
    K = np.zeros((z.shape))
    for i in range(z.shape[0]):
        K[i] = max(0, 1 - np.dot(y[:,i],z[i]))
    return K
    #########################################
