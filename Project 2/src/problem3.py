# -------------------------------------------------------------------------
'''
    Problem 3: SMO training algorithm

'''
from problem1 import *
from problem2 import *

import numpy as np

import copy

class SVMModel():
    """
    The class containing information about the SVM model, including parameters, data, and hyperparameters.

    DONT CHANGE THIS DEFINITION!
    """
    def __init__(self, train_X, train_y, C, kernel_function, sigma=1):
        """
            train_X: n x m training feature matrix. n: number of features; m: number training examples.
            train_y: 1 x m labels (-1 or 1) of training data.
            C: a positive scalar
            kernel_function: a kernel function implemented in problem1 (Python treats functions as objects).
            sigma: need to be provided when Gaussian kernel is used.
        """
        # data
        self.train_X = train_X
        self.train_y = train_y
        self.n, self.m = train_X.shape

        # hyper-parameters
        self.C = C
        self.kernel_func = kernel_function
        self.sigma = sigma

        # parameters
        self.alpha = np.zeros((1, self.m))
        self.b = 0

def train(model, max_iters = 10, record_every = 1, max_passes = 1, tol=1e-6):
    """
    SMO training of SVM
    model: an SVMModel
    max_iters: how many iterations of optimization
    record_every: record intermediate dual and primal objective values and models every record_every iterations
    max_passes: each iteration can have maximally max_passes without change any alpha, used in the SMO alpha selection.
    tol: numerical tolerance (exact equality of two floating numbers may be impossible).
    :return: 4 lists (of iteration numbers, dual objectives, primal objectives, and models)
    Hint: refer to subsection 3.5 "SMO" in notes.
    """
    #########################################
    # Creating the lists
    iter_num = []
    duals = []
    primals = []
    models = []

    # set dual variables to zero
    #alpha = np.zeros(model.train_y.shape)
    #b = 0

    # begin iteration
    for t in range(max_iters):
        passes = 0

        while passes < max_passes:
            num_changes = 0
            for i in range(model.m):
                
                if model.kernel_func.__name__ == 'linear_kernel':

                    kkt = model.train_y[0,i]*((np.dot(np.multiply(model.alpha, model.train_y), np.dot(model.train_X[:,i],model.train_X))+model.b) - model.train_y[:,i])

                else:
                    kkt = model.train_y[0,i]*((np.dot(np.multiply(model.alpha, model.train_y), Gaussian_kernel(np.array([model.train_X[:,i]]), model.train_X).T)+model.b) - model.train_y[:,i])

                if (kkt < -tol and model.alpha[0,i] < model.C) or (kkt > tol and model.alpha[0,i] > 0):
                    # Random initialize where i != j               
                    list_for_randoms = [z for z in range(model.m)]

                    list_for_randoms.remove(i)
                    j = np.random.choice(list_for_randoms)
                    # Traing Example
                    x = np.array([model.train_X[:,i], model.train_X[:,j]]).T
                    alpha_vec = np.array([[model.alpha[0][i], model.alpha[0][j]]])
                    y_vec = np.array([[model.train_y[0][i], model.train_y[0][j]]])

                    # Upper and Lower Bounds
                    if y_vec[0,0] != y_vec[0,1]: 
                        H = min(model.C, model.C - (alpha_vec[0,0] - alpha_vec[0,1]))
                        L = max(0, -(alpha_vec[0,0] - alpha_vec[0,1]))

                    else:
                        H = min(model.C, alpha_vec[0,0] + alpha_vec[0,1])
                        L = max(0, alpha_vec[0,0] + alpha_vec[0,1] - model.C)

                    # Kernel Used
                    if model.kernel_func.__name__ == 'linear_kernel':
                        K = linear_kernel(x, x)
                    elif model.kernel_func.__name__ == 'Gaussian_kernel':
                        K = Gaussian_kernel(x, x)

                    #g vector with g_1 and g_2 inside for efficiency
                    g_vec = np.dot(np.multiply(alpha_vec,y_vec), K) + model.b

                    # alpha_2 value
                    alpha_new = alpha_vec[0, 1] + ((y_vec[0, 1]*(g_vec[0][0] - y_vec[0, 0] - g_vec[0][1] + y_vec[0, 1]))
                                            /(K[0][0] + K[1][1] - 2*K[0][1]))

                    # alpha_2 classification
                    if alpha_new > H:
                        alpha_new_clipped = H

                    elif alpha_new < L:
                        alpha_new_clipped = L

                    else: 
                        alpha_new_clipped = alpha_new

                    # Stopping if not meaningful change
                    if np.abs(alpha_new_clipped-alpha_vec[0, 1]) <= tol:

                        continue

                    # Getting alpha_1 if meaningful change
                    alpha_1_new = alpha_vec[0, 0] + (y_vec[0, 0]*y_vec[0, 1]) * (alpha_vec[0, 1] - alpha_new_clipped)

                    # b primal
                    # E vector (E_1, E_2)
                    E = g_vec - y_vec
                    b1 = model.b - E[0][0] - y_vec[0, 0]*(alpha_1_new - alpha_vec[0,0])*K[0][0] - y_vec[0, 1]*(alpha_new_clipped - alpha_vec[0, 1])*K[0][1]
                    b2 = model.b - E[0][1] - y_vec[0, 0]*(alpha_1_new - alpha_vec[0,0])*K[0][1] - y_vec[0, 1]*(alpha_new_clipped - alpha_vec[0, 1])*K[1][1]

                    # update model values
                    # alphas
                    model.alpha[0][i] = alpha_1_new
                    model.alpha[0][j] = alpha_new_clipped

                    # b primal
                    if 0 < model.alpha[0][i] < model.C:
                        b_new = b1
                    
                    elif 0 < model.alpha[0][j] < model.C:
                        b_new = b2

                    else:
                        b_new = (b1 + b2)/2

                    model.b = b_new
                    num_changes += 1

            if num_changes == 0:
                passes +=1
            else:
                passes = 0

        iter_num.append(t)
        duals.append(dual_objective_function(model.alpha, model.train_y, model.train_X, model.kernel_func, model.sigma))
        primals.append(primal_objective_function(model.alpha, model.train_y, model.train_X, model.b, model.C, model.kernel_func, model.sigma))
        models.append(vars(model))
    return iter_num, duals, primals, models

def predict(model, test_X):
    """
    Predict the labels of test_X
    model: an SVMModel
    test_X: n x m matrix, test feature vectors
    :return: 1 x m matrix, predicted labels
    """
    #########################################
    vals = decision_function(model.alpha, model.train_y, model.train_X, model.b, model.kernel_func, model.sigma, test_X)
    assignments = np.array([1 if i > 0 else -1 for i in vals[0]])
    return assignments
    #########################################
