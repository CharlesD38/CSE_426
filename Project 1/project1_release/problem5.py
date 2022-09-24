# -------------------------------------------------------------------------
'''
    Problem 5: Gradient Descent and Newton method Training of Logistic Regression
    20/100 points
'''

import problem3 as p3
import problem4 as p4
from problem2 import *
import numpy as np # linear algebra
import pickle

def batch_gradient_descent(X, Y, X_test, Y_test, num_iters = 50, lr = 0.01, log=True):
    '''
    Train Logistic Regression using Gradient Descent
    X: d x m training sample vectors
    Y: 1 x m labels
    X_test: test sample vectors
    Y_test: test labels
    num_iters: number of gradient descent iterations
    lr: learning rate
    log: True if you want to track the training process, by default True
    :return: (theta, training_log)
    training_log: contains training_loss, test_loss, and norm of theta
    '''
    #########################################
    Y_shaped = np.reshape(Y, (len(Y), 1))
    trainging_log = []
    theta = np.random.rand(len(X), 1)
    # theta = np.atleast_2d([0 for i in range(len(X))]).T
    theta_lst = []
    #theta_lst.append(theta)
    losers=[]
    for i in range(num_iters):

        # Trainging
        
        Z = p4.linear(theta, X)
        # A = p4.sigmoid(Z)
        gradients = p4.dtheta(Z, X, Y)
        theta = theta - lr*gradients

        Z_new = p4.linear(theta, X)

        phi = p4.sigmoid(Z_new)

        loser = p4.loss(phi, Y_shaped)


        if i == 0 or min(losers) > loser:
            losers.append(loser)
            theta_lst.append(theta)
        elif i > 0 and min(losers) < loser:
            loser = losers[-1]
            losers.append(loser)
            theta = theta_lst[-1]
            theta_lst.append(theta)


    theta = theta_lst[-1]
    Y_shaped_testing = np.reshape(Y_test, (len(Y_test), 1))
    losers_test = []

    for i in range(num_iters):

        # Trainging
        
        #Z = p4.linear(theta_lst[i], X_test)
        # A = p4.sigmoid(Z)
        #gradients = p4.dtheta(Z, X_test, Y_test)
        #theta = theta - lr*gradients

        Z_new = p4.linear(theta_lst[i], X_test)

        phi = p4.sigmoid(Z_new)

        loser = p4.loss(phi, Y_shaped_testing)
        losers_test.append(loser)
        trainging_log.append((losers[i], losers_test[i], np.linalg.norm(theta_lst[i], ord =2)))
    
    if log == True:
        return theta_lst[-1], trainging_log
    elif log == False:
        return theta_lst[-1]
    #########################################

def stochastic_gradient_descent(X, Y, X_test, Y_test, num_iters = 50, lr = 0.01, log=True):
    '''
    Train Logistic Regression using Gradient Descent
    X: d x m training sample vectors
    Y: 1 x m labels
    X_test: test sample vectors
    Y_test: test labels
    num_iters: number of gradient descent iterations
    lr: learning rate
    log: True if you want to track the training process, by default True
    :return: (theta, training_log)
    training_log: contains training_loss, test_loss, and norm of theta
    '''
    #########################################
    lr = lr/10
    # Y_shaped = np.reshape(Y, (len(Y), 1))
    trainging_log = []
    #theta = np.array([np.linspace(-0.5,-0.5,8)]).T
    theta = np.random.rand(len(X), 1)
    # theta = np.atleast_2d([0 for i in range(len(X))]).T
    theta_lst = []
    #theta_lst.append(theta)
    losers=[]
    norms = []
    m = X.shape[1]

    loss=[]

    for i in range(num_iters):

        
        idx = np.random.randint(0, m)

        X_sample = np.atleast_2d((X[:,idx])).T
        Z = p3.linear(theta, X_sample)
        # A = p4.sigmoid(Z)
        gradients = p3.dtheta(Z, X_sample, Y[idx])
        theta = theta - lr*gradients

        Z_new = p3.linear(theta, X_sample)

        phi = p3.sigmoid(Z_new)

        loser = p4.loss(phi, Y[idx])
        losers.append(loser)
        loss.append(np.mean(losers))
        theta_lst.append(theta)
        norms.append(np.linalg.norm(theta))
        # chosen.append(idx)

    loss = [np.mean(losers[0:i]) if i > 1 else losers[i] for i in range(len(losers))]
    theta = theta_lst[-1]
    # Y_shaped_testing = np.reshape(Y_test, (len(Y_test), 1))
    losers_test = []

    for i in range(num_iters):

        # Trainging
        
        #Z = p4.linear(theta_lst[i], X_ran_batch_test)
        # A = p4.sigmoid(Z)
        #gradients = p4.dtheta(Z, X_ran_batch_test, Y_test)
        #theta = theta - lr*gradients
        losers_inner = []
        for j in range(len(X_test)):

            Z_new = p3.linear(theta_lst[i], np.atleast_2d(X_test[:, j]).T)

            phi = p3.sigmoid(Z_new)

            loser = p3.loss(phi, Y_test[j])
            losers_inner.append(loser)
            
        losers_test.append(np.mean(losers_inner))

        trainging_log.append((loss[i], losers_test[i], np.linalg.norm(theta_lst[i], ord =2))) 

    if log == True:
        return theta_lst[-1], trainging_log
    elif log == False:
        return theta_lst[-1]
    #########################################


def Newton_method(X, Y, X_test, Y_test, num_iters = 50, log=True):
    '''
    Train Logistic Regression using Gradient Descent
    X: d x m training sample vectors
    Y: 1 x m labels
    X_test: test sample vectors
    Y_test: test labels
    num_iters: number of gradient descent iterations
    log: True if you want to track the training process, by default True
    :return: (theta, training_log)
    training_log: contains training_loss, test_loss, and norm of theta
    '''
    #########################################
    Y_shaped = np.reshape(Y, (len(Y), 1))
    trainging_log = []
    #theta = np.array([np.linspace(-0.5,-0.5,8)]).T
    theta = np.random.rand(len(X), 1)
    # theta = np.atleast_2d([0 for i in range(len(X))]).T
    theta_lst = []
    #theta_lst.append(theta)
    losers=[]

    for i in range(num_iters):

        # Trainging
        
        Z = p4.linear(theta, X)
        # A = p4.sigmoid(Z)
        gradients = p4.dtheta(Z, X, Y)
        hess = p4.Hessian(Z, X)
        hess_inv = np.linalg.inv(hess)


        theta = theta - np.dot(hess_inv,gradients)

        Z_new = p4.linear(theta, X)

        phi = p4.sigmoid(Z_new)

        loser = p4.loss(phi, Y_shaped)

        if i == 0 or min(losers) > loser:
            losers.append(loser)
            theta_lst.append(theta)
        elif i > 0 and min(losers) < loser:
            loser = losers[-1]
            losers.append(loser)
            theta = theta_lst[-1]
            theta_lst.append(theta)


    theta = theta_lst[-1]
    Y_shaped_testing = np.reshape(Y_test, (len(Y_test), 1))
    losers_test = []

    for i in range(num_iters):

        # Trainging
        
        #Z = p4.linear(theta_lst[i], X_test)
        # A = p4.sigmoid(Z)
        #gradients = p4.dtheta(Z, X_test, Y_test)
        #theta = theta - lr*gradients

        Z_new = p4.linear(theta_lst[i], X_test)

        phi = p4.sigmoid(Z_new)

        loser = p4.loss(phi, Y_shaped_testing)
        losers_test.append(loser)
        trainging_log.append((losers[i], losers_test[i], np.linalg.norm(theta_lst[i], ord = 2)))
    if log == True:
        return theta_lst[-1], trainging_log
    elif log == False:
        return theta_lst[-1]
    #########################################


# --------------------------
def train_SGD(**kwargs):
    # use functions defined in problem3.py to perform stochastic gradient descent

    tr_X = kwargs['Training X']
    tr_y = kwargs['Training y']
    te_X = kwargs['Test X']
    te_y = kwargs['Test y']
    num_iters = kwargs['num_iters']
    lr = kwargs['lr']
    log = kwargs['log']
    return stochastic_gradient_descent(tr_X, tr_y, te_X, te_y, num_iters, lr, log)


# --------------------------
def train_GD(**kwargs):
    # use functions defined in problem4.py to perform batch gradient descent

    tr_X = kwargs['Training X']
    tr_y = kwargs['Training y']
    te_X = kwargs['Test X']
    te_y = kwargs['Test y']
    num_iters = kwargs['num_iters']
    lr = kwargs['lr']
    log = kwargs['log']
    return batch_gradient_descent(tr_X, tr_y, te_X, te_y, num_iters, lr, log)

# --------------------------
def train_Newton(**kwargs):
    tr_X = kwargs['Training X']
    tr_y = kwargs['Training y']
    te_X = kwargs['Test X']
    te_y = kwargs['Test y']
    num_iters = kwargs['num_iters']
    log = kwargs['log']
    return Newton_method(tr_X, tr_y, te_X, te_y, num_iters, log)


if __name__ == "__main__":
    '''
    Load and split data, and use the three training methods to train the logistic regression model.
    The training log will be recorded in three files.
    The problem5.py will be graded based on the plots in plot_training_log.ipynb (a jupyter notebook).
    You can plot the logs using the "jupyter notebook plot_training_log.ipynb" on commandline on MacOS/Linux.
    Windows should have similar functionality if you use Anaconda to manage python environments.
    '''
    X, y = loadData()
    X = appendConstant(X)
    (tr_X, tr_y), (te_X, te_y) = splitData(X, y)

    kwargs = {'Training X': tr_X,
              'Training y': tr_y,
              'Test X': te_X,
              'Test y': te_y,
              'num_iters': 1000,
              'lr': 0.01,
              'log': True}

    theta, training_log = train_SGD(**kwargs)
    with open(r'C:\MFE\MFE Sem 3\CSE 426\CSE_426\Project 1\project1_release\data\SGD_outcome.pkl', 'wb') as f:
        pickle.dump((theta, training_log), f)


    theta, training_log = train_GD(**kwargs)
    with open(r'C:\MFE\MFE Sem 3\CSE 426\CSE_426\Project 1\project1_release\data\batch_outcome.pkl', 'wb') as f:
        pickle.dump((theta, training_log), f)
#
#
    theta, training_log = train_Newton(**kwargs)
    with open(r'C:\MFE\MFE Sem 3\CSE 426\CSE_426\Project 1\project1_release\data\newton_outcome.pkl', 'wb') as f:
        pickle.dump((theta, training_log), f)


