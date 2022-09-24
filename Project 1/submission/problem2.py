# -------------------------------------------------------------------------
'''
    Problem 2: reading data set from a file, and then split them into training, validation and test sets.

    The functions for handling data

    20/100 points
'''

import numpy as np # for linear algebra

def loadData():
    '''
        Read all labeled examples from the text files.
        Note that the data/X.txt has a row for a feature vector for intelligibility.

        n: number of features
        m: number of examples.

        :return: X: numpy.ndarray. Shape = [n, m]
                y: numpy.ndarray. Shape = [m, ]
    '''
    #########################################
    File_X = open('C:\MFE\MFE Sem 3\CSE 426\CSE_426\Project 1\project1_release\data\X.txt', 'r')
    File_y = open('C:\MFE\MFE Sem 3\CSE 426\CSE_426\Project 1\project1_release\data\y.txt', 'r')

    X_list_str = File_X.readlines()
    y_list_str = File_y.readlines()

    X_list_list_str = [X_list_str[i].split() for i in range(len(X_list_str))]
    X_list_int = [list(map(float, i)) for i in X_list_list_str]
    
    y_list_int = list(map(float, y_list_str)) 

    X = np.array(X_list_int).T
    y = np.array(y_list_int)
    return X, y
    #########################################


def appendConstant(X):
    '''
    Appending constant "1" to the beginning of each training feature vector.
    X: numpy.ndarray. Shape = [n, m]
    :return: return the training samples with the appended 1. Shape = [n+1, m]
    '''
    #########################################
    X = np.insert(X, [0], [[1],], axis=0)
    
    return X 
    #########################################


def splitData(X, y, train_ratio = 0.8):
    '''
	X: numpy.ndarray. Shape = [n+1, m]
	y: numpy.ndarray. Shape = [m, ]
    split_ratio: the ratio of examples go into the Training, Validation, and Test sets.
    Split the whole dataset into Training, Validation, and Test sets.
    :return: return (training_X, training_y), (test_X, test_y).
            training_X is a (n+1, m_tr) matrix with m_tr training examples;
            training_y is a (m_tr, ) column vector;
            test_X is a (n+1, m_test) matrix with m_test test examples;
            training_y is a (m_test, ) column vector.
    '''
    #########################################
    m = X.shape[1]
    index_for_split = int(m*train_ratio)


    train_X = X[:,:index_for_split]
    train_y = y[:index_for_split]

    test_X = X[:, index_for_split:]
    test_Y = y[index_for_split:]

    return (train_X, train_y), (test_X, test_Y)
    #########################################
