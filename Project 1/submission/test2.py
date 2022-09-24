from problem2 import *
import sys

import numpy as np
from scipy.sparse import csc_matrix
'''
    Unit test 1:
    This file includes unit tests for problem1.py.
    You could test the correctness of your code by typing `nosetests -v test1.py` in the terminal.
'''

# -------------------------------------------------------------------------
def test_python_version():
    ''' ---------- Problem 2 (0 point in total) ------------'''
    assert sys.version_info[0] == 3  # require python 3 (instead of python 2)


# -------------------------------------------------------------------------
def test_loadData():
    ''' (5 points) loadData()'''
    sample_matrix, label_vector = loadData()

    # test whether or not data_matrix is a numpy matrix
    assert type(sample_matrix) == np.ndarray

    # test the shape of the matrix
    # we want the feature vectors to be in columns
    assert sample_matrix.shape == (7, 400)

    # test whether or not label_vector is a numpy matrix
    assert type(label_vector) == np.ndarray

    # test the shape of the matrix
    assert label_vector.shape == (400, )


# -------------------------------------------------------------------------
def test_splitData():
    ''' (10 points) splitData()'''

    X = np.random.rand(2, 10)
    y = np.random.rand(10)
    (tr_X, tr_y), (test_X, test_y) = splitData(X, y, 0.8)

    # test whether tr is a numpy matrix
    assert type(tr_X) == np.ndarray

    # test the shape of tr
    assert tr_X.shape == (2, 8)

    # check if tr is the first 7 columns
    assert np.allclose(X[:, 0:8], tr_X, atol=1e-3)

    # test whether test is a numpy matrix
    assert type(test_X) == np.ndarray

    # test the shape of test
    assert test_X.shape == (2, 2)

    # check if the test is the last column
    assert np.allclose(X[:, 8:], test_X, atol=1e-3)


# -------------------------------------------------------------------------
def test_appendConstant():
    ''' (5 points) appendConstant()'''
    X = np.asmatrix(np.random.rand(2, 10))

    one_X = appendConstant(X)
    
    # test the shape of the matrix
    assert one_X.shape == (3, 10)
    assert np.allclose(one_X[0, :], np.ones((1, X.shape[1])))
    assert np.allclose(one_X[1:, :], X)
