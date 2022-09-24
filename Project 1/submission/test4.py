from problem4 import *

import sys
import numpy as np
''' 
    Unit test 4:
    This file includes unit tests for problem4.py.
    You could test the correctness of your code by typing `nosetests -v test4.py` in the terminal.
'''

def test_python_version():
    ''' ---------- Problem 4 (0 point in total) ------------'''
    assert sys.version_info[0] == 3  # require python 3 (instead of python 2)

def test_linear():
    ''' (2 points) linear()'''
    theta = np.array([[1.5, 1, 2, 3]]).T
    X = np.array([[1,1], [2, 1], [2, 3], [1, 3]])
    Z = linear(theta,X)

    # test whether or not Z is a numpy array
    assert type(Z) == np.ndarray

    # test whether Z is of the correct shape
    assert Z.shape == (1, 2)

    # test value of Z
    assert np.allclose(Z, np.array([10.5, 17.5]), atol=1e-03)

def test_sigmoid():
    ''' (2 points) sigmoid()'''
    Z = np.array([1, 0, -1])
    assert np.allclose(sigmoid(Z), np.array([0.731, 0.5, 0.2689,]), atol = 1e-03)

def test_loss():
    ''' (3 points) loss()'''
    theta = np.array([1.5, 1, 2]).T
    X = np.array([[1,1],[2, 2], [1, 1]])
    Z = linear(theta, X)
    A = sigmoid(Z)
    Y = np.array([[1, 0]])

    assert np.allclose(loss(A, Y), np.mean([0.00407,5.50407]), atol=1e-03)

def test_dZ():
    ''' (3 points) dZ()'''
    # Z 1 x 2 vector
    Z = np.array([[5.5, 5.5]])
    # Y 1 x 2 vector
    Y = np.array([[1, 0]])

    assert type(dZ(Z, Y)) == np.ndarray

    assert dZ(Z, Y).shape == (1, 2)

    assert np.allclose(dZ(Z, Y), np.array([-0.004, 0.9959]), atol=1e-02)

def test_dtheta():
    ''' (5 points) dtheta()'''
    X = np.array([[2, 2], [1,1]])
    Z = np.array([[5.5, 5.5]])
    Y = np.array([[1, 0]])

    assert type(dtheta(Z, X, Y)) == np.ndarray

    assert dtheta(Z, X, Y).shape == (2, 1)

    assert np.allclose(dtheta(Z, X, Y), np.mean(np.asmatrix([[-0.008, -0.004], [1.9918, 0.9959]]).T, axis=1), atol=1e-03)

def test_Hessian():
    ''' (5 points) Hessian()'''
    X = np.array([[2, 2], [1,1]])
    Z = np.array([[5.5, 5.5]])
    
    assert Hessian(Z, X).shape == (2,2)
    assert np.allclose(Hessian(Z, X), [[0.0324, 0.0162], [0.0162, 0.0081]], atol=1e-03)
