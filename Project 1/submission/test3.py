from problem3 import *
import sys

import numpy as np
'''
    Unit test 3:
    This file includes unit tests for problem3.py.
    You could test the correctness of your code by typing `nosetests -v test3.py` in the terminal.
'''

# -------------------------------------------------------------------------
def test_python_version():
    ''' ---------- Problem 3 (0 point in total) ------------'''
    assert sys.version_info[0] == 3  # require python 3 (instead of python 2)


# -------------------------------------------------------------------------
def test_linear():
    ''' (2 points) linear()'''
    theta = np.array([1.5, 1, 2]).T
    x = np.array([1, 2, 1]).T
#    b = 1.5

    assert linear(theta, x) == 5.5

# -------------------------------------------------------------------------
def test_sigmoid():
    ''' (2 points) sigmoid()'''
    z = 1
    assert np.allclose(sigmoid(z), 0.731, atol=1e-02)
    z = 0
    assert np.allclose(sigmoid(z), 0.5, atol=1e-02)
    z = -1
    assert np.allclose(sigmoid(z), 0.2689, atol=1e-02)

# -------------------------------------------------------------------------
def test_loss():
    ''' (3 points) loss()'''
    a = 0.6
    y = 1

    assert np.allclose(loss(a, y), 0.5108, atol=1e-02)
    # assert np.allclose(loss(a, y), -np.log(0.6), atol=1e-03)

    a = 0.6
    y = 0

    assert np.allclose(loss(a, y), 0.9162, atol=1e-02)
    # assert np.allclose(loss(a, y), -np.log(0.4), atol=1e-03)

# -------------------------------------------------------------------------
def test_dz():
    ''' (3 points) dz()'''
    z = 5.5
    y = 1
    assert np.allclose(dz(z, y), -0.004, atol=1e-03)

# -------------------------------------------------------------------------
def test_dtheta():
    ''' (5 points) dtheta()'''
    x = np.array([2, 1])
    z = 5.5
    y = 1
    assert np.allclose(dtheta(z, x, y), [-0.008, -0.004], atol=1e-03)

    y = 0
    assert np.allclose(dtheta(z, x, y), [1.9918, 0.9959], atol=1e-03)


# -------------------------------------------------------------------------
def test_Hessian():
    ''' (5 points) Hessian()'''
    x = np.array([[2], [1]])
    z = 5.5
    
    assert Hessian(z, x).shape == (2,2)
    assert np.allclose(Hessian(z, x), [[0.0163 , 0.0081], [0.0081, 0.0041]], atol=1e-03)
