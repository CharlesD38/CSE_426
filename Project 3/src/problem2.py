# -------------------------------------------------------------------------
'''
    Problem 2: Implement a simple feedforward neural network.
'''

from problem1 import *
import numpy as np
from sklearn.metrics import accuracy_score

class NN:
    #--------------------------
    def __init__(self, dimensions, activation_funcs, loss_func, rand_seed = None):
        """
        Specify an L-layer feedforward network.
        Design consideration: we don't include data in this neural network class.
        Use these passed-in parameters to initialize the hyper-parameters
            (width of each layer, number of layers (depth), activation functions)
        and parameters (W, b) of your self.

        Also define the variables A and Z to be computed.

        It is recommended to use a dictionary with key = layer index and value = parameters/functions
            for easy referencing to these objects using the 1-based indexing of the layers (excluding the input layer).
            For example, W[l] and b[l] will be referring to the parameters of the l-th layer.
                        A[l] and Z[l] will be the activation and linear terms at the l-th layer.
                        g[l] will be the activation function at the l-th layer.

        Being consistent with the notation in the lecture note will make coding and debugging easier.

        dimensions: list of L+1 integers , with dimensions[l+1] and dimensions[l]
                            being the number of rows and columns for the W at layer l+1.
                            dimensions[0] is the dimension of the input data.
                            dimensions[L] is the dimension of output units.
                            dimension[l] = n[l] is the width of layer l in our lecture note.
        activation_funcs: dictionary with key=layer number, value = an activation class (e.g., ReLU)
        loss_func: loss function at the top layer
        rand_seed: set this to a number if you want deterministic experiments.
                    This will be useful for reproducing your bugs for debugging.
        """

        if rand_seed is not None:
            np.random.seed(rand_seed)

        self.num_layers = len(dimensions) - 1
        self.loss_func = loss_func

        self.W = {}
        self.b = {}
        self.g = {}
        num_neurons = {}
        for l in range(self.num_layers):
            num_neurons[l + 1] = dimensions[l + 1]
            # Xavier initialization
            # self.W[l + 1] = np.random.rand(dimensions[l + 1], dimensions[l])
            # self.b[l + 1] = np.random.rand(dimensions[l + 1], 1)
            nin, nout = dimensions[l], dimensions[l + 1]
            sd = np.sqrt(2.0 / (nin + nout))
            self.W[l + 1] = np.random.normal(0.0, sd, (nout, nin))
            self.b[l + 1] = np.zeros((dimensions[l + 1], 1))
            self.g[l + 1] = activation_funcs[l + 1]

        self.A = {}
        self.Z = {}
        self.dZ = {}
        self.dW = {}
        self.db = {}

    #--------------------------
    def forward(self, X):
        """
        Forward computation of activations at each layer.
        The A and Z matrices at all layers will be computed and cached for backprop.
        Vectorize as much as possible and the only loop is to go through the layers.
        :param X: an n[0] x m matrix. m examples with n[0] dimension features.
        :return:  an n[L] x m matrix (the activations at output layer with n[L] neurons)
        """
        #########################################
        # Initializing the input layer
        self.A[0] = X

        # Forward Propogation
        for l in range(self.num_layers):
            self.Z[l+1] = self.W[l+1] @ self.A[l] + self.b[l+1]         # Linear Components
            self.A[l+1] = self.g[l+1].activate(self.Z[l+1])             # Activation

        return self.A[self.num_layers]
        #########################################

    #--------------------------
    def backward(self, Y):
        """
        Back propagation to compute the gradients of parameters at all layers.
        Use the A and Z cached in forward.
        Vectorize as much as possible and the only loop is to go through the layers.
        You should use the gradient of the activation and loss functions defined in problem1.py

        :param Y: an k x m matrix. Each column is the one-hot vector of the label of an training example.

        :return: two dictionaries of gradients of W and b respectively.
                dW[i] is the gradient of the loss to W[i]
                db[i] is the gradient of the loss to b[i]
        """
        #########################################
        B = Y.shape[1]

        for l in reversed(range(self.num_layers)):
            if l == self.num_layers - 1:
                self.dZ[l+1] = self.loss_func.gradient(Y,self.A[l+1])
                self.dW[l+1] = (1/B)*self.dZ[l+1]*self.A[l].T
                self.db[l+1] = (1/B)*self.dZ[l+1]*np.ones((B,1))
            else:
                self.dZ[l+1] = np.multiply(self.W[l+2].T*self.dZ[l+2], self.g[l+1].gradient(self.Z[l+1]))
                self.dW[l+1] = (1/B)*self.dZ[l+1]*self.A[l].T
                self.db[l+1] = (1/B)*self.dZ[l+1]*np.ones((B,1))
        
        return self.dW, self.db
        #########################################

    #--------------------------
    def update_parameters(self, lr, weight_decay = 0.001):
        """
        Use the gradients computed in backward to update all parameters

        :param lr: learning rate.
        """
        #########################################
        for l in reversed(range(self.num_layers)):
            #W = np.array(self.W[l+1] - lr*np.average(self.dW[l+1], axis = 0) - weight_decay*lr*self.W[l+1])
            #B = self.W[l+1].shape[1]
            #averages = []

            #W = self.W[l+1] - lr*(1/B)*np.sum(self.dW[l+1], axis = 1) - lr*weight_decay*self.W[l+1]
            #W = self.W[l+1] - lr*weight_decay*self.W[l+1]
            #W = np.array(self.W[l+1] - lr*np.average(self.dW[l+1], axis = 1) - weight_decay*lr*self.W[l+1])
            W = np.asarray(self.W[l+1] - (lr*self.dW[l+1]+weight_decay*self.W[l+1]))
            b = np.asarray(self.b[l+1] - lr*np.average(self.db[l+1], axis = 1))
            
            self.W[l+1] = W
            self.b[l+1] = b
        #########################################

    #--------------------------
    def train(self, **kwargs):
        """
        Implement mini-batch stochastic gradient descent.

        :param kwargs:
        :return: the loss at the final step
        """
        X_train = np.asmatrix(kwargs['Training X'])
        Y_train = kwargs['Training Y']
        num_samples = X_train.shape[1]
        iter_num = kwargs['max_iters']
        lr = kwargs['Learning rate']
        weight_decay = kwargs['Weight decay']
        batch_size = kwargs['Mini-batch size']

        record_every = kwargs['record_every']

        losses = []
        grad_norms = []

        idx_total = np.array([i for i in range(num_samples)])

        # iterations of mini-batch stochastic gradient descent
        for it in range(iter_num):
            #########################################
            idx_chosen = np.random.choice(idx_total, batch_size)
            batch_X = X_train[:, idx_chosen]
            batch_Y = Y_train[:, idx_chosen]

            self.forward(batch_X)
            self.backward(batch_Y)
            self.update_parameters(lr, weight_decay)

            #########################################
            # tracking the test error during training.
            if (it + 1) % record_every == 0:
                if 'Test X' in kwargs and 'Test Y' in kwargs:
                   prediction_accuracy = self.test(**kwargs)
                   print(', test error = {}'.format(prediction_accuracy))

    #--------------------------
    def test(self, **kwargs):
        """
        Test accuracy of the trained model.
        :return: classification accuracy (for classification) or
                    MSE loss (for regression)
        """
        X_test = np.asmatrix(kwargs['Test X'])
        Y_test = kwargs['Test Y']

        loss_func = kwargs['Test loss function name']

        output = self.forward(X_test)

        if loss_func == '0-1 error':
            predicted_labels = np.argmax(output, axis = 0)
            true_labels = np.argmax(Y_test, axis = 0)
            return 1.0 - accuracy_score(np.array(true_labels).flatten(), np.array(predicted_labels).flatten())
        else:
            # return the MSE (=Frobenius norm of the difference between y and y_hat, divided by (2m))
            return np.linalg.norm(output - Y_test) ** 2 / (2 * Y_test.shape[1])

    # --------------------------
    def explain(self, x, y):
        """
        Given MNIST images from the same class,
            output the explanation of the neural network's prediction of all the 10 classes.

        :return: an matrix of size n x 10, where n is the number of features of a MNIST image.
            We will visualize this in a IPython Notebook.
        """
        #########################################
        self.forward(x)
        d_zz = {}
        for l in reversed(range(self.num_layers)):
            if l == self.num_layers - 1:
                W_c = np.array([self.W[l+1][y,:]])
                d_zz[l]= np.multiply(W_c.T, self.g[l].gradient(self.Z[l]))
            elif 0 < l < self.num_layers - 1:
                d_zz[l] = np.multiply(self.W[l+1].T @ d_zz[l+1], self.g[l].gradient(self.Z[l]))
            else:
                d_zz[l] = self.W[l+1].T @ d_zz[l+1]


        return d_zz[0]
        #########################################
