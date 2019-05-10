import numpy as np
import copy

class Dense():  
    def __init__(self, n_units, input_shape=None):
        self.layer_input = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
        self.W = None
        self.w0 = None  
    
    def set_input_shape(self, shape):
        self.input_shape = shape
    
    def initialize(self, optimizer):
        limit = 1 / np.sqrt(self.input_shape[0])
        self.W  = np.random.uniform(-limit, limit, (self.input_shape[0], self.n_units))
        self.w0 = np.zeros((1, self.n_units))
        
        self.W_opt  = copy.copy(optimizer)
        self.w0_opt = copy.copy(optimizer)

    def forward_pass(self, X):
        self.layer_input = X
        return X.dot(self.W) + self.w0

    def backward_pass(self, grad):
        W = self.W
        #print grad[0,0]

        if self.trainable:
            grad_w = self.layer_input.T.dot(grad)
            grad_w0 = np.sum(grad, axis=0, keepdims=True)
            self.W = self.W_opt.update(self.W, grad_w)
            self.w0 = self.w0_opt.update(self.w0, grad_w0)
        grad = grad.dot(W.T)
        return grad

    def output_shape(self):
        return (self.n_units, )
