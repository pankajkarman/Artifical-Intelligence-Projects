import numpy as np
    
class Sigmoid():
    def __call__(self, x):
    	y = 1 / (1 + np.exp(-x))
        return y

    def gradient(self, x):
    	y = self.__call__(x) 
    	grad = y * (1 - y)
        return grad

class Softmax():
    def __call__(self, x):
        y = np.exp(x - np.max(x, axis=-1, keepdims=True))
        ysum = np.sum(y, axis=-1, keepdims=True)
        grad = y / ysum
        return grad

    def gradient(self, x):
        y = self.__call__(x) 
    	grad = y * (1 - y)
        return grad

class ReLU():
    def __call__(self, x):
    	y = np.where(x >= 0, x, 0)
        return y

    def gradient(self, x):
    	y = np.where(x >= 0, 1, 0)
        return y
        

class Activation(object):
    def __init__(self, name):
        self.activation_name = name
        self.activation_func = self.activation_functions(name)()
        self.trainable = True
        
    def activation_functions(self, name):       
        func = dict()
        func['relu']    = ReLU
        func['sigmoid'] = Sigmoid
        func['softmax'] = Softmax
        return func[name]
        
    def set_input_shape(self, shape):
        self.input_shape = shape

    def forward_pass(self, X):
        self.layer_input = X
        return self.activation_func(X)
        
    def backward_pass(self, grad):
    	#print accum_grad[0,0]
        return grad * self.activation_func.gradient(self.layer_input)

    def output_shape(self):
        return self.input_shape
