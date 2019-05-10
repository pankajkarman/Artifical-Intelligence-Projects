import numpy as np

class StochasticGradientDescent():
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate 

    def update(self, w, wgrad):
    	w -= self.learning_rate * wgrad
        return w
       
class CrossEntropy():
    def __init__(self): pass

    def loss(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-10, 1 - 1e-10)
        err = - y * np.log(p) - (1 - y) * np.log(1 - p)
        return err
     
    @staticmethod    
    def accuracy_score(y_true, y_pred):
        score = np.sum(y_true == y_pred, axis=0) 
        score = score / len(y_true)
        return score

    def acc(self, y, p):
    	y_true = np.argmax(y, axis=1)
    	y_pred = np.argmax(p, axis=1)
    	score  = self.accuracy_score(y_true, y_pred)
        return score

    def gradient(self, y, p):
    	#print p, y
        p = np.clip(p, 1e-10, 1 - 1e-10)
        grad = p - y
        #grad = - (y / p) + (1 - y) / (1 - p)
        return grad
