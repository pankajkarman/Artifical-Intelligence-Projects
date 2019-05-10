import numpy as np

class NeuralNetwork():
    def __init__(self, optimizer, loss, validation_data=None):
        self.optimizer = optimizer
        self.layers = []
        self.errors = {"training": [], "validation": []}
        self.loss_function = loss()
        self.val_set = None
        
        if validation_data:
            X, y = validation_data
            self.val_set = {"X": X, "y": y} 
            
    @staticmethod        
    def generate_batches(X, y=None, batch_size=64):
    	n_samples = X.shape[0]
    	for i in np.arange(0, n_samples, batch_size):
       	    begin, end = i, min(i+batch_size, n_samples)
            if y is not None:
            	yield X[begin:end], y[begin:end]           
            else:
            	yield X[begin:end]            

    def add(self, layer):
        if self.layers:
            layer.set_input_shape(shape=self.layers[-1].output_shape())
            
        if hasattr(layer, 'initialize'):
            layer.initialize(optimizer=self.optimizer)
            
        self.layers.append(layer)
        return self

    def test_on_batch(self, X, y):
        y_pred = self._forward_pass(X)
        error = np.mean(self.loss_function.loss(y, y_pred))
        score = self.loss_function.acc(y, y_pred)
        return error, score

    def train_on_batch(self, X, y):
        y_pred = self._forward_pass(X)
        error = np.mean(self.loss_function.loss(y, y_pred))
        score = self.loss_function.acc(y, y_pred)
        loss_grad = self.loss_function.gradient(y, y_pred)     
        self._backward_pass(loss_grad=loss_grad)
        return error, score

    def _forward_pass(self, X):
        layer_output = X
        for layer in self.layers:
            layer_output = layer.forward_pass(layer_output)

        return layer_output

    def _backward_pass(self, loss_grad):
        for layer in reversed(self.layers):
            loss_grad = layer.backward_pass(loss_grad)
            
    def fit(self, X, y, n_epochs, batch_size):
        for _ in range(n_epochs):
            
            batch_error = []
            for X_batch, y_batch in self.generate_batches(X, y, batch_size=batch_size):
                loss, _ = self.train_on_batch(X_batch, y_batch)
                batch_error.append(loss)
            self.errors["training"].append(np.mean(batch_error))

            if self.val_set is not None:
                val_loss, _ = self.test_on_batch(self.val_set["X"], self.val_set["y"])
                self.errors["validation"].append(val_loss)
        return self.errors["training"], self.errors["validation"]            

    def predict(self, X):
        return self._forward_pass(X)
