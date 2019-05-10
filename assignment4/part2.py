import numpy as np
import argparse
import matplotlib.pyplot as plt
from activation import Activation
from preprocessing import PreProcess, binary_label
from neural_net import NeuralNetwork
from layers import Dense
from other import accuracy_score, CrossEntropy
from other import StochasticGradientDescent as SGD


parser = argparse.ArgumentParser(description='custom deep neural network.')
parser.add_argument('-n', '--number_of_neurons', default=100, help='pass desired number of neurons in a hidden layer.')
inp = parser.parse_args()

neurons = int(inp.number_of_neurons)

pre = PreProcess("Assignment_4_data.txt")
X_train, X_test, y_train, y_test = pre.process(test_size=0.2) 
y_train = binary_label(y_train)
y_test = binary_label(y_test)
y_test.shape


n_features = X_train.shape[1]
print(n_features)

activation = 'sigmoid'
network = NeuralNetwork(optimizer= SGD(learning_rate=0.1), loss=CrossEntropy, validation_data=(X_test, y_test))
network.add(Dense(neurons, input_shape=(n_features,))).add(Activation(activation))
network.add(Dense(neurons)).add(Activation(activation))
network.add(Dense(2)).add(Activation('softmax'))

train_err, test_err = network.fit(X_train, y_train, n_epochs=50, batch_size=200)

fig, ax = plt.subplots(1, 1, figsize=(13,5))
ax.plot(train_err, label="Training Error")
ax.plot(test_err, label="Test Error")
plt.ylabel('Error')
plt.xlabel('Epochs')
plt.legend()
plt.minorticks_on()
plt.show()

y_pred = network.predict(X_test)
accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
print("Accuracy:", accuracy)

fig, ax = plt.subplots(1, 1, figsize=(14, 6))
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
ax.plot(y_true, '--', label='True Label')
ax.plot(y_pred, label='Predicted Label')
plt.legend()
plt.show()

