import keras
import numpy as np
import argparse
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from preprocessing import PreProcess#, to_categorical

parser = argparse.ArgumentParser(description='custom deep neural network.')
parser.add_argument('--number_of_neurons', default=100, help='pass desired number of neurons in a hidden layer.')
inp = parser.parse_args()

n = int(inp.number_of_neurons)

pre = PreProcess("Assignment_4_data.txt")
X_train, X_test, y_train, y_test = pre.process(test_size=0.2) 
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

n_features = X_train.shape[1]
print n_features

activation = 'sigmoid'
model = Sequential()
model.add(Dense(n, activation=activation, input_dim=n_features))
model.add(Dense(n, activation=activation))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer=keras.optimizers.SGD(lr=0.1, momentum=0.0), loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=50, batch_size=200, validation_data=(X_test, y_test), verbose=0)


train_err = history.history['loss']
train_acc = history.history['acc']
test_err = history.history['val_loss']
test_acc = history.history['val_acc']


fig, ax = plt.subplots(1, 1, figsize=(13, 6))
ax.plot(train_err, label='train_error')
ax.plot(test_err, label='test_error')
ax.set_xlabel('Epochs')
ax.set_ylabel('Error')
ax.minorticks_on()
ax.legend()
plt.show()

accuracy = model.evaluate(X_test, y_test, batch_size=200, verbose=0)[1]
print("Final Test Accuracy:", accuracy)

fig, ax = plt.subplots(1, 1, figsize=(14, 6))
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
ax.plot(y_true, '--', label='True Label')
ax.plot(y_pred, label='Predicted Label')
plt.legend()
plt.show()

