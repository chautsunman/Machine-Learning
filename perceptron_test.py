import numpy as np
from perceptron import Perceptron

x_train = np.random.randn(1000, 2)
x_test = np.random.randn(100, 2)
w = np.array([2, 16])
b = 0
y_train = np.where(np.dot(x_train, w) + b > 0, 1, 0)
y_test = np.where(np.dot(x_test, w) + b > 0, 1, 0)

perceptron = Perceptron(x_train.shape[1], 1e-3)

epochs = 10
batch_size = 10
for epoch in range(epochs):
  for batch_idx in range(int(np.ceil(x_train.shape[0] / batch_size))):
    batch_start_idx = batch_idx * batch_size
    batch_end_idx = batch_start_idx + batch_size
    if batch_end_idx > x_train.shape[0]:
      batch_end_idx = x_train.shape[0]

    perceptron.train(x_train[batch_start_idx:batch_end_idx], y_train[batch_start_idx:batch_end_idx])

print("weights:", perceptron.weights)
print("bias:", perceptron.bias)
print("accuracy on training set:", perceptron.accuracy(x_train, y_train))
print("accuracy on testing set:", perceptron.accuracy(x_test, y_test))
