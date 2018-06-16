import numpy as np
from adaline import Adaline

x_train = np.random.randn(1000, 2)
x_test = np.random.randn(100, 2)
w = np.array([2, 16])
b = 18
y_train = np.dot(x_train, w) + b
y_test = np.dot(x_test, w) + b

adaline = Adaline(x_train.shape[1], 1e-3)

epochs = 10
batch_size = 10
for epoch in range(epochs):
  for batch_idx in range(int(np.ceil(x_train.shape[0] / batch_size))):
    batch_start_idx = batch_idx * batch_size
    batch_end_idx = batch_start_idx + batch_size
    if batch_end_idx > x_train.shape[0]:
      batch_end_idx = x_train.shape[0]

    adaline.train(x_train[batch_start_idx:batch_end_idx], y_train[batch_start_idx:batch_end_idx])

print("weights:", adaline.weights)
print("bias:", adaline.bias)
print("root mean squared error on training set:", adaline.root_mean_squared_error(x_train, y_train))
print("root mean squared error on testing set:", adaline.root_mean_squared_error(x_test, y_test))
