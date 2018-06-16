import numpy as np

class Adaline:
  def __init__(self, input_dimension, learning_rate):
    self.weights = np.random.randn(input_dimension)
    self.bias = 0
    self.learning_rate = learning_rate

  def predict(self, xs):
    return np.dot(xs, self.weights) + self.bias

  def error(self, xs, ys):
    return ys - self.predict(xs)

  def root_mean_squared_error(self, x, y):
    return np.sqrt(np.mean(np.square(self.error(x, y))))

  def train(self, xs, ys):
    e = self.error(xs, ys)

    dw = np.zeros(self.weights.shape[0])
    db = 0

    for i in range(xs.shape[0]):
      dw += -e[i] * xs[i]
      db += -e[i]

    self.weights -= self.learning_rate * dw
    self.bias -= self.learning_rate * db
