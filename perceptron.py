import numpy as np

class Perceptron:
  def __init__(self, input_dimension, learning_rate):
    self.weights = np.random.randn(input_dimension)
    self.bias = 0
    self.learning_rate = learning_rate

  def weighted_sum(self, xs):
    return np.dot(xs, self.weights) + self.bias

  def predict(self, xs):
    return 1 / (1 + np.exp(-self.weighted_sum(xs)))

  def accuracy(self, xs, ys):
    predictions = np.where(self.predict(xs) >= 0.5, 1, 0)
    return predictions[predictions == ys].shape[0] / ys.shape[0]

  def train(self, xs, ys):
    predictions = self.predict(xs)

    dw = np.zeros(self.weights.shape[0])
    db = 0

    for i in range(xs.shape[0]):
      e = -(ys[i] - predictions[i])
      dw += e * predictions[i] * (1 - predictions[i]) * xs[i]
      db += e * predictions[i] * (1 - predictions[i])

    self.weights -= self.learning_rate * dw
    self.bias -= self.learning_rate * db
