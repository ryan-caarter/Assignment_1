import numpy as np
import math

class SLP(object):
    def __init__(self, learn_rate, weights, epochs):
        self.learn_rate = learn_rate
        self.weights = weights
        self.epochs = epochs

    def train(self, training, testing):
        for i in range(self.epochs):
            for train, test in zip(training, testing):
                answer = self.activate(train)
                self.weights += self.learn_rate * (test - answer) * train

        print self.weights


    def activate(self, input):
        return 1 / (1 + math.exp(-np.dot(input, self.weights)))


train = np.array([[1,1,0,0],[1,0,1,0]])
test = np.array([0,1,1,0])

weights = ([0, 0, 0, 0])

SLP = SLP(0.05, weights, 50)

SLP.train(train, test)

test_net = np.array([0, 1, 0, 1])
#print SLP.activate(test_net)
