import numpy as np
import random

class NN(object):
  def __init__(self, input_layer, hidden_layer, output_layer, learning_constant):
    self.input_layer = input_layer
    self.hidden_layer = hidden_layer
    self.output_layer = output_layer
    self.learning_constant = learning_constant

  #weights
    self.weight_1 = np.zeros([self.input_layer, self.hidden_layer])  # (3x2) weight matrix from input to hidden layer
    self.weight_2 = np.zeros([self.hidden_layer, self.output_layer])  # (3x1) weight matrix from hidden to output layer
    for i in range(0, self.input_layer):
        for j in range(0, self.hidden_layer):
            if random.randint(0, 1) == 1:
                self.weight_1[i][j] = (random.uniform(0.3, 0.7) + 0.3)
            else:
                self.weight_1[i][j] = (random.uniform(0.3, 0.7) - 0.3)
    for i in range(0, self.hidden_layer):
        for j in range(0, self.output_layer):
            if random.randint(0, 1) == 1:
                self.weight_2[i][j] = (random.uniform(0.3, 0.7) + 0.3)
            else:
                self.weight_2[i][j] = (random.uniform(0.3, 0.7) - 0.3)

    #print(self.weight_1)
    #print(self.weight_2)

  def feed_forward(self, input):
    self.z = np.dot(input, self.weight_1) # dot product of X (input) and first set of weights
    self.z2 = self.activate(self.z) # activation function
    self.z3 = np.dot(self.z2, self.weight_2) # dot product of hidden layer (z2) and second set of 3x1 weights
    o = self.activate(self.z3) # final activation function
    return o

  def activate(self, a):
    # activation function
    return 1/(1+np.exp(-a))


  def backward_pass(self, X, y, output):
    # backward propagate through the network
    self.o_error = (y - output) * self.learning_constant # error in output
    self.o_delta = self.o_error*(output*(1-output)) # applying derivative of sigmoid to error

    self.z2_error = self.o_delta.dot(self.weight_2.T) # z2 error: how much our hidden layer weights contributed to output error
    self.z2_delta = self.z2_error*(self.z2*(1-self.z2)) # applying derivative of sigmoid to z2 error

    self.weight_1 += X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
    self.weight_2 += self.z2.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights

  def train(self, input, train):
    output = self.feed_forward(input)
    self.backward_pass(input, train, output)

  def MSE(self, observed, predicted):
    sum = 0.0
    for i in range(0, len(observed)-1):
        sum += (observed[i] - predicted[i])**2
    return (1.0 / len(observed)) * sum




train = np.array(([1, 1], [1, 0], [0, 1], [0, 0]))
teach = np.array(([0], [1], [1], [0]))
NN = NN(2, 2, 1, 1)
MSE = 1
count = 0
while(MSE > 0.05): # trains the NN 1,000 times
  NN.train(train, teach)
  MSE = NN.MSE(teach, NN.feed_forward(train))
  count+=1
print "MSE: "+ str(MSE)
print("Epochs:" + str(count))
print("Predicted: \n" + str(NN.feed_forward(train)))
