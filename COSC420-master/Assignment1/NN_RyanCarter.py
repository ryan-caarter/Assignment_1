import numpy as np
import sys

__author__ = "Ryan Carter"
__date__ = "4/4/2019"


# NN class, implements back-propagation in a 3 layer NN architecture
class NN(object):
    # NN Constructor, takes the number of perceptrons, the learning constant and the momentum factor
    def __init__(self, input_layer, hidden_layer, output_layer, learning_constant, momentum):
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        self.learning_constant = learning_constant
        self.momentum = momentum

        self.bias1 = np.random.uniform(-0.3, 0.3)
        self.bias2 = np.random.uniform(-0.3, 0.3)

        print self.bias1
        print self.bias2
        print "-----"

        #
        # self.delta1_last = np.random.uniform(-0.3, 0.3, (1, self.hidden_layer))  # last weight changes for each layer, used in momentum
        # self.delta2_last = np.random.uniform(-0.3, 0.3, (1, self.output_layer))

        self.delta1_last = 0.0
        self.delta2_last = 0.0

        print self.delta1_last
        print self.delta2_last
        print "-----"


        # Initialises the weights randomly between the interval [-0.3, 0.3]
        self.weights_1 = np.random.uniform(-0.4, 0.4, (self.input_layer, self.hidden_layer))
        self.weights_2 = np.random.uniform(-0.4, 0.4, (self.hidden_layer, self.output_layer))

        print self.weights_1
        print self.weights_2
        print "-----"

    # Feed forward step, takes input and passes it through the network. Returns the network's output
    def feed_forward(self, teach, train):

        self.dot_1 = np.dot(train, self.weights_1) + self.bias1  # dot product of input and first set of weights

        self.dot_2 = 1 / (1 + np.exp(-self.dot_1))  # get output from first layer using the activation function

        self.dot_3 = np.dot(self.dot_2, self.weights_2) + self.bias2  # dot product of output from last layer and second set of weights

        return 1 / (1 + np.exp(-self.dot_3))  # return the activated output from the last layer




    # Backward pass, uses the generalised delta rule to update the weights backward through the network
    def backward_pass(self, train, teach, output):
        # update second layer of weights using sigmoid derivative, the learning constant, and momentum

        momentum = self.delta2_last * self.momentum
        delta_weight2 = (self.learning_constant * (teach - output) * (output * (1 - output))) #+ momentum
        # update last weight change for layer 2
        self.delta2_last = delta_weight2

        # weight change for first layer of weights, uses ".T" to transpose the weights to calculate dot product
        momentum = self.delta1_last * self.momentum
        delta_weight1 = (np.dot(delta_weight2, self.weights_2.T) * (self.dot_2 * (1 - self.dot_2))) #+ momentum
        # update last weight change for layer 1
        self.delta1_last = delta_weight1

        self.weights_2 += np.dot(self.dot_2.T, delta_weight2)  # update second set of weights
        self.weights_1 += np.dot(train.T, delta_weight1)  # update first set of weights

    # Returns the population error generated by the current prediction and observation
    def pop_error(self, teach, output):
        return sum(sum(np.square(teach - output))) / self.output_layer * len(teach)


# Gives the number of lines in a txt file
def file_length(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


# begin to read in files
try:
    param_file = open(sys.argv[1], "r")
    train_file = open(sys.argv[2], "r")
    teach_file = open(sys.argv[3], "r")
    param = []
    train = []
    teach = []
except IndexError:
    print "Usage: >python NN_RyanCarter.py <parameter-file> <input-file> <teach-file>"  # print correct usage
    sys.exit()

for i in range(0, file_length(sys.argv[1])):
    param.append(param_file.readline())

for i in range(0, file_length(sys.argv[2])):
    temp = []
    line = train_file.readline().strip().split(" ")
    line = filter(None, line)
    for j in range(0, len(line)):
        temp.append(float(line[j]))
    train.append(temp)
train = np.array(train)

for i in range(0, file_length(sys.argv[3])):
    temp = []
    line = teach_file.readline().strip().split(" ")
    line = filter(None, line)
    for j in range(0, len(line)):
        temp.append(float(line[j]))
    teach.append(temp)
teach = np.array(teach)

#close the files after reading
param_file.close()
teach_file.close()
train_file.close()

# constructor call from parameters taken from file input
NN = NN(int(param[0]), int(param[1]), int(param[2]), float(param[3]), float(param[4]))

pop_error = 100  # random large value to start with
epochs = 0

# train until specified error criteria is met
#while pop_error > float(param[5]):
while epochs < 100000:
    output = NN.feed_forward(teach, train)  # set each layers output data fields by doing a forward pass first

    NN.backward_pass(train, teach, output)  # update weights with back propagation
    pop_error = NN.pop_error(teach, NN.feed_forward(teach, train))  # update population error calculation

    epochs += 1
    if epochs % 1000 == 0:
        print "Population error: " + str(pop_error)  # print population error every 100 epochs
        print "Epoch no.: "+str(epochs)
        print NN.feed_forward(teach, train)
        # print "\n"
        # print NN.weights_1
        # print "-----"
        # print NN.weights_2

print "\n------------------------------------\nResults\n------------------------------------"
print "Epochs: " + str(epochs)
print "Population error after reaching threshold: " + str(pop_error)
print("Prediction: \n" + str(NN.feed_forward(teach, train)))