import numpy as np
import math

train = np.array([[1,1,0,0],[1,0,1,0]])
test = np.array([0,1,1,0])

#print(train)
#print(test)

def mse(y, y_test):
    total = [y.shape[1]]
    print y.shape[1]
    for x in range(0,y.shape[1]):
        total += (y[x] - y_test[x])**2
    mean = np.mean(total)
    print mean
    return mean

def learn():
    while error < 2:
        y = get_output(p)


learn()