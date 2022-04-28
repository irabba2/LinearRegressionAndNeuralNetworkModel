import sys
import matplotlib.pyplot as plt
import numpy as num

def plotRegression():
    plt.rcParams['figure.figsize'] = (10.0, 5.0)
    # stores the training points 
    X = []
    Y = []
    inputFile = input("Enter data file: ")
    #parse file to store the points into the lists
    with open(inputFile, 'r') as f:
        readData = f.read()
        lines = readData.split("\n")
        for i in range (1, len(lines)):
            values = lines[i].split(", ")
            X.append(float(values[0]))
            Y.append(float(values[1]))
    plt.scatter(X, Y) #plot the training points 
    #regression line is y=temp0 + (temp1 * x)
    temp0 = 0
    temp1 = 0
    n = float(len(X)) # number of training points 
    learningRate = 0.001
    deriv0Val = [0] * len(X) # initialize the lists to be the same as n
    deriv1Val = [0] * len(X)
    #gradient descent 
    for j in range (1000):
        for i in range (0, len(X)):
            predY = (temp1 * X[i]) + temp0 #current predicted y value 
            deriv0Val[i] = Y[i] - predY 
            deriv1Val[i] = (Y[i]-predY) * X[i]
        derivTemp0 = (-1/n) * sum(deriv0Val)
        derivTemp1 = (-1/n) * sum(deriv1Val)
        temp0 = temp0 - (learningRate * derivTemp0)
        temp1 = temp1 - (learningRate * derivTemp1)
    #print(temp1, temp0)
    regressionLine = [] # to store the predicted y values 
    for i in X:
        regressionLine.append((i * temp1) + temp0)
    plt.scatter(X, regressionLine) # plot the predicted y values 
    plt.plot([min(X), max(X)], [min(regressionLine), max(regressionLine)], color='orange') #plot regression line 
    plt.show()

if __name__ == '__main__':
    plotRegression()

    