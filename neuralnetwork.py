import numpy as np
from random import random 

# reference: https://www.geeksforgeeks.org/implementation-of-neural-network-from-scratch-using-numpy/
# reference: https://pythonalgos.com/create-a-neural-network-from-scratch-in-python-3/ 
# reference: https://github.com/jonasbostoen/simple-neural-network/blob/master/main.py
# reference: https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

def create_network(input_neuron, output_neuron, hidden_neuron):
    network = list()
    hidden_layer = []
    output_layer = []
    
    hidden_layer = [{'weights':[random() for i in range(input_neuron + 1)]} for i in range(hidden_neuron)]
    output_layer = [{'weights':[random() for i in range(hidden_neuron + 1)]} for i in range(output_neuron)]
    
    network.append(hidden_layer)
    network.append(output_layer)
    
    return network

def activate(weights, vals):
    activation = weights[-1]
    for i in range(len(weights)):
        activation += weights[i] * vals[i]

# sigmoid / logistic function 
def sigmoid(x):
    # sigmoid / logistic function 
    sigmoid = (1/(1 + np.exp(-x)))
    return sigmoid

# compute net input / output of each unit in the hidden and output layers 
def forward_prop(network, row):
    for layer in network:
        new_inputs = []
        for n in layer:
            activation = activate(n['weights'], row)
            n['output'] = sigmoid(activation)
            new_inputs.append(n['output'])
        row = new_inputs 
    
    return row
    #a = np.dot(weights, input) + bias
    #return activation_func(a)

# use the square loss function - take the difference between (true and predicted) squared
def loss_func(act_true, predicted):
    return ((act_true - predicted) ** 2)

# calculate partial derivative
def back_prop(network, expected_vals):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i+1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(neuron['output'] - expected_vals[j])
                
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * sigmoid(neuron['output'])
    #b = activation_func(x)
    #c = b * (1-b)
    #return c
    
def train_network(network, train, epoch, output_neuron):
    for epoch in range(epoch):
        sum_error = 0
        for row in train:
            outputs = forward_prop(network, row)
            expected = [0 for i in range(output_neuron)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            back_prop(network, expected)
            print('>epoch=%d, error=%.3f' % (epoch, sum_error))


if __name__ == '__main__':
    # get user input for training file, test file, 
    # number of neurons per layer + minibatch size 
    
    # attribute values are all numerical except the class attribute
    training_set = input("Enter training .txt file here: ")
    train = []
    input_neuron = int(input("Enter the number of neurons for the input layer: "))
    hidden_neuron = int(input("Enter the number of neurons for the hidden layer: "))
    output_neuron = int(input("Enter the number of neurons for the output layer: "))
    #minibatchSize = input("Enter the minibatch size for SGD: ")
    #test_set = input("Enter test .txt file here: ")
    test = [] 
    network = create_network(input_neuron, output_neuron, hidden_neuron)
    print(network)
    #total_layers = hidden_neuron + output_neuron + input_neuron
    
    # every row will be a row in the training/test data
    with open (training_set, 'r') as f:
        read_data = f.read()
        content_list = read_data.split("\n")
        del(content_list[0])
        classVal = []
    
        for i in content_list:
            a = list(i.split(', '))
            classVal.append(a[-1]) # add the class output to a separate list 
            del(a[-1])
            train.append(np.asarray(a))
            
    
    """ with open (test_set, 'r') as f:
        read_data = f.read()
        content_list = read_data.split("\n")
        del(content_list[0])
    
        for i in content_list:
            a = list(i.split(', '))
            test.append(a) """

    # "weights in the network are initialized to random numbers from interval -1,1"
    attribNum = len(train[0]) 
    print(attribNum)
    #bias =  4 #np.random.randint(0,1,attribNum)
    #weights =  np.array([0,1])#np.random.randint(0,1,attribNum)
    #input = np.array([1.5, 2.2])
    
    train_network(network, train, 20, output_neuron)