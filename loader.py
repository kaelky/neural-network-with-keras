import csv
import numpy
import hickle

from keras.models import Sequential
from keras.layers.core import Dense

def load_raw_data(fname):
    X, Y = [], []
    
    with open(fname, 'r') as data:
        reader = csv.reader(data, delimiter=',')
        for row in reader:
            Y.append([int(row[1])])
            temp = []
            for i in range(len(row)):
                if i in [2,3,4,5,6,7,8,9,11,12]:
                    temp.append(int(row[i]))
                elif i in [10, 13]:
                    temp.append(round(float(row[i]), 2))
            X.append(temp)
    
    return X, Y

def load_processed_data(fname):
    data = hickle.load(fname)
    return data

def construct_model(model_definition):
    activations = []
    neurons     = []
    with open(model_definition, 'r') as model:
        for row in model:
            row = row.split(" ")
            row[1] = int(row[1].replace("\n", ""))
            activations.append(row[0])
            neurons.append(row[1])
    
    model = Sequential()
    for activation_function, total_neuron in zip(activations, neurons):
        model.add(Dense(total_neuron, input_dim = 12, activation=str(activation_function)))

    return model

def load_model_information(model_definition):
    activations = []
    with open(model_definition, 'r') as model:
        for row in model:
            row = row.split(" ")
            activations.append(row[0])

    input_layer, hidden_layer, output_layer = str(activations[0]), str(activations[1]), str(activations[2])

    return input_layer, hidden_layer, output_layer
