"""
File: trainMLP.py
Language: Python 3.5.1
Author: Karan Jariwala( kkj1811@rit.edu )
        Aravindh Kuppusamy ( axk8776@rit.edu )
        Deepak Sharma ( ds5930@rit.edu )
Description: It takes a file containing training data as input and trained
             neural network weights after some epochs. It uses batch
             gradient descent to trained neural network weights.
"""

__author__ = "Karan Jariwala, Aravindh Kuppusamy, and Deepak Sharma"

# Importing python module
import numpy as np
import matplotlib.pyplot as plt
import random
import os, sys

# Global Constants
DATA_File = 'train_data.csv'
DATA_FILE_TEST = "test_data.csv"
WEIGHTS_FILE='weights'
numNeuronsOutputLayer= 4
numNeuronsHiddenLayer = 5

CLASSES = 4
# EPOCHS = 1000
NUM_LAYER = 2
LEARNING_RATE = 0.1

class neuron:
    """
    Represents a neuron unit in network
    """
    __slots__= ( 'num_inputs','num_outputs','weights','input','activation' )

    def __init__(self,inputs,outputs):
        """
        Initializing parameters
        :param inputs: Number of Inward connections
        :param outputs: Number of outward connection
        :return: None
        """
        self.num_inputs=inputs
        self.num_outputs=outputs
        self.weights=self.__inti_weight__(self.num_inputs)
        self.input=None
        self.activation=None

    def __inti_weight__(self,num_input):
        """
        Initializing neuron weights with small random number
        :param num_input: number of weight units for incoming connections
        :return: random weights
        """
        weights=[]
        for counter in range(num_input):
            weights.append(random.uniform(-1, 1))
        weights=np.array(weights)
        return weights

    def __sigmoid__(self,input):
        """
        Implementation of activation function
        :param input: sum(Weights*inputs)
        :return: activation of neuron
        """
        return 1/(1+np.exp(-1*input))

    def __activation__(self,inputs):
        """
        Produce activation for a given input
        :param inputs: inputs to neuron
        :return: activation
        """
        activation=0
        for counter in range(self.num_inputs):
            activation+=self.weights[counter]*inputs[counter]
        return self.__sigmoid__(activation)

    def response(self,inputs):
        """
        Method for finding activation for a given output
        record the state of the neuron for back propagation
        :param inputs: (list) inputs to the neurons
        :return: activation
        """
        self.input=inputs
        activation=self.__activation__(inputs)
        self.activation=activation
        return activation

    def get_weights(self):
        """
        Return weights of the the neuron
        :return: Weight np array
        """
        return self.weights


    def set_weights(self,weights):
        """
        Set weights of the neuron
        :param weights: Np array of size num_input
        :return: None
        """
        self.weights=weights

class layer:
    """
    Represents a layer in MLP, can contain multiple neurons
    Assumption: each input to the layer is connected to each neuron of the layer
    """
    __slots__= ( 'num_inputs','num_outputs','num_neurons','neurons' )

    def __init__(self,num_inputs=1,num_outputs=1,num_neuron=1):
        """
        Initializing the layer
        :param num_inputs: number of inputs reaching to layer
        :param num_outputs: number of output requires from layer
        :param num_neuron: number of neuron in layer
        """
        self.num_inputs=num_inputs
        self.num_outputs=num_outputs
        self.num_neurons=num_neuron
        self.neurons=self.__init_neurons(num_neuron,num_inputs,num_outputs)

    def __init_neurons(self,num_neurons,inputs,outputs):
        """
        Creating required number of neurons for the layer
        :param num_neurons: required number of neurons(int)
        :param inputs: List of values
        :param outputs: List of output values for the next layer
        :return: list of neurons
        """
        neurons=[]
        for _ in range(num_neurons):
            neurons.append(neuron(inputs,outputs))
        return neurons

    def response(self,inputs):
        """
        Generating response of the layer by collecting activation of each
        neuron of the layer
        :param inputs: input list containing activation of previous/input layer
        :return: list containing response of each neuron of the layer for
        the provided inputs
        """
        response=[]
        for neuron in self.neurons:
            response.append(neuron.response(inputs))
        return response

    def get_neurons(self):
        """
        Getter method to return a list of neuron objects
        :return: return a list of neuron objects
        """
        return self.neurons

    def get_num_neurons(self):
        """
        Getter method to return number of neuron in a layer
        :return: return number of neuron in a layer
        """
        return self.num_neurons

class MLP:
    """
    Representation of a neural network. It contain neuron layers.
    """
    __slot__= ( 'network' )
    def __init__(self,num_input,num_output):
        """
        Creating a MLP with number of input and output channels
        :param num_input:=Number of inputs to MLP
        :param num_output:=Number of outputs from MLP
        :return: None
        """
        a_hidden_layer = layer(num_input, numNeuronsHiddenLayer + 1, numNeuronsHiddenLayer)
        a_Output_layer = layer(numNeuronsHiddenLayer + 1, num_output, numNeuronsOutputLayer)
        self.network=list([a_hidden_layer,a_Output_layer])

    def forward_prop(self,input):
        """
        Generating the activation of the MLP
        :param input: input to the MLP
        :return: return prediction/activation
        """
        activation=input
        for layer in range(NUM_LAYER):
            activation=self.network[layer].response(activation)
            if layer == 0:
                activation.insert(0,1)
        return activation



    def network_update(self,weights):
        """
        Assign weights of the MLP
        :param weights: list(layer) of list(neuron) of np arrays(weights)
        :param network: MLP (list of layer)
        :return: None
        """
        for layer_counter in range(len(self.network)):
            neurons=self.network[layer_counter].get_neurons()
            for neuron_counter in range(len(neurons)):
                neurons[neuron_counter].set_weights(weights[layer_counter][neuron_counter])

    def get_netWork_weights(self):
        """
        It returns the network weights
        :param network: MLP (list of layer)
        :return: list(layer) of list(neuron) of np arrays(weights)
        """
        weights=[]
        for layer in self.network:
            weights.append([])
            for neuron in layer.get_neurons():
                weights[-1].append(neuron.get_weights())
        return weights

    def configure_network_weight(self,weight_file):
        """
        Assign weights to the MLP's neurons provided in text file
        :param weight_file:
        :return:
        """
        file = open(weight_file, "r")
        for line in file:
            pass
        lastline=line
        weight_vector=lastline.strip().split(",")
        weight_counter=0
        for layer in self.network:
            neurons=layer.get_neurons()
            for neuron in neurons:
                weight=[]
                for counter in range(neuron.num_inputs):
                    weight.append(float(weight_vector[weight_counter]))
                    weight_counter+=1
                neuron.set_weights(np.array(weight))

def back_prop(mlp, old_weight,error):
    """
    Implementation of back propagation
    :param old_weight: list(size=layer_count) of list(size=neuron_count) of np arrays(weights)
    :param sample: input
    :param error: list of true_lable-[output layer activation]
    :param prediction: activation of last layer a list
    :param network: MLP
    :return: updated weight
    """
    net_output_layer=mlp.network[-1] # 1 output layer
    output_neurons=net_output_layer.get_neurons()
    previous_delta=[]
    for neuron_counter in range(len(output_neurons)): #
        activation=output_neurons[neuron_counter].activation
        input=output_neurons[neuron_counter].input #list
        dsigmoid = activation*(1-activation)#[ acti * (1 - acti) for acti in activation] # 2 sigmoid
        delta = error[neuron_counter]*dsigmoid#[ err * dsig for err, dsig in zip(error, dsigmoid)] # 2 delta
        dw = [ LEARNING_RATE * delta* inp for inp in input]  # will be a dot product in future
        old_weight[-1][neuron_counter]+=dw #temperary
        previous_delta.append(delta)

    net_hidden_layer = mlp.network[-2]  # 2nd layer
    hidden_neurons = net_hidden_layer.get_neurons() # 3 neuron
    output_weights = []
    for neu in output_neurons:
        output_weights.append( neu.get_weights() ) # 2 list of 4 element each

    hidden_delta = []
    for neuron_counter in range(len(hidden_neurons)):  # 3 neurons
        acti = hidden_neurons[neuron_counter].activation  # 1 activation
        input = hidden_neurons[neuron_counter].input  # 3 elements
        delta = 0
        for delta_counter in range(len(previous_delta)):
            delta += previous_delta[delta_counter] * \
                    output_weights[delta_counter][neuron_counter + 1]
        hidden_delta.append(delta * acti * (1 - acti))
        dw = [LEARNING_RATE * hidden_delta[neuron_counter] * inp for inp in
              input]
        old_weight[-2][neuron_counter] += dw  # temperary
    return old_weight

def load_dataset(file_name):
    """
    Read data line wise from the input file
    create attribute array with appending 1 (for bias implementation)
    looks like =[x1, x2, 1]
    :param file_name:
    :return: np array of attribute and labels
    """
    data=[]
    with open(file_name) as data_file:
        for line in data_file:
            line_list=line.strip().split(",")
            data.append([])
            data[-1].append(float(1))
            data[-1].append(float(line_list[0]))
            data[-1].append(float(line_list[1]))
            if float(line_list[2]) == 1.0:
                data[-1].extend([float(1),float(0),float(0),float(0)])
            if float(line_list[2]) == 2.0:
                data[-1].extend([float(0),float(1),float(0),float(0)])
            if float(line_list[2]) == 3.0:
                data[-1].extend([float(0),float(0),float(1),float(0)])
            if float(line_list[2]) == 4.0:
                data[-1].extend([float(0),float(0),float(0),float(1)])

    data=np.array(data)
    label = data[:, 3:7]
    attributes = data[:, 0:3]
    return attributes,label

def gradient_decent(network, data_file):
    """
    Implementation of Batch gradient decent algorithm
    :return: None
    """

    #loading data
    attributes,label=load_dataset(data_file)

    #initalizing sum of square error
    SSE_History=[] #list for storing sse after each epoch
    num_samples=attributes.shape[0]
    epochs = int(sys.argv[2])

    wt_file = WEIGHTS_FILE + "_" + str(epochs) + ".csv"
    if os.path.isfile(wt_file):
        os.remove(wt_file)

    for epoch in range(epochs):
        SSE = 0
        new_weight=network.get_netWork_weights()
        for sample in range(num_samples):
            prediction=network.forward_prop(attributes[sample])
            error=[]
            for bit_counter in range(len(label[sample])):
                error.append(label[sample][bit_counter] - prediction[bit_counter])
            for bit_error in error:
                SSE+=(bit_error)**2
            new_weight=\
            back_prop(network, new_weight,error)
        network.network_update(new_weight)
        #storing the Sum of squre error after each epoch
        SSE_History.append(SSE)
        write_csv(network)
        print("After epoch "+str(epoch+1)+ "  SSE: "+str(SSE ))
    # write_csv(network)
    return network, SSE_History

def write_csv(network):
    """
    It writes the weights in a CSV file
    :param network: A neuron network
    :return: None
    """
    weight_line=""
    epochs = int(sys.argv[2])
    weights=network.get_netWork_weights()
    for layer_counter in range(len(weights)):
        for neuron_counter in range(len(weights[layer_counter])):
            for weight in weights[layer_counter][neuron_counter]:
                weight_line+=str(weight)+","
    weight_line=weight_line[0:len(weight_line)-1]
    myStr = WEIGHTS_FILE + "_" + str(epochs) + ".csv"
    fp = open(myStr, "a+")
    fp.write(weight_line+"\n")
    fp.close()

def SSE_vs_epoch_curve(figure, loss_matrix):
    """
    It generate a plot of  SSE vs epoch curve
    :param figure: A matplotlib object
    :param loss_matrix: A matrix loss
    :return: None
    """
    loss__curve = figure.add_subplot(111)
    loss__curve.plot(loss_matrix, label='Training')
    loss__curve.set_title("SSE vs Epochs")
    loss__curve.set_xlabel("Epochs count")
    loss__curve.set_ylabel("SSE")
    loss__curve.legend()

def main():
    """
    Main method
    return: None
    """
    if len(sys.argv) != 3:
        print("FAILED: Please provide the proper python command line arguments")
        print("Usage: python3 trainMLP <file> <N>")
        print("<file> = train csv file")
        print("<N> = Number of epochs")
        sys.exit(1)

    network = MLP(3, 4)
    trained_network,SSE_History = gradient_decent(network, sys.argv[1])
    figure = plt.figure()
    SSE_vs_epoch_curve(figure, SSE_History)
    figure.show()
    plt.show()

if __name__=="__main__":
    main()