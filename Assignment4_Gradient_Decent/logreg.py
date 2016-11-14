__author__ = 'Deepak Sharma'

"""
CSCI631 Foundation of inteligent systems
Homework 4
Author: Deepak Sharma (ds5930@g.rit.edu)
Implementation of batch gradient decent optimizer
"""
# Importing Liberaries
import numpy as np
import matplotlib.pyplot as plt
import random

DATA_File = 'non_linear.csv'
Weight_File="weights.csv"
#global constants
CLASSES=2
EPOCHS=1000
LEARNING_RATE=.1

class neuron:
    """
    Represents a neuron unit
    """
    __slots__='num_inputs','num_outputs','weights','inputs','output'

    def __init__(self,inputs=1,outputs=1):
        """
        Intializing perametets
        :param inputs:
        :param outputs:
        """
        self.num_inputs=inputs
        self.num_outputs=outputs
        self.weights=self.__inti_weight__(self.num_inputs)
        self.inputs=None

    def __inti_weight__(self,num_input):
        """
        Intializing neuron weights and bias with small random number
        [bias w1 w2]
        :param num_input: number of weight units for incoming connections
        :return: random weights
        """
        weights=[]
        for counter in range(num_input):
            weights.append(random.randint(0,1))
        weights=np.array(weights)
        weights=weights-np.mean(weights)
        return weights

    def __sigmoid__(self,input):
        """
        implementation of activation function
        :param input: sum(Weights*inputs)
        :return: activation of neuron
        """

        return 1/(1+np.exp(-1*input))

    def __activation__(self,inputs):
        """
        produce activation for given input
        :param inputs: inputs to neuron
        :return: activation
        """
        activation=0
        for counter in range(self.num_inputs):
            activation+=self.weights[counter]*inputs[counter]
        return self.__sigmoid__(activation)
        #return activation+self.bias

    def response(self,inputs):
        """
        Public method for finding activation for given output
        record the state of the neuron for back prop(will be useful in project)
        :param inputs: inputs to the neurons
        :return:
        """
        self.inputs=inputs
        activation=self.__activation__(inputs)
        self.output=activation
        #return [activation for _ in range(self.num_outputs)]
        return activation

    def get_weights(self):
        """
        Return weights of the neuron
        :return: Weight [np array] 1*3
        """
        return self.weights


    def set_weights(self,weights):
        """
        Set weights of the neuron
        :param weights: Np array
        :return: None
        """
        self.weights=weights

def load_dataset(file_name):
    """
    Read data linewise from the input file
    create attribute array with appending 1 (for bias implementation)
    looks like =[1 x1, x2]
    :param file_name:
    :return: np array of attribute and labels
    """
    data=[]
    with open(file_name) as data_file:
        for line in data_file:
            line_list=line.strip().split(",")
            data.append([])
            data[-1].append(float(1))
            for item in line_list:
                data[-1].append(float(item))

    data=np.array(data)
    label = data[:, 3]
    attributes = data[:, 0:3]
    return attributes,label

def gradient_decent(data_file,epoches=EPOCHS):
    """
    Implementation of Batch gradient decent algorithm
    :return: trained Neuron and training SSE history(list)
    """
    #loading data
    attributes,label=load_dataset(data_file)
    #intalizing sum of square error
    SSE=0
    SSE_History=[] #list for storing sse after each epoch
    num_samples=attributes.shape[0]
    aNeuron=neuron(attributes.shape[1],1)
    for epoch in range(EPOCHS):
        new_weight=aNeuron.get_weights()
        for sample in range(num_samples):
            prediction=aNeuron.response(attributes[sample])
            error=label[sample]-prediction
            SSE+=(error)**2
            delta=error*prediction*(1-prediction)
            new_weight+=LEARNING_RATE*delta*attributes[sample]
        aNeuron.set_weights(new_weight)
        write_csv(new_weight)
        SSE=SSE/num_samples
        #storing the Sum of squre error after each epoch
        SSE_History.append(SSE)
        print("After epoch "+str(epoch+1)+ "  SSE: "+str(SSE ))
    return aNeuron,SSE_History

def get_weight_from_file(weight_file=Weight_File):
    with open(weight_file) as weight_file_p:
        for line in weight_file_p:
            pass
        last_line = line
        last=last_line.strip().split(sep=",")
        last=[float(weight) for weight in last]
        return last

def decision_boundry( figure,weight_file=Weight_File,data_file=DATA_File):
    """
    plot decision boundary and datapoints
    :param figure for embedding plot
    :param weight file containing weights achived while training
    :param datafile data file
    :return: none
    """

    decision_plot = figure.add_subplot(122)
    attribute, label = load_dataset(data_file)
    classes = [0, 1]
    colors_box = ['r', 'b']
    #weights=aNeuron.get_weights()
    weights=get_weight_from_file(weight_file)
    bias=weights[0]
    w1=weights[1]
    w2=weights[2]
    x1s=[np.min(attribute[:,1])-.05,np.max(attribute[:,1])+.05]
    x2s_first=(-1*bias-w1*x1s[0])/w2
    x2s_last=(-1*bias-w1*x1s[1])/w2
    x2s=[x2s_first,x2s_last]
    decisionplot=figure.add_subplot(122)
    decisionplot.plot(x1s,x2s)
    for index in classes:
        attribute_1 = [attribute[i][1] for i in range(len(attribute[:]))
                       if label[i] == classes[index]]
        attribute_2 = [attribute[i][2] for i in range(len(attribute[:]))
                       if label[i] == classes[index]]
        decision_plot.scatter(attribute_1, attribute_2, c=colors_box[index - 1]
                              ,label=str(index))
    decision_plot.legend(loc='upper right')
    decision_plot.set_xlabel("X1 attribute")
    decision_plot.set_ylabel("X2 attribute")
    decision_plot.set_title("Decision boundary ")

def SSE__vs_epoch_curve(figure, loss_matrix):
    """
    generating SSE vs epoch curve
    :param figure:
    :param loss_matrix:
    :return: None
    """
    loss__curve = figure.add_subplot(121)
    loss__curve.plot(loss_matrix, label='Training')
    loss__curve.set_title("Epochs Vs SSE")
    loss__curve.set_xlabel("Epochs count")
    loss__curve.set_ylabel(" SSE")
    loss__curve.legend()

def write_csv(weights):
    """
    for storing the weights in a CSV file
    :param weights for recording:
    :return: None
    """
    string=str(weights[0]) + "," +str(weights[1])+ "," +str(weights[2])+"\n"
    fp = open(Weight_File, "a+")
    fp.write(string)
    fp.close()

def class_predictor(value):
    '''
    Assign class label to a activation value
    Threshold for activation value
    :param value: activation value
    :return: predicted class
    '''
    if value>.5:
        return 1
    return 0

def miss_Classification(dataFile,aNeuron):
    """
    Report number of miss and correct classified samples in a provided dataset
    :param dataFile: datafile/dataset
    :param aNeuron: trained Neuron
    :return: Number of miss classified and correct classified samples
    """
    miss_prediction=[0,0]
    correct_prediction=[0,0]
    attribute, label = load_dataset(dataFile)
    for sample_index in range(len(attribute)):
        if class_predictor(aNeuron.response(attribute[sample_index]))\
                !=label[sample_index]:
            miss_prediction[int(label[sample_index])]+=1
        else:
            # print("1: ", sample_index)
            # print("2: ", label[sample_index])
            correct_prediction[int(label[int(sample_index)])]+=1

    return miss_prediction,correct_prediction

def pedding(input_num,digit):
    """
    helper method for padding the input number so that program can maintain
    symtery in tables
    :param input_num: a number
    :param digit: number of digits required
    :return:
    """
    pedded=str(input_num)
    for pedding_counter in range(digit-len(str(input_num))):
        pedded+=" "
    return pedded
def print_data(miss_prediction,correct_prediction):
    """
    Print the results acheived after classification
    :param miss_prediction: list containing value for both classes
    :param correct_prediction: list containing value for both classes
    :return:
    """
    digit=5
    print(pedding("",8),pedding("Correct",digit),"\t\t|\t",pedding("Incorrect",digit),"\t|\ttotal")
    print("Class 0:\t", pedding(correct_prediction[0],digit), "\t\t|\t", pedding(miss_prediction[0],digit),
          "\t\t|\t", pedding(correct_prediction[0]+miss_prediction[0],digit))
    print("Class 1:\t", pedding(correct_prediction[1],digit), "\t\t|\t", pedding(miss_prediction[1],digit),
          "\t\t|\t", pedding(correct_prediction[1]+miss_prediction[1],digit))
    print("Total  :\t", pedding(correct_prediction[1]+correct_prediction[0],digit), "\t\t|\t",
          pedding(miss_prediction[0]+miss_prediction[1],digit), "\t\t|\t")

def main():
    """
    Main method
    :return: None
    """
    datafile=input('provide the csv file name EX: iris.csv')
    #datafile=DATA_File
    aNeuron,SSE_History=gradient_decent(datafile)
    missed,correct = miss_Classification(datafile,aNeuron)
    print_data(missed,correct)
    figure = plt.figure()
    decision_boundry(figure,Weight_File,datafile)
    SSE__vs_epoch_curve(figure, SSE_History)
    figure.show()
    plt.show()


main()
