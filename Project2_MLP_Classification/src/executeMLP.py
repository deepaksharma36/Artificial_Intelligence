"""
File: executeMLP.py
Language: Python 3.5.1
Author: Deepak Sharma ( ds5930@rit.edu )
        Karan Jariwala( kkj1811@rit.edu )
        Aravindh Kuppusamy ( axk8776@rit.edu )
Description: It takes a trained network weight file and a test data file, and
             runs the network on the test data file. It produce recognition rate,
             profit, confusion matrix, and a class region image
"""

__author__ = "Deepak Sharma, Karan Jariwala, and Aravindh Kuppusamy"

# Importing python module
import numpy as np
import matplotlib.pyplot as plt
from trainMLP import MLP, load_dataset
import sys

# Global Constants
DATA_File = 'train_data.csv'
# DATA_FILE_TEST = "test_data.csv"
# WEIGHTS_FILE='weights.csv'
CLASSES=4
NUM_LAYER=2


def get_netWork_weights(network):
    """
    It returns a list of neural weights
    :param network: MLP (list of layer)
    :return: list(layer) of list(neuron) of np arrays(weights)
    """
    weights=[]
    for layer in network:
        weights.append([])
        for neuron in layer.get_neurons():
            weights[-1].append(neuron.get_weights())
    return weights

def miss_Classification(dataFile,dataset, anetwork):
    """
    Report number of incorrect and correct classified samples in the
    provided dataset
    :param dataFile: datafile/dataset
    :param aNeuron: trained Neuron
    :return: Number of correct and incorrect classified samples
    """
    miss_prediction=[0,0,0,0]
    correct_prediction=[0,0,0,0]
    correct=0
    incorrect=0
    attribute, label = load_dataset(dataFile)
    for sample_index in range(attribute.shape[0]):
        prediction=class_predictor(attribute[sample_index],anetwork)
        if argument_max(label[sample_index])==argument_max(prediction):
            correct_prediction[argument_max(label[sample_index])-1] += 1
            correct+=1
        else:
            miss_prediction[argument_max(label[sample_index])-1] += 1
            incorrect+=1
    accuracy=correct/(correct+incorrect)
    mean_per_class_accuracy=0
    for counter in range(CLASSES):
        correct=correct_prediction[counter]
        incorrect=miss_prediction[counter]
        mean_per_class_accuracy+=(correct)/(CLASSES*(correct+incorrect))
    print("\n*********************************************Accuracy "+dataset+
          "*****************************************************\n")
    print("Accuracy: correct_prediction/total: ",accuracy)
    print("Mean Per Class Accuracy: ",mean_per_class_accuracy)
    print("\n****************************************************************"
          "**************************************************\n")

    return miss_prediction,correct_prediction

def padding(input_num, digit):
    """
    Helper method for padding the input number so that program can maintain
    symmetry in tables
    :param input_num: An input number
    :param digit: number of digits required
    :return: A padding
    """
    padded=str(input_num)
    for pedding_counter in range(digit-len(str(input_num))):
        padded+=" "
    return padded

def print_data(confusion, dataSetName):
    """
    Print the results achieved after classification
    :param confusion: A confusion matrix
    :param dataSetName: A data set Name
    :return: None
    """
    print("************************************* Confusion Matrix for", dataSetName,
          " *******************************************")
    confusion=np.array(confusion)
    digit=5
    print("Pre/Act","*\t",
          padding("Class 1", digit), "\t|\t",
          padding("Class 2", digit), "\t|\t",
          padding("Class 3", digit),"\t|\t",
          padding("Class 4", digit), "\t*\ttotal")
    for counter in range(4):
        print("Class " + str(counter+1),"*\t",
              padding(confusion[counter][0], digit), "\t\t|\t",
              padding(confusion[counter][1], digit), "\t\t|\t",
              padding(confusion[counter][2], digit), "\t\t|\t",
              padding(confusion[counter][3], digit),
          "\t\t*\t", padding(np.sum(confusion[counter]), digit))
    print("*******************************************************************"
          "************************************************")
    print("Total  *\t",
          padding(np.sum(confusion[:, 0]), digit), "\t\t|\t",
          padding(np.sum(confusion[:, 1]), digit), "\t\t|\t",
          padding(np.sum(confusion[:, 2]), digit), "\t\t|\t",
          padding(np.sum(confusion[:, 3]), digit), "\t\t|\t")
    print("\n")

def class_predictor(sample,network):
    """
    A class prediction for an given input using MLP
    :param sample: An input lis of sample
    :param network: MLP
    :return: predicted label [0,0,0,1].......
    """
    value=network.forward_prop(sample)
    output=[]
    max_val=np.max(value)
    for bit in range(len(value)):
        if value[bit] ==max_val:
            output.append(1)
        else:
            output.append(0)
    return output

def argument_max(label):
    """
    It finds the actual class for the given encoding
    :param label: An encoding of a class
    :return: An actual class
    """
    label_num=0
    for counter in range(len(label)):
        label_num+=(counter+1)*label[counter]
    return int(label_num)

def decision_boundary(network, figure, data_file=None):
    """
    It plots a graph of decision boundary and datapoints
    :param network: MLP (list of layer)
    :param datafile data file
    :return: none
    """

    decision_plot = figure.add_subplot(111)
    attribute, label = load_dataset(data_file)
    classes = [1,2,3,4]
    colors_box = ['y', 'b','g','m']
    marker_box = ['>', '+', 'x', 'o']
    label_box = ['bolts', 'nuts', 'rings', 'scrap']
    step = .01

    # creating meshgrid for generating decision
    # boundries using combinations of attributes values
    x1_corr, x2_corr = np.meshgrid(
        np.arange(0, 1, step),
        np.arange(0, 1, step))
    Y_predicted = []
    for i in range(x1_corr.shape[0]):
        Y_predicted.append([])
        for j in range(x1_corr.shape[1]):
            sample=[float(1),x1_corr[i][j],x2_corr[i][j]]
            predicted=argument_max(class_predictor(np.array(sample),network))
            Y_predicted[i].append(predicted)
    # decision_plot.contourf(x1_corr, x2_corr, np.array(Y_predicted))
    plot = decision_plot.contourf(x1_corr, x2_corr, np.array(Y_predicted))
    plt.colorbar(plot, ticks=[1, 2, 3, 4])

    for index in classes:
        x1 = [attribute[i][1] for i in range(len(attribute[:]))
                       if argument_max(label[i]) == index]
        x2 = [attribute[i][2] for i in range(len(attribute[:]))
                       if argument_max(label[i]) == index]
        decision_plot.scatter(x1, x2, c=colors_box[index - 1],
                              marker=marker_box[index-1]
                              ,label=label_box[index-1],
                              s=100)
    decision_plot.legend(loc='upper right')
    decision_plot.set_xlabel("Six fold Rotational Symmetry")
    decision_plot.set_ylabel("Eccentricity")
    decision_plot.set_title("Decision boundary")
    return decision_plot

def get_confusion_matrix(network,data_file, dataSetName):
    """
    Construct confusion matrix by predicting class for given set of data and MLP
    :param network: MLP
    :param data_file: CSV data file
    :return: confusion matrix List of List
    """
    attribute,label=load_dataset(data_file)

    confusion_matrix=[]
    for _ in range(CLASSES):
        confusion_matrix.append([])
        for _ in range(CLASSES):
            confusion_matrix[-1].append(0)
    for sample_counter in range(attribute.shape[0]):
        actual_class=argument_max(label[sample_counter])
        predicted_label=class_predictor(attribute[sample_counter],network)
        predicted_class=argument_max(predicted_label)
        confusion_matrix[int(predicted_class-1)][int(actual_class-1)]+=1

    print_data(confusion_matrix, dataSetName)
    return confusion_matrix

def profit(confusion_matrix):
    profit_values=[[20,-7,-7,-7],[-7,15,-7,-7],[-7,-7,5,-7],[-3,-3,-3,-3]]
    total_profit=0
    for counter in range(CLASSES):
        for inner_counter in range(CLASSES):
            total_profit+=confusion_matrix[counter][inner_counter]*\
                          profit_values[counter][inner_counter]

    print("*******************************************************Profit******"
          "************************************************")
    print("\nTotal Profit: ",total_profit)
    print("*******************************************************************"
          "************************************************")

def main():
    """
    Main method
    return: None
    """
    if len(sys.argv) != 3:
        print("FAILED: Please provide the proper python command line arguments")
        print("Usage: python3 executeMLP <Weight_file> <test_file>")
        print("<Weight_file> = weights csv file")
        print("<test_file> = Test data file")
        sys.exit(1)

    attributes,label=load_dataset(sys.argv[2])
    network=MLP(attributes.shape[1],2)
    network.configure_network_weight(sys.argv[1])
    # miss_Classification(DATA_File,"Training",network)
    miss_Classification(sys.argv[2],"Testing",network)
    # confusion_matrix= get_confusion_matrix(network, DATA_File, "Trained data")
    # profit(confusion_matrix)
    confusion_matrix= get_confusion_matrix(network, sys.argv[2], "Test data")
    profit(confusion_matrix)
    figure = plt.figure()
    decision_boundary(network, figure, sys.argv[2])
    plt.show()

if __name__ == '__main__':
    main()
