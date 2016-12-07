Read Me:

Main file name: trainMLP.py, executeMLP.py, trainDT.py, executeDT.py
Python version: 3.5.1
Required packages: matplotlib, numpy


1. trainMLP.py:

- Run the program as "python3 trainMLP <train_file.csv> <Number of epochs>"
- It generates SSE vs Epochs graph
- It write the weights in the csv file.

2. executeMLP.py:

- Run the program as "python3 executeMLP <Weight_file> <test_file>"
- It generates a decision boundary graph.
- It prints the accuracy, profit, and confusion matrix for test data.

3. trainDT.py:

- Run the program as "python3 trainDT <train_file>"
- It generates a decision boundary graph for trained data.
- It write the generated tree into DTree.csv( without prune ) and PDTree.csv( with prune ).
- It prints decision tree with and without prune.
- It also prints Number of nodes, number of leaf (decision) nodes, maximum, minimum and average depth of root-to-leaf paths

4. executeDT.py:

- Run the program as "python3 executeDT <decision tree csv file> <test data csv file>"
- It generates a decision boundary graph for test data( with and without prune ).
- It prints the accuracy, profit, and confusion matrix for test data ( with and without prune ).

NOTE:
- For changing the number of neurons in hidden layer please change the global constant in trainMLP.py at line number 25. 