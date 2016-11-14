1.Packages required:
numpy
matplotlib

2.Run the program logreg.py using below command
python3 logreg.py

3. It will ask for .csv file name, provide file name Example iris.csv

4. It will print SSE values after each epoch and store the new wights in file
weights.csv. If there is any existing weights.csv then remove it before running
the program.Each row of the weight.csv contains: <bias W1,W2>. Here
W1, W2 and Bias for input attribute X1 and X2 respectively.

5. By the end of training, it will report number of correct and wrong
classified sample for the training dataset.

6. Number of epoch and Learning rate are global constant in program logreg.py
One can change both by changing line by 18 and 19 of the program.

7. It will produce an output figure which contains the SSE vs Epoch curve and
Decision Boundary figure.

Note:- For the implementation of bias,
bias has been added with weights' list as [bias W1, w2 ] and input has been
changed accordingly [1 x1 x2].
