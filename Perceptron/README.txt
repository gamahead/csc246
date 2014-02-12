Name: Joshua A. Rose
Email: joshua.rose@rochester.edu
Course: CSC246
Homework: "Homework 4, due Wed 2/12 3:25pm by turn_in script - Implement the perceptron algorithm for the voting dataset, and compare results with the previous assignment."
"
************ Files *********
README - General description of code and instructions
perceptron.py - The script implementing the perceptron algorithm
voting2.dat - The data provided for implementing regression. The target variable is a binary value denoting whether the target is republican or democrat, and the explanatory variables denote yes, no, or unknown (1,-1,0) votes made by the target.

************ Algorithm *****
The perceptron algorithm was used to classify the voting data. The perceptron algorithm is an online algorithm, which works by moving through each of the training cases, updating the weight vector when it incorrectly classifies a case by shifting each component of the weight vector by a multiple of an arbitrary constant alpha. If it classifies a case correctly, then the weight vector is not changed.

The formula for adjusting a component is given below, directly from the code:

w[j] = w[j] + alpha*(classify(y[i]) - predicted)*X[i,j]

where 'predicted' is the piecewise function that returns +1 for (X[i] * w) > 0 and -1 otherwise. The classify function is just a helper function that converts the targets values y[i] to 1 or -1 from 1 or 0 on the fly (the 0 or 1 notation is an artifact of reusing code from last week, and this was the easiest way to adapt it). 

************ Instructions ***
Run the command:

> python perceptron.py

************ Results *******
The results of this algorithm vary greatly with the choice of alpha and the number of iterations through the training data (either holding alpha constant or variable). For example, iterating through the training set 7 times with alpha = .05, I was able to produce a categorization accuracy of 90.7692307692% for the test data, which is exactly the same maximum accuracy that was attained by the logistic regression I (mistakenly) implemented for last week's homework. I was able to obtain this accuracy with other alphas, but was not able to obtain a better accuracy. Perhaps shuffling would increase accuracy.

************ Your interpretation ****
The perceptron algorithm's highly variable behavior was dissappointing relative to logistic regression. I may have implemented it poorly, but relatively, logistic regression seems to be a faster performer and more likely to produce an accurate weight vector. 

************ References ************
I made heavy use of both the wikipedia page on perceptrons (http://en.wikipedia.org/wiki/Perceptron) and Bishop. 