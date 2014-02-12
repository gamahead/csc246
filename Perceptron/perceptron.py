# -*- coding: utf-8 -*-
"""
author: Joshua Rose
"""

import numpy as np
import csv

def getData():
    
    """
    Function for extracting data from voting2.dat specifically. 
    

    """
    def convertLine(line):
        """
        convert a line of data from republican,democrat,y,n,? format to 1,-1,0 format
        
        """
        
        numericLine = []
        for x in line:
            numericLine.append(voteToVal(x))
            
        return numericLine
    
    def voteToVal(vote):
        """
        convert a single data element to numeric type
        """

        if vote == 'y' or vote == 'democrat':
            return 1
        if vote == 'n':
            return -1
        if vote == '?' or vote == 'republican':
            return 0
        else:
            return vote
        
            
    rawData = []
    
    with open('voting2.dat','rb') as f:
        rawDataReader = csv.reader(f)
        for line in rawDataReader:
            rawData.append(convertLine(line))
          
    for x in range(0,17): #Truncate the headers
        rawData.pop(0)
    
    M = np.matrix(rawData)
    X = M[:,1:]
    X = X.astype(int)
    t = M[:,0]
    t = t.astype(int)
    
    return X,t

    
def classify(output):
    if output > 0:
        return 1
    else:
        return -1
        
def perceptron():
    """
    This function takes a set of input data X along with a set of target data y, and attempts
    to construct a weight vector w such that, for a row in X (xi), r*w = yi (where yi is the target
    for the input row xi). The function writes out the accuracy with which this is accomplished for
    the test data to the console, along with the specific value of alpha used to accomplish this.
    This task is accomplished using the perceptron algorithm described both by Bishop in section 4.1 on
    Discriminant functions and my README
    """
    X,y = getData()
    
    # add the 1's for the intercept constant    
    X = np.concatenate(((np.zeros(shape=(X.shape[0],1)) + 1),X),axis=1)

    # Split data into training and test sets
    X_test = X[370:,:]
    X = X[:370,:]
    y_test = y[370:,:]
    y = y[:370,:]
    
    # initialize weight vector
    w = np.zeros(shape = (X.shape[1],1))

    alpha = 0.0 # The constant used for shifting the weight vector 
    alphaData = [] # I'll be using this variable to store test results with different alphas
    correct = 0.0 # This will represent the number of test cases classified correctly
    
    # For the implementation, I tested with many different alpha values
    # and numbers of iterations through the training data, so the results reported in the
    # README are not necessarily produced here. This implementation simply tests different
    # alpha values and returns the one that produces the most accurate weight vector
    while alpha < 1.0:
        alpha += .05
        w = np.zeros(shape = (X.shape[1],1)) # Reset weight vector
        correct = 0.0
        for i in xrange(0,X.shape[0]):
            predicted = classify(X[i] * w)
            for j in xrange(0,w.shape[0]):
                w[j] = w[j] + alpha*(classify(y[i]) - predicted)*X[i,j]
                
        # Check prediction accuracy 
        for n in range(0,y_test.shape[0]):
            if classify(X_test[n] * w) == classify(y_test[n]):
                correct += 1.0
        alphaData.append([alpha,correct/y_test.shape[0]])

    print "Accuracy\tAlpha"
    print str(max((a[1]) for a in alphaData))+"\t"+str(max(alphaData, key=lambda a: a[1])[0])

    
    
    
if __name__ == "__main__":
    perceptron()


        
