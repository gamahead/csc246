# -*- coding: utf-8 -*-
"""
author: Joshua Rose
"""

import numpy as np
import csv

e = np.e
log = np.log

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

def logit(X):
    """
    Computes the sigmoid function based on information from pg. 197
    """
    
    return 1.0/(1.0 + np.power(e,-1*X))

def computeCost(y,sig):
    """
    computes the cross-entropy error function based on eq. 4.90
    """
    
    sum = 0
    J = -1.0 * (1.0/y.shape[0])
    
    # Unfortunately, I was having problems with doing this part via numpy's 
    # simply vector operations (in the manner used for computing gradient), 
    # so I ended up actually summing things in order to avoid wasting time
    for n in range(0,y.shape[0]):
        sum = sum + (y[n]*log(sig[n])+(1-y[n])*log(1-sig[n]))

    J = J * sum
    
    return J

def computeGrad(X,y,sig):
    """
    Computer the gradient based on eq. 4.91
    """
    
    grad = 1.0/y.shape[0]
    fix = X.T.dot(sig - y) 
    
    return fix * grad
    
    
def main():
    """
    X=input variable; t=target variable
    """
    X,y = getData()
    
    # add the 1's for the intercept constant    
    X = np.concatenate(((np.zeros(shape=(X.shape[0],1)) + 1),X),axis=1)
    
    # Split data into training and test sets
    X_test = X[370:,:]
    X = X[:370,:] # This is not a testing set for Y
    y_test = y[370:,:]
    y = y[:370,:]
    
    # initialize weight vector
    w = np.zeros(shape = (X.shape[1],1))
    
    # Compute the initial parameters. 
    sigmoid = logit(X.dot(w))
    cost = computeCost(y,sigmoid)
    grad = computeGrad(X,y,sigmoid)
    
    # Gradient-descent implementation to find the min of the cost function
    # by optimizing the weight vector
    for n in range(0,10000):
        w = w - .01 * computeGrad(X,y,logit(X.dot(w)))
    
    # Check prediction accuracy 
    correct = 0.0;
    for n in range(0,y_test.shape[0]):
        p = logit(X_test[n].dot(w))
        if p > .5 and y_test[n] == np.matrix([[1]]):
            correct = correct + 1.0
        if p < .5 and y_test[n] == np.matrix([[0]]):
            correct = correct + 1.0
    
    print("Classification Accuracy for the Testing Set: " + str(100.0 * correct/y_test.shape[0]) + "%")
    

if __name__ == "__main__":
    main()


        
