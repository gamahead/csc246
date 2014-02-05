Name: Joshua A. Rose
Email: joshua.rose@rochester.edu
Course: CSC246
Homework: "Homework 3, due Wed 2/5 3:25pm by turn_in script - Implement linear regression for the voting dataset using Python. Split the 435 examples into 348 train / 45 dev / 42 test and report your classification accuracy on the test set."
"
************ Files *********
README.txt - General description of code and instructions
regression.py - All of the code needed to run logistic regression 
voting2.dat - The data provided for implementing regression. The target variable is a binary value denoting whether the target is republican or democrat, and the explanatory variables denote yes, no, or unknown (1,-1,0) votes made by the target.

************ Algorithm *****
I used logistic regression to classify targets with a typical gradient-descent approach to minimizing the cost function.

The link function used is the inverse logit function: 
logit'(x) = 1/(1+exp(-x))

For calculating cost, I used the cross-entropy erro function defined in Bishop by equation 4.9:
E(w) = Sum{from n=1 to N}(t_n*ln(y_n) + (1-t_n)*ln(1-y_n)) where y_n = logit'(X.dot(w)) and X is the matrix of input variables and w is the parameter vector
-- I mixed up the variable names y and t in my code, however. 

Optimization was accomplished via gradient-decsent of the form: 
w = w - l*grad(E(w)) where grad(E(w)) = X'(w-y) and l = .01. l was decided upon by testing different values of l with different discent iteration counts to find the best combination of the two. 


************ Instructions ***
Run the command:

> python regression.py

************ Results *******
A classification accuracy for the testing set of 90.8% was accomplished.

************ Your interpretation ****
I did not expect such a high accuracy rate given the data, and it was interesting to find how easily regression could make predictions. Given more time, I would like to test different regression schemes like generative models and optimization techniques such as Iterative reweighted least squares. 

************ References ************
The course textbook: Chrisopher Bishop, Pattern Recognition and Machine Learning
Andrew Ng's machine learning lectures on youtube (this lecture in particular: http://www.youtube.com/watch?v=HZ4cvaztQEs&list=PLA89DCFA6ADACE599) were also particularly useful