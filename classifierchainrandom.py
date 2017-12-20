
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import random
import itertools
import sys
class Skregression(object):
    def fit(self, X, Y):
        # declare the learning algorithm
        logreg = linear_model.LogisticRegression(C=1e5,max_iter=500)
        
        
        
        n_instances, n_labels = np.shape(Y)
        n_instances, n_features = np.shape(X)
        # the times we permute the labels
        iter = 20
        acc_avg = np.zeros(iter)
        for j in range(iter):
            acc = np.zeros(n_labels)
            label = random.sample(range(0,n_labels),n_labels)
            Y_rand = Y[:,label]
            X_train, X_test, y_train, y_test = train_test_split(X, Y_rand, test_size=0.33, random_state=42)
            y_pred = np.zeros((np.shape(y_test)))
            logreg.fit(X_train, y_train[:,0])
            y_pred[:,0] = logreg.predict(X_test)
            acc[0] = accuracy_score(y_test[:,0], y_pred[:,0], normalize=True)
            
            
            for i in range(n_labels-1):
                X_train = np.hstack((X_train,(np.matrix(y_train[:,i])).T))
                y1 = y_train[:,i+1]
                
                X_test = np.hstack((X_test, (np.matrix(y_test[:,i])).T))
                y2 = y_test[:,i+1]
                # we create an instance of Neighbours Classifier and fit the data.
                
                logreg.fit(X_train, y1)
                
                y_pred[:,i+1] = logreg.predict(X_test)
                acc[i+1] = accuracy_score(y2, y_pred[:,i+1], normalize=True)
            acc_avg[j] = np.mean(acc)
            print ("dataset 1")
            print ("accuracy mean {} \nvariance {}".format(np.mean(acc), np.var(acc)))
            print ("accuacy for all labels {}".format(accuracy_score(y_test, y_pred, normalize=True)))
        print ("mean {} variance {}".format(np.mean(acc_avg), np.var(acc_avg)))
        plt.plot(np.arange(iter), acc_avg,'-o')
        plt.xlabel("iterations")
        plt.ylabel("average accuracies")
        plt.title("random permutations")
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.show()

with open ('output.txt', 'w') as f_out:
    df = np.genfromtxt('emotions-train.csv',delimiter=',')
    X = df[1:, :72]
    Y = df[1:, 72:]
    
    
    clf = Skregression()
    clf.fit(X,Y)
