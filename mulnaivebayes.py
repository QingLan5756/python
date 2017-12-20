
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import itertools
from sklearn.naive_bayes import GaussianNB

class Skmultilabels(object):
    def fit(self, X, Y):
        n_instances, n_labels = np.shape(Y)
        
        # initialize the new label space vector
        lst = list(itertools.product([0, 1], repeat=n_labels))
        L = np.reshape(lst,(np.power(2,n_labels),n_labels))
        y = np.zeros(n_instances)
        #reconstruct the label space
        for i in range(n_instances):
            [m,] = np.where(np.all(L==Y[i,:],axis=1))
            y[i] = m
        gnb = GaussianNB()
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
        
        
        n_train = len(y_train)
        n_test = len(y_test)
    
        y_pred = gnb.fit(X_train,y_train).predict(X_test)
        
        
        accuracy_score(y_test, y_pred)
        
        print accuracy_score(y_test, y_pred, normalize=True)
        Y_pred = np.zeros((n_test,n_labels))
        Y_test = np.zeros((n_test,n_labels))
        for i in range(n_test):
            Y_pred[i] = L[y_pred[i]]
            Y_test[i] = L[y_test[i]]
        acc = np.zeros(n_labels)
        for i in range(n_labels):
            print accuracy_score(Y_test[:,i], Y_pred[:,i], normalize=True)
            acc[i] = accuracy_score(Y_test[:,i], Y_pred[:,i], normalize=True)

        print np.mean(acc)
with open ('output.txt', 'w') as f_out:
    df = np.genfromtxt('emotions-train.csv',delimiter=',')
    X = df[1:, :72]
    Y = df[1:, 72:]
    
    clf = Skmultilabels()
    clf.fit(X,Y)
