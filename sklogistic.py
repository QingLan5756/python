
# first-order strategy with the logistic method

import matplotlib.pyplot as plt
import sys
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from datetime import datetime

start=datetime.now()
class Skregression(object):
    def fit(self, X1, Y1, X2, Y2):
        # run the algorithm through the two datasets
        # ctrl+z to stop after the 500-iteration for the second dataset 'mediamill' has been done
        for X,Y in zip([X1,X2], [Y1,Y2]):
            # randomly select 33% instances to train the dataset,and the rest to predict and test the performance
            X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.33, random_state=42)
            
            n_train, n_labels = np.shape(y_train)
            n_test, n_labels = np.shape(y_test)
            #initialize the prediction label space
            y_pred = np.zeros(np.shape(y_test))
            plt.figure()
            # try 500, 1000, and 5000 iterations
            iter_range = np.arange(500,5500,500)
            m = np.zeros(10)
            for j in range(10):
                logreg = linear_model.LogisticRegression(C=1e5,max_iter=iter_range[j])
                acc = np.zeros(n_labels)
                for i in range(n_labels):
                    # train the dataset
                    logreg.fit(X_train, y_train[:,i])
                    
                    # predict one label at one time
                    y_pred[:,i] = logreg.predict(X_test)
                    # accuracy for every label
                    acc[i] = accuracy_score(y_test[:,i], y_pred[:,i], normalize=True)
                # average accuracy among all labels in each iteration
                m[j] = np.mean(acc)
                print ("iteration {}".format(iter_range[j]))
                print ("accuracy mean {}, variance {}".format(m[j],np.var(acc)))
                #accuracy for all labels in each iteration
                print("accuacy {}".format(accuracy_score(y_test, y_pred, normalize=True)))
                if (n_labels == 101 and j == 1):
                   sys.exit('Due to too much running time, only experiment 500 iterations for mediamill')
        # plot the average accuacy-iteration figure for the first dataset 'emotions'
            plt.plot(iter_range, m, '-o')
            plt.title('Logistic for emotions')
            plt.xlabel('iteration')
            plt.ylabel('average accuacy')
            plt.yticks(np.arange(0, 1.1, 0.1))
            plt.show()



with open ('output.txt', 'w') as f_out:
    df1 = np.genfromtxt('emotions-train.csv',delimiter=',')
    X1 = df1[1:, :72]
    Y1 = df1[1:, 72:]
    df2 = np.genfromtxt('mediamill-train.csv',delimiter=',')
    X2 = df2[1:, :120]
    Y2 = df2[1:, 120:]
    
    clf = Skregression()
    clf.fit(X1,Y1,X2,Y2)

