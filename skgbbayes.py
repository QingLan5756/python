# BR using Gaussian naive bayesian
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB



# Code source: Jaques Grobler
# License: BSD 3 clause
class Skbayes(object):
    def fit(self, X, Y):
        
        X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.33, random_state=42)
            
        n_train, n_labels = np.shape(y_train)
        n_test, n_labels = np.shape(y_test)
        
        y_pred = np.zeros(np.shape(y_test))
        gnb = GaussianNB()

        acc = np.zeros(n_labels)
        
        for i in range(n_labels):
            # we create an instance of Neighbours Classifier and fit the data.
            gnb.fit(X_train, y_train[:,i])
                
                
            y_pred[:,i] = gnb.predict(X_test)
            acc[i] = accuracy_score(y_test[:,i], y_pred[:,i], normalize=True)
            print acc[i]
        
        plt.plot(np.arange(n_labels),acc,'-o')
        plt.title("BR using Gaussian naive")
        plt.xlabel("label")
        plt.ylabel("accuracy")
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.show()

        print ("accuracy for all labels {}".format(accuracy_score(y_test, y_pred, normalize=True)))


with open ('output.txt', 'w') as f_out:
    df = np.genfromtxt('emotions-train.csv',delimiter=',')
    X = df[1:, :72]
    Y = df[1:, 72:]
    
    clf = Skbayes()
    clf.fit(X,Y)
