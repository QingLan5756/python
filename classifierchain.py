
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
class Skregression(object):
    def fit(self, X, Y):
        # declare the learning algorithm : logistic regression
        logreg = linear_model.LogisticRegression(C=1e5,max_iter=500)
        
        # run through the two datasets
        
       
        n_instances, n_labels = np.shape(Y)
        n_instances, n_features = np.shape(X)
        # initialize the accuracy array
        acc = np.zeros(n_labels)
        # randomly select 33% example for training adn the rest for testing
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
        y_pred = np.zeros((np.shape(y_test)))
        # the initial X as the instances, and the first label as the
        logreg.fit(X_train, y_train[:,0])
        y_pred[:,0] = logreg.predict(X_test)
        acc[0] = accuracy_score(y_test[:,0], y_pred[:,0], normalize=True)
        
        
        y1 = y_train
        y2 = y_test
        
        for i in range(n_labels-1):
            X_train = np.hstack((X_train,(np.matrix(y_train[:,i])).T))
            y1 = y_train[:,i+1]
            
            X_test = np.hstack((X_test, (np.matrix(y_test[:,i])).T))
            y2 = y_test[:,i+1]
            # we create an instance of Neighbours Classifier and fit the data.
            
            logreg.fit(X_train, y1)

            y_pred[:,i+1] = logreg.predict(X_test)
            acc[i+1] = accuracy_score(y2, y_pred[:,i+1], normalize=True)
        
        print ("dataset 1")
        print ("accuracy mean {} \nvariance {}".format(np.mean(acc), np.var(acc)))
        print ("accuacy for all labels {}".format(accuracy_score(y_test, y_pred, normalize=True)))

        plt.plot(np.arange(n_labels),acc,'-o')
        plt.title("Classifier chain with the original order")
        plt.xlabel("label")
        plt.ylabel("accuracy")
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.show()

with open ('output.txt', 'w') as f_out:
    df = np.genfromtxt('emotions-train.csv',delimiter=',')
    X = df[1:, :72]
    Y = df[1:, 72:]
    
    
    clf = Skregression()
    clf.fit(X,Y)
