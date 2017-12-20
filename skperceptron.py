
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn import datasets, linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from datetime import datetime

start=datetime.now()
class Skperceptron(object):
    
    def fit(self, X1, Y1, X2, Y2):
        for X,Y in zip([X1,X2], [Y1,Y2]):
            X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.33, random_state=42)
        
            n_train, n_labels = np.shape(y_train)
            n_test, n_labels = np.shape(y_test)
            
            y_pred = np.zeros(np.shape(y_test))
            plt.figure()
            iter_range = np.arange(500,5500,500)
            m = np.zeros(len(iter_range))
            for j in range(len(iter_range)):
                
                per = linear_model.Perceptron(n_iter = iter_range[j])
                acc = np.zeros(n_labels)
                for i in range(n_labels):
                # we create an instance of Neighbours Classifier and fit the data.
                    per.fit(X_train, y_train[:,i])
                    
                    
                    y_pred[:,i] = per.predict(X_test)
                    #accuaracy for each label
                    acc[i] = accuracy_score(y_test[:,i], y_pred[:,i], normalize=True)
                        
                
                
                m[j] = np.mean(acc)
                print ("iteration {}".format(iter_range[j]))
                print ("accuracy mean {}, variance {}".format(m[j],np.var(acc)))
                #accuracy for all labels
                print("accuacy {}".format(accuracy_score(y_test, y_pred, normalize=True)))
                if n_labels > 10:
                   break
            plt.plot(iter_range, m, '-o')
            plt.title("BR perceptron for \"emotions\"")
            plt.yticks(np.arange(0, 1.1, 0.1))
            plt.show()


print datetime.now()-start
with open ('output.txt', 'w') as f_out:
    df1 = np.genfromtxt('emotions-train.csv',delimiter=',')
    X1 = df1[1:, :72]
    Y1 = df1[1:, 72:]
    df2 = np.genfromtxt('mediamill-train.csv',delimiter=',')
    X2 = df2[1:, :120]
    Y2 = df2[1:, 120:]
    
    clf = Skperceptron()
    clf.fit(X1,Y1,X2,Y2)
