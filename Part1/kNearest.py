#!/usr/bin/env python3
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import KFold
import numpy as np

def predict(X, classifier):
    prediction = classifier.predict(X)
    return prediction

def train(X, y):
    neigh = KNeighborsClassifier(n_neighbors=11)
    neigh.fit(X,y.ravel())
    return neigh

def plot_knn_conf_matrix(classifier, y_actual, y_pred, testing_X): 
    Z = confusion_matrix(y_actual,y_pred)
    print(Z)
    plot_confusion_matrix(classifier, testing_X, y_actual)
    plt.show()