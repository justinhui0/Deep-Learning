

def predict(X, classifier):
    prediction = classifier.predict(X)
    return prediction

def train(X, y):
    MLP = MLPClassifier(n_neighbors=11)
    MLP.fit(X,y.ravel())
    return neigh

def plt_conf_matrix(classifier, y_actual, y_pred, testing_X): 
    Z = confusion_matrix(y_actual,y_pred)
    print(Z)
    plot_confusion_matrix(classifier, testing_X, y_actual)
    plt.show() 