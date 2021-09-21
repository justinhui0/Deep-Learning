import numpy as np
#from sklearn.svm import LinearSVC
import kNearest

def load_dataset(name, split_percentage):
    data = np.loadtxt(name)
    np.random.shuffle(data)
    training = data[:int(len(data) * split_percentage)]
    testing = data[int(len(data) * split_percentage):]

    training_X = training[:,:9]
    training_y = training[:,9:]

    testing_X = testing[:,:9]
    testing_y = testing[:,9:]

    return ({"X":training_X, "y":training_y}, {"X":testing_X, "y":testing_y})
    
split_percentage = 0.8

#tictac_final
tictac_final_training, tictac_final_testing = load_dataset('tictac_final.txt', split_percentage)
knn_classifier = kNearest.train(tictac_final_training['X'], tictac_final_training['y'])
knn_y_pred = kNearest.predict(tictac_final_testing['X'], knn_classifier)
y_actual = tictac_final_testing['y']
kNearest.plt_conf_matrix(knn_classifier, y_actual, knn_y_pred, tictac_final_testing['X'])

#tictac_single
tictac_single_training, tictac_single_testing = load_dataset('tictac_single.txt', split_percentage)
knn_classifier = kNearest.train(tictac_single_training['X'], tictac_single_training['y'])
knn_y_pred = kNearest.predict(tictac_single_testing['X'], knn_classifier)
y_actual = tictac_single_testing['y']
kNearest.plt_conf_matrix(knn_classifier, y_actual, knn_y_pred, tictac_single_testing['X'])