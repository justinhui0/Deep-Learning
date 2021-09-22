#!/usr/bin/env python3
import numpy as np
#from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline


def load_dataset(name, split_percentage):
    data = np.loadtxt(name)
    #shuffled when read in
    np.random.shuffle(data)
    
    X = data[:,:9]
    y = data[:,9:]
    return (X, y)

def k_fold_validation(k, X, y, model):
    kf = KFold(n_splits=k)
    kf.get_n_splits(X)
    KFold(n_splits=k, random_state=None, shuffle=False)
    acc_score = []
    conf_matrices = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train,y_train.ravel())
        pred_values = model.predict(X_test)
        
        acc = accuracy_score(pred_values , y_test)
        acc_score.append(acc)

        conf_matrix = confusion_matrix(y_test, pred_values)
        conf_matrices.append(conf_matrix)
    avg_acc_score = sum(acc_score)/k
    mean_conf_matrix = np.mean(conf_matrices, axis=0)
    normed_matrix = normalize(mean_conf_matrix, axis=1, norm='l1')
    return avg_acc_score, normed_matrix



split_percentage = 0.8


def knn_get_model(X, y):
    neigh = KNeighborsClassifier(n_neighbors=11)
    neigh.fit(X,y.ravel())
    return neigh

def MLP_get_model(X, y, alpha, hidden_layers, max_iter):
    MLP = MLPClassifier(solver='lbfgs', alpha=alpha, hidden_layer_sizes=hidden_layers, random_state=1, max_iter=max_iter)
    MLP.fit(X,y.ravel())
    return MLP

def SVM_get_model(X,y):
    SVM = make_pipeline(StandardScaler(), SVC(random_state=0, tol = 1e-5))
    SVM.fit(X,y.ravel())
    return SVM

#CLASSIFICATION

#FINAL
#knn
def run_knn_final_accuracy():
    knn_classifier = knn_get_model(tictac_final_X, tictac_final_y)
    acc_avg, norm_conf_matrix = k_fold_validation(10, tictac_final_X, tictac_final_y, knn_classifier)
    print("Accuracy score for tic_tac_final, KNN:\n%f" % acc_avg )
    print("Confusion matrix for tic_tac_final, KNN:")
    print(norm_conf_matrix)

#MLP
def run_mlp_final_accuracy():
    mlp_classifier = MLP_get_model(tictac_final_X, tictac_final_y, 1e-5, (5,2), 1000)
    acc_avg, norm_conf_matrix = k_fold_validation(10, tictac_final_X, tictac_final_y, mlp_classifier)
    print("Accuracy score for tic_tac_final, MLP:\n%f" % acc_avg )
    print("Confusion matrix for tic_tac_final, MLP:")
    print(norm_conf_matrix)

#SVM
def run_svm_final_accuracy():
    SVM_classifier = SVM_get_model(tictac_final_X, tictac_final_y)
    acc_avg, norm_conf_matrix = k_fold_validation(10, tictac_final_X, tictac_final_y, SVM_classifier)
    print("Accuracy score for tic_tac_final, SVM:\n%f" % acc_avg )
    print("Confusion matrix for tic_tac_final, SVM:")
    print(norm_conf_matrix)


#SINGLE
#KNN
def run_knn_single_accuracy():
    knn_classifier = knn_get_model(tictac_single_X, tictac_single_y)
    acc_avg, norm_conf_matrix = k_fold_validation(10, tictac_single_X, tictac_single_y, knn_classifier)
    print("Accuracy score for tic_tac_single, KNN:\n%f" % acc_avg )
    print("Confusion matrix for tic_tac_single, KNN:")
    print(norm_conf_matrix)

#MLP
def run_mlp_single_accuracy():
    mlp_classifier = MLP_get_model(tictac_single_X, tictac_single_y, 1e-5, (27,27,27), 4000)
    acc_avg, norm_conf_matrix = k_fold_validation(10, tictac_single_X, tictac_single_y, mlp_classifier)
    print("Accuracy score for tic_tac_single, MLP:\n%f" % acc_avg )
    print("Confusion matrix for tic_tac_single, MLP:")
    print(norm_conf_matrix)

#SVM
def run_svm_single_accuracy():
    SVM_classifier = SVM_get_model(tictac_single_X, tictac_single_y)
    acc_avg, norm_conf_matrix = k_fold_validation(10, tictac_single_X, tictac_single_y, SVM_classifier)
    print("Accuracy score for tic_tac_single, SVM:\n%f" % acc_avg )
    print("Confusion matrix for tic_tac_single, SVM:")
    print(norm_conf_matrix)

#REGRESSORS

if __name__ == '__main__':
    np.set_printoptions(linewidth=1000,precision=3)
    #tic_tac_single
    tictac_single_X, tictac_single_y = load_dataset('tictac_single.txt', split_percentage)
    
    #tic_tac_final
    tictac_final_X, tictac_final_y = load_dataset('tictac_final.txt', split_percentage)
    
    run_svm_final_accuracy()
    run_svm_single_accuracy()