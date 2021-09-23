#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

def load_dataset(name):
    data = np.loadtxt(name)
    #shuffled when read in
    np.random.shuffle(data)
    
    X = data[:,:9]
    y = data[:,9:]
    return (X, y)

def k_fold_validation_classifier(k, X, y, model):
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

# TODO implement k_fold for regression (or investigate how previous func can be made to work w/ regressors)
def k_fold_validation_regressor(k, X, y, model):
    return (0,0)

# CLASSIFIERS
def KNN_class_get_model(X, y):
    KNN = KNeighborsClassifier(n_neighbors=11)
    KNN.fit(X,y.ravel())
    return KNN

def MLP_class_get_model(X, y, alpha, hidden_layers, max_iter):
    MLP = MLPClassifier(solver='lbfgs', alpha=alpha, hidden_layer_sizes=hidden_layers, random_state=1, max_iter=max_iter)
    MLP.fit(X,y.ravel())
    return MLP

def SVM_class_get_model(X,y):
    SVM = make_pipeline(StandardScaler(), SVC(random_state=0, tol = 1e-5))
    SVM.fit(X,y.ravel())
    return SVM

# REGRESSORS
def KNN_reg_get_model(X, y):
    KNN = KNeighborsRegressor(n_neighbors=11)
    KNN.fit(X, y.ravel())
    return KNN

def MLP_reg_get_model(X, y, alpha, hidden_layers, max_iter):
    MLP = MLPRegressor(solver='lbfgs', alpha=alpha, hidden_layer_sizes=hidden_layers, random_state=1, max_iter=max_iter)
    MLP.fit(X,y.ravel())
    return MLP

def SVM_reg_get_model(X, y):
    # TODO implement with normal functions
    pass

#CLASSIFICATION

#FINAL
#KNN
def run_knn_class_final_accuracy():
    knn_classifier = KNN_class_get_model(tictac_final_X, tictac_final_y)
    acc_avg, norm_conf_matrix = k_fold_validation_classifier(10, tictac_final_X, tictac_final_y, knn_classifier)
    print("Accuracy score for tictac_final, KNN:\n%f" % acc_avg )
    print("Confusion matrix for tictac_final, KNN:")
    print(norm_conf_matrix)

#MLP
def run_mlp_class_final_accuracy():
    mlp_classifier = MLP_class_get_model(tictac_final_X, tictac_final_y, 1e-5, (5,2), 1000)
    acc_avg, norm_conf_matrix = k_fold_validation_classifier(10, tictac_final_X, tictac_final_y, mlp_classifier)
    print("Accuracy score for tictac_final, MLP:\n%f" % acc_avg )
    print("Confusion matrix for tictac_final, MLP:")
    print(norm_conf_matrix)

#SVM
def run_svm_class_final_accuracy():
    SVM_classifier = SVM_class_get_model(tictac_final_X, tictac_final_y)
    acc_avg, norm_conf_matrix = k_fold_validation_classifier(10, tictac_final_X, tictac_final_y, SVM_classifier)
    print("Accuracy score for tictac_final, SVM:\n%f" % acc_avg )
    print("Confusion matrix for tictac_final, SVM:")
    print(norm_conf_matrix)


#SINGLE
#KNN
def run_knn_class_single_accuracy():
    knn_classifier = KNN_class_get_model(tictac_single_X, tictac_single_y)
    acc_avg, norm_conf_matrix = k_fold_validation_classifier(10, tictac_single_X, tictac_single_y, knn_classifier)
    print("Accuracy score for tictac_single, KNN Classifier:\n%f" % acc_avg )
    print("Confusion matrix for tictac_single, KNN Classifier:")
    print(norm_conf_matrix)

#MLP
def run_mlp_class_single_accuracy():
    mlp_classifier = MLP_class_get_model(tictac_single_X, tictac_single_y, 1e-5, (27,27,27), 4000)
    acc_avg, norm_conf_matrix = k_fold_validation_classifier(10, tictac_single_X, tictac_single_y, mlp_classifier)
    print("Accuracy score for tictac_single, MLP Classifier:\n%f" % acc_avg )
    print("Confusion matrix for tictac_single, MLP Classifier:")
    print(norm_conf_matrix)

#SVM
def run_svm_class_single_accuracy():
    SVM_classifier = SVM_class_get_model(tictac_single_X, tictac_single_y)
    acc_avg, norm_conf_matrix = k_fold_validation_classifier(10, tictac_single_X, tictac_single_y, SVM_classifier)
    print("Accuracy score for tictac_single, SVM Classifier:\n%f" % acc_avg )
    print("Confusion matrix for tictac_single, SVM Classifier:")
    print(norm_conf_matrix)



#REGRESSORS

#SINGLE
#KNN
def run_knn_reg_single_accuracy():
    knn_classifier = KNN_reg_get_model(tictac_single_X, tictac_single_y)
    acc_avg, norm_conf_matrix = k_fold_validation_regressor(10, tictac_single_X, tictac_single_y, knn_classifier)
    print("Accuracy score for tictac_single, KNN Regressor:\n%f" % acc_avg )
    print("Confusion matrix for tictac_single, KNN Regressor:")
    print(norm_conf_matrix)

#MLP
def run_mlp_reg_single_accuracy():
    mlp_classifier = MLP_reg_get_model(tictac_single_X, tictac_single_y, 1e-5, (27,27,27), 4000)
    acc_avg, norm_conf_matrix = k_fold_validation_regressor(10, tictac_single_X, tictac_single_y, mlp_classifier)
    print("Accuracy score for tictac_single, MLP Regressor:\n%f" % acc_avg )
    print("Confusion matrix for tictac_single, MLP Regressor:")
    print(norm_conf_matrix)

#SVM
def run_svm_reg_single_accuracy():
    svm_classifier = SVM_reg_get_model(tictac_single_X, tictac_single_y)
    acc_avg, norm_conf_matrix = k_fold_validation_regressor(10, tictac_single_X, tictac_single_y, svm_classifier)
    print("Accuracy score for tictac_single, SVM Regressor:\n%f" % acc_avg )
    print("Confusion matrix for tictac_single, SVM Regressor:")
    print(norm_conf_matrix)


#MULTI
#KNN
def run_knn_reg_multi_accuracy():
    knn_classifier = KNN_reg_get_model(tictac_multi_X, tictac_multi_y)
    acc_avg, norm_conf_matrix = k_fold_validation_regressor(10, tictac_multi_X, tictac_multi_y, knn_classifier)
    print("Accuracy score for tictac_multi, KNN Regressor:\n%f" % acc_avg )
    print("Confusion matrix for tictac_multi, KNN Regressor:")
    print(norm_conf_matrix)

#MLP
def run_mlp_reg_multi_accuracy():
    mlp_classifier = MLP_reg_get_model(tictac_multi_X, tictac_multi_y, 1e-5, (27,27,27), 4000)
    acc_avg, norm_conf_matrix = k_fold_validation_regressor(10, tictac_multi_X, tictac_multi_y, mlp_classifier)
    print("Accuracy score for tictac_multi, MLP Regressor:\n%f" % acc_avg )
    print("Confusion matrix for tictac_multi, MLP Regressor:")
    print(norm_conf_matrix)

#SVM
def run_svm_reg_multi_accuracy():
    svm_classifier = SVM_reg_get_model(tictac_multi_X, tictac_multi_y)
    acc_avg, norm_conf_matrix = k_fold_validation_regressor(10, tictac_multi_X, tictac_multi_y, svm_classifier)
    print("Accuracy score for tictac_multi, SVM Regressor:\n%f" % acc_avg )
    print("Confusion matrix for tictac_multi, SVM Regressor:")
    print(norm_conf_matrix)

# TODO 
# produce confusion matrix plots
# implement regressors on single and multi
# tie regressors to tic tac game
# create report/video
# submit!

if __name__ == '__main__':
    np.set_printoptions(linewidth=1000,precision=3)
    tictac_single_X, tictac_single_y = load_dataset('tictac_single.txt')
    tictac_final_X, tictac_final_y = load_dataset('tictac_final.txt')
    tictac_multi_X, tictac_multi_y = load_dataset('tictac_multi.txt')
    
    run_mlp_reg_single_accuracy()

    #code skeleton for plotting, not sure how to implement w/ k-fold tho
    #plot_confusion_matrix(knn_classifier, testing_X, y_actual)
    #plt.show() 