#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix, multilabel_confusion_matrix
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

def load_dataset(name):
    data = np.loadtxt(name)
    #shuffled when read in
    np.random.shuffle(data)
    
    X = data[:,:9]
    y = data[:,9:]
    return (X, y)

def linear_k_fold(k, X, y):
    kf = KFold(n_splits=k)
    kf.get_n_splits(X)
    KFold(n_splits=k, random_state=None, shuffle=False)
    acc_score = []
    conf_matrices = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        for i in range(9):
            yi = y_train[:,i].T * 2
            y_test_i = y_test[:,i]
                        
            theta = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(yi)
            curr_pred = X_test.dot(theta)
            curr_pred = curr_pred.round().clip(0,1) 

            acc = accuracy_score(y_test_i, curr_pred)
            acc_score.append(acc)
            #print("our result:",curr_pred)
            #print("expected:",y_test_i)
            conf_matrix = confusion_matrix(y_test_i, curr_pred)
            conf_matrices[i].append(conf_matrix)
            
    avg_acc_score = sum(acc_score)/(k*9)
    mean_conf_matrix = [np.mean(conf_matrices[i], axis=0) for i in range(9)]
    #normed_matrix = normalize(mean_conf_matrix, axis=1, norm='l1')
    return avg_acc_score, mean_conf_matrix
        

def k_fold_validation(k, X, y, model, is_lin_regressor=False, is_multi=False):
    kf = KFold(n_splits=k)
    kf.get_n_splits(X)
    KFold(n_splits=k, random_state=None, shuffle=False)
    acc_score = []
    conf_matrices = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if is_lin_regressor:
            if is_multi:
                pred_values = np.full_like(y_test, 1)
                for i in range(9):

                    curr_out = y_train[:,i]
                    curr_out = curr_out.T
                    theta = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(curr_out)
                    curr_pred = X_test.dot(theta)
                    #print(curr_pred)
                    pred_values = curr_pred.round()
                    #print(pred_values)
                    pred_values = pred_values.clip(0,1) 
                    acc = accuracy_score(y_test[:,i], pred_values )
                    acc_score.append(acc)
                    #print(acc, pred_values, y_test[:,i])
                    # if is_multi:
                    #     conf_matrix = multilabel_confusion_matrix(y_test, pred_values)
                    # else:
                    #     conf_matrix = confusion_matrix(y_test, pred_values)
                    # conf_matrices.append(conf_matrix)
            avg_acc_score = sum(acc_score)/k
            # mean_conf_matrix = np.mean(conf_matrices, axis=0)
            return avg_acc_score, []
            break
                
            # else:
            #     theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
            #     pred_values = X_test.dot(theta)
        elif is_multi:
            model.fit(X_train,y_train)
            pred_values = model.predict(X_test)       
        else:
            model.fit(X_train,y_train.ravel())
            pred_values = model.predict(X_test)
            
        pred_values = pred_values.round()
        if is_multi:
            pred_values = pred_values.clip(0,1)
        else:
            pred_values.clip(0,8)

        # if is_multi:
        #     print(len(pred_values), len(pred_values[0]),pred_values)
        #     print(len(y_test),len(pred_values[0]), y_test)
        #     acc = accuracy_score(np.array([pred_values , y_test]), np.ones((9, 9)))
        # else:
        #print("shapes:", pred_values.shape, y_test.shape, pred_values)
        try:
            acc = accuracy_score(y_test , pred_values )
            acc_score.append(acc)
            if is_multi:
                conf_matrix = multilabel_confusion_matrix(y_test, pred_values)
            else:
                conf_matrix = confusion_matrix(y_test, pred_values)
            conf_matrices.append(conf_matrix)
        except:
            print("failed with")
            print(len(pred_values), len(pred_values[0]),pred_values)
            print(len(y_test),len(pred_values[0]), y_test)

    avg_acc_score = sum(acc_score)/k
    mean_conf_matrix = np.mean(conf_matrices, axis=0)
    if is_multi:
        normed_matrix = np.array([normalize(x, axis=1, norm='l1') for x in mean_conf_matrix])
    else:
        normed_matrix = normalize(mean_conf_matrix, axis=1, norm='l1')
    return avg_acc_score, normed_matrix

# Plot Classier
def pltClassierMatrix(norm_conf_matrix, title):
    _, ax = plt.subplots()
    afig = ax.matshow(norm_conf_matrix,cmap=plt.cm.RdYlGn)
    plt.colorbar(afig)
    ax.set_title(title)
    for i in range(len(norm_conf_matrix)):
        for j in range (len(norm_conf_matrix[0])):
            c = round(norm_conf_matrix[j][i],3)
            ax.text(i, j, str(c), va='center', ha='center')
    plt.show()

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
    KNN = KNeighborsRegressor(n_neighbors=1)
    KNN.fit(X, y)
    return KNN

def MLP_reg_get_model(X, y, alpha, hidden_layers, max_iter):
    MLP = MLPRegressor(solver='lbfgs', alpha=alpha, hidden_layer_sizes=hidden_layers, random_state=1, max_iter=max_iter)
    MLP.fit(X,y)
    return MLP

def SVM_reg_get_model(X, y):
    SVM = LinearRegression()
    SVM.fit(X, y)
    return SVM

#CLASSIFICATION

#FINAL
#KNN
def run_knn_class_final_accuracy():
    knn_classifier = KNN_class_get_model(tictac_final_X, tictac_final_y)
    acc_avg, norm_conf_matrix = k_fold_validation(10, tictac_final_X, tictac_final_y, knn_classifier)
    print("Accuracy score for tictac_final, KNN:\n%f" % acc_avg )
    print("Confusion matrix for tictac_final, KNN:")
    print(norm_conf_matrix)
    pltClassierMatrix(norm_conf_matrix, "KNN Classifier - Final")

#MLP
def run_mlp_class_final_accuracy():
    mlp_classifier = MLP_class_get_model(tictac_final_X, tictac_final_y, 1e-5, (5,2), 1000)
    acc_avg, norm_conf_matrix = k_fold_validation(10, tictac_final_X, tictac_final_y, mlp_classifier)
    print("Accuracy score for tictac_final, MLP:\n%f" % acc_avg )
    print("Confusion matrix for tictac_final, MLP:")
    print(norm_conf_matrix)
    pltClassierMatrix(norm_conf_matrix, "MLP Classifier - Final")

#SVM
def run_svm_class_final_accuracy():
    SVM_classifier = SVM_class_get_model(tictac_final_X, tictac_final_y)
    acc_avg, norm_conf_matrix = k_fold_validation(10, tictac_final_X, tictac_final_y, SVM_classifier)
    print("Accuracy score for tictac_final, SVM:\n%f" % acc_avg )
    print("Confusion matrix for tictac_final, SVM:")
    print(norm_conf_matrix)
    pltClassierMatrix(norm_conf_matrix, "SVM Classifier - Final")

#SINGLE
#KNN
def run_knn_class_single_accuracy():
    knn_classifier = KNN_class_get_model(tictac_single_X, tictac_single_y)
    acc_avg, norm_conf_matrix = k_fold_validation(10, tictac_single_X, tictac_single_y, knn_classifier)
    print("Accuracy score for tictac_single, KNN Classifier:\n%f" % acc_avg )
    print("Confusion matrix for tictac_single, KNN Classifier:")
    print(norm_conf_matrix)
    pltClassierMatrix(norm_conf_matrix, "KNN Classifier - Single")

#MLP
def run_mlp_class_single_accuracy():
    mlp_classifier = MLP_class_get_model(tictac_single_X, tictac_single_y, 1e-5, (27,27,27), 4000)
    acc_avg, norm_conf_matrix = k_fold_validation(10, tictac_single_X, tictac_single_y, mlp_classifier)
    print("Accuracy score for tictac_single, MLP Classifier:\n%f" % acc_avg )
    print("Confusion matrix for tictac_single, MLP Classifier:")
    print(norm_conf_matrix)
    pltClassierMatrix(norm_conf_matrix, "MLP Classifier - Single")

#SVM
def run_svm_class_single_accuracy():
    SVM_classifier = SVM_class_get_model(tictac_single_X, tictac_single_y)
    acc_avg, norm_conf_matrix = k_fold_validation(10, tictac_single_X, tictac_single_y, SVM_classifier)
    print("Accuracy score for tictac_single, SVM Classifier:\n%f" % acc_avg )
    print("Confusion matrix for tictac_single, SVM Classifier:")
    print(norm_conf_matrix)
    pltClassierMatrix(norm_conf_matrix, "SVM Classifier - Single")


#REGRESSORS


#MULTI
#KNN
def run_knn_reg_multi_accuracy():
    knn_classifier = KNN_reg_get_model(tictac_multi_X, tictac_multi_y)
    acc_avg, norm_conf_matrix = k_fold_validation(10, tictac_multi_X, tictac_multi_y, knn_classifier, is_multi=True)
    print("Accuracy score for tictac_multi, KNN Regressor:\n%f" % acc_avg )
    print("Confusion matrix for tictac_multi, KNN Regressor:")

#MLP
def run_mlp_reg_multi_accuracy():
    mlp_classifier = MLP_reg_get_model(tictac_multi_X, tictac_multi_y, 1e-5, (27,27,27), 5000)
    acc_avg, norm_conf_matrix = k_fold_validation(10, tictac_multi_X, tictac_multi_y, mlp_classifier,is_multi=True)
    print("Accuracy score for tictac_multi, MLP Regressor:\n%f" % acc_avg )
    print("Confusion matrix for tictac_multi, MLP Regressor:")

#SVM
#Ref: https://towardsdatascience.com/performing-linear-regression-using-the-normal-equation-6372ed3c57
def run_svm_reg_multi_accuracy():
    acc_avg, norm_conf_matrix = linear_k_fold(10, tictac_multi_X, tictac_multi_y)
     #k_fold_validation(10, tictac_multi_X, tictac_multi_y, None, is_lin_regressor=True, is_multi=True)
    print("Accuracy score for tictac_multi, SVM Regressor:\n%f" % acc_avg )
    print("Confusion matrix for tictac_multi, SVM Regressor:")

# TODO
# make sure works for all tests
# produce confusion matrix plots
# integrate w/ game and test
# create report/video
# cite this https://users.auth.gr/~kehagiat/Research/GameTheory/12CombBiblio/TicTacToe.pdf
# submit!

if __name__ == '__main__':
    np.set_printoptions(linewidth=1000,precision=3)
    tictac_single_X, tictac_single_y = load_dataset('tictac_single.txt')
    tictac_final_X, tictac_final_y = load_dataset('tictac_final.txt')
    tictac_multi_X, tictac_multi_y = load_dataset('tictac_multi.txt')
    
    # run_knn_reg_single_accuracy()
    # run_mlp_reg_single_accuracy()
    # run_svm_reg_single_accuracy()
    run_knn_class_final_accuracy()
    run_mlp_class_final_accuracy()
    run_svm_class_final_accuracy()
    run_knn_class_single_accuracy()
    run_mlp_class_single_accuracy()
    run_svm_class_single_accuracy()

    #run_knn_reg_multi_accuracy()
    #run_mlp_reg_multi_accuracy()
    #run_svm_reg_multi_accuracy()

    #code skeleton for plotting, not sure how to implement w/ k-fold tho
    #plot_confusion_matrix(knn_classifier, testing_X, y_actual)
    #plt.show() 