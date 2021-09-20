from operator import neg
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.index_tricks import diag_indices_from
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import random

#check if any player won
def gameover(arr):
    if arr[0] == arr[1] == arr[2] and arr[0] != 0:
        return True
    elif arr[3] == arr[4] == arr[5] and arr[3] != 0:
        return True
    elif arr[6] == arr[7] == arr[8] and arr[6] != 0:
        return True

    elif arr[0] == arr[3] == arr[6] and arr[0] != 0:
        return True
    elif arr[1] == arr[4] == arr[7] and arr[1] != 0:
        return True
    elif arr[2] == arr[5] == arr[8] and arr[2] != 0:
        return True

    elif arr[0] == arr[4] == arr[8] and arr[0] != 0:
        return True
    elif arr[2] == arr[4] == arr[6] and arr[2] != 0:
        return True

    else:
        return False


#prints board, uses array to print 1s or -1s
def printboard(arr):
    print('{:2}'.format(arr[0]), " | ", '{:2}'.format(arr[1]), " | ", '{:2}'.format(arr[2]), "\n",
          '{:2}'.format(arr[3]), " | ", '{p:2}'.format(arr[4]), " | ", '{:2}'.format(arr[5]), "\n",
          '{:2}'.format(arr[6]), " | ", '{:2}'.format(arr[7]), " | ", '{:2}'.format(arr[8]), "\n", sep='')


def loadBoard():
    # array that represents board
    board = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    # determines who's turn it is
    turn = input("0 to go first, 1 to go second:\n")

    while not gameover(board):
        if turn == "0":
            move = input("input?\n")
            # checking if valid move
            if 0 > int(move) or int(move) > 8:
                print("invalid move\n")
            elif board[int(move)] != 0:
                print("invalid move\n")

            #update board
            else:
                if move == "0":
                    board[0] = 1
                if move == "1":
                    board[1] = 1
                if move == "2":
                    board[2] = 1
                if move == "3":
                    board[3] = 1
                if move == "4":
                    board[4] = 1
                if move == "5":
                    board[5] = 1
                if move == "6":
                    board[6] = 1
                if move == "7":
                    board[7] = 1
                if move == "8":
                    board[8] = 1
                turn = "1"

        elif turn == "1":
            move = input("input?\n")
            if 0 > int(move) or int(move) > 8:
                print("invalid move\n")
            elif board[int(move)] != 0:
                print("invalid move\n")
            else:
                if move == "0":
                    board[0] = -1
                if move == "1":
                    board[1] = -1
                if move == "2":
                    board[2] = -1
                if move == "3":
                    board[3] = -1
                if move == "4":
                    board[4] = -1
                if move == "5":
                    board[5] = -1
                if move == "6":
                    board[6] = -1.,
                if move == "7":
                    board[7] = -1
                if move == "8":
                    board[8] = -1
                turn = "0"

        printboard(board)

    print("game over!")


def play():
    board = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    printboard(board)
    loadBoard()

def linearSVM(data):
    print(data)

def MLP(data, board):
    print("test")

def kneigh(testing, neigh):
    X = testing[:,:9]
    prediction = neigh.predict(X)
    return prediction

def trainKNN(data):
    X = data[:,:9]   #input
    y = data[:,9:]   #output
    neigh = KNeighborsClassifier(n_neighbors=11)
    neigh.fit(X,y.ravel())
    return neigh

def plt_conf_matrix(X_classifer, y_actual, y_pred, training, testing): 
    Z = confusion_matrix(y_actual,y_pred)
    print(Z)
    plot_confusion_matrix(X_classifer, testing[:,:9], y_actual) #cite grepper maybe?
    plt.show()  


def split_dataset(data, percentage):
    training = data[:int(len(data) * percentage)]
    testing = data[int(len(data) * percentage):]
    return (training, testing)



    

board = [1,-1,1,1,-1,1,1,-1,-1]
percentage = 0.8


#tictac_final.txt 
tictac_final = np.loadtxt('tictac_final.txt')
np.random.shuffle(tictac_final)
final_training, final_testing = split_dataset(tictac_final, percentage)

#kneigh
neigh = trainKNN(final_training)    # KNN Classifier
knn_results = kneigh(final_testing,neigh)   # y_pred from KNN
y_actual = final_testing[:,9:] 
plt_conf_matrix(neigh,y_actual,knn_results,final_training,final_testing) 


#SVM

#MLP



#tictac_single.txt
tictac_single = np.loadtxt('tictac_single.txt')
np.random.shuffle(tictac_single)
single_training, single_testing = split_dataset(tictac_single, percentage)

#kneigh
neigh = trainKNN(single_training)
knn_results = kneigh(single_testing,neigh)
y_actual = single_testing[:,9:]
plt_conf_matrix(neigh,y_actual,knn_results,single_training,single_testing)

#linearSVM(tictac_final)




#play()
