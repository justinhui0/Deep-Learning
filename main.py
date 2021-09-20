import numpy as np
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
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

def split_dataset(data, percentage):
    training = data[:int(len(data) * percentage)]
    testing = data[int(len(data) * percentage):]
    return (training, testing)

def compare(results, testing_data, dataset):

    FP = FN = TP = TN = 0
    
    y_real = (x[0] for x in testing_data[:,9:])
    #final
    if dataset == 0:
        for test_y, real_y in zip(results, y_real):
            if test_y == real_y and test_y == 1:
                TP += 1
            elif test_y == 1 and real_y == -1:
                FP += 1
            elif test_y == -1 and real_y == 1 :
                FN += 1
            elif test_y == real_y and test_y == -1:
                TN += 1
    #single  
    elif dataset == 1:
        for test_y, real_y in zip(results, y_real):
            if test_y == real_y:
                TP += 1
            elif test_y > real_y:
                FP += 1
            elif test_y < real_y :
                FN += 1
            elif test_y == real_y:
                TN += 1


    if TP + FN == 0: 
        row1 = [0 , 0]
    else:
        row1 = [TP/ (TP + FN), FN / (TP + FN)]
    if TN + FP == 0:
        row2 = [0 , 0]
    else:
        row2 = [FP/ (TN + FP), TN / (TN + FP)] 
    return [row1,row2]
    
    

board = [1,-1,1,1,-1,1,1,-1,-1]
percentage = 0.8


#tictac_final.txt 
tictac_final = np.loadtxt('tictac_final.txt')
np.random.shuffle(tictac_final)
final_training, final_testing = split_dataset(tictac_final, percentage)

#kneigh
neigh = trainKNN(final_training)
knn_results = kneigh(final_testing,neigh)
row1, row2 = compare(knn_results, final_training)
print("TP: %0.2f, FN: %0.2f\nFP: %.2f, TN: %.2f" % (row1[0], row1[1], row2[0], row2[1]))

#SVM

#MLP



#tictac_single.txt
tictac_single = np.loadtxt('tictac_single.txt')
np.random.shuffle(tictac_single)
single_training, single_testing = split_dataset(tictac_single, percentage)

#kneigh
neigh = trainKNN(single_training)
knn_results = kneigh(single_testing,neigh)
row1, row2 = compare(knn_results, single_testing)
print("TP: %0.2f, FN: %0.2f\nFP: %.2f, TN: %.2f" % (row1[0], row1[1], row2[0], row2[1]))

#linearSVM(tictac_final)




#play()
