import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import main
import math
#check if computer(2) or player(1) won
def get_state(arr):
    #horizontals(3)
    if arr[0] == arr[1] == arr[2] and arr[0] != 0:
        #3 have lined up, if one of them is a 1, player won, else, computer won
        return arr[0]
    elif arr[3] == arr[4] == arr[5] and arr[3] != 0:
        return arr[3]
    elif arr[6] == arr[7] == arr[8] and arr[6] != 0:
        return arr[6]
    
    #verticals(3)
    elif arr[0] == arr[3] == arr[6] and arr[0] != 0:
        return arr[3]

    elif arr[1] == arr[4] == arr[7] and arr[1] != 0:
        return arr[1]

    elif arr[2] == arr[5] == arr[8] and arr[2] != 0:
        return arr[2]

    #diagonals(2)
    elif arr[0] == arr[4] == arr[8] and arr[0] != 0:
        return arr[0]

    elif arr[2] == arr[4] == arr[6] and arr[2] != 0:
        return arr[2]

    #Tie
    elif arr[0] !=  0 and arr[1] !=  0 and arr[2] !=  0 and arr[3] !=  0 and arr[4] !=  0 and arr[5] !=  0 and arr[6] !=  0 and arr[7] !=  0 and arr[8] != 0:
        return 3
        
    #no winner
    else:
        return 0


#prints board, uses array to print 1s or -1s
def printboard(arr):
    print('{:2}'.format(arr[0]), " | ", '{:2}'.format(arr[1]), " | ", '{:2}'.format(arr[2]), "\n",
          '{:2}'.format(arr[3]), " | ", '{:2}'.format(arr[4]), " | ", '{:2}'.format(arr[5]), "\n",
          '{:2}'.format(arr[6]), " | ", '{:2}'.format(arr[7]), " | ", '{:2}'.format(arr[8]), "\n", sep='')


def ValidMove(board, move):
    if 0 > int(move) or int(move) > 8:
        return False
    elif board[int(move)] != 0:
        return False
    else:
        return True

def loadBoard():
    # array that represents board
    board = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    # determines who's turn it is
    turn = input("0 to go first, 1 to go second:\n")
    if turn != "0" and turn != "1":
        turn = print("Error, please enter 0 or 1\n")       
        
    #determines which regresser to play against
    regressor = input("Enter KNN or linear or MLP\n")
    if regressor != "KNN" and regressor != "linear" and regressor != "MLP":
        regressor = input("error, please enter KNN, linear or MLP\n")
    #load data
    tictac_multi_X, tictac_multi_y = main.load_dataset('tictac_multi.txt')

    while not get_state(board):
        if turn == "0":
            move = input("input?\n")
            # checking if valid move
            while(ValidMove(board, int(move)) == False):
                move = input("invalid move\n")
            #update board
            else:
                board[int(move)] = 1
                turn = "1"

        elif turn == "1":
            if(regressor == "KNN"):
                #get model
                KNN = main.KNN_reg_get_model(tictac_multi_X, tictac_multi_y)
                #predict best move 
                best_moves = KNN.predict([board])[0]
                moves_sorted = [(x, best_moves[x]) for x in range(len(best_moves))]

                moves_sorted.sort(key = lambda x: x[1], reverse=True) 
                chosen_move = -1
                for move in moves_sorted:
                    if ValidMove(board, move[0]):
                        chosen_move = move[0]
                        break

                #set board
                board[chosen_move] = -1
                print("computer placed on %d\n" % chosen_move)
                
            elif(regressor == "linear"):
                # Linear = main.SVM_reg_get_model(tictac_multi_X, tictac_multi_y)
                # best_move = int(round((Linear.predict([board])[0])))
                # board[chosen_move] = -1
                # print("computer placed on %d\n" % chosen_move)
                Linear = main.SVM_reg_get_model(tictac_multi_X, tictac_multi_y)
                best_moves = (Linear.predict([board])[0])
                moves_sorted = [(x, best_moves[x]) for x in range(len(best_moves))]

                moves_sorted.sort(key = lambda x: x[1], reverse=True) 
                print(moves_sorted)
                chosen_move = -1
                for move in moves_sorted:
                    if ValidMove(board, move[0]):
                        chosen_move = move[0]
                        break
                print(chosen_move)
                board[chosen_move] = -1
                print("computer placed on %d\n" % chosen_move)
            elif(regressor == "MLP"):
                #MLP = main.MLP_reg_get_model(tictac_multi_X, tictac_multi_y, 1e-5, (27,27,27), 5000)
                best_moves = MLP.predict([board])[0]
                moves_sorted = [(x, best_moves[x]) for x in range(len(best_moves))]

                moves_sorted.sort(key = lambda x: x[1], reverse=True) 
                chosen_move = -1
                for move in moves_sorted:
                    if ValidMove(board, move[0]):
                        chosen_move = move[0]
                        break

                #set board
                board[chosen_move] = -1
                print("computer placed on %d\n" % chosen_move)
                
            turn = "0"

        printboard(board)

    if get_state(board) == 1:
        print("game over! Player won!")
    elif get_state(board) == -1:
        print("game over! Computer won!")
    elif get_state(board) == 3:
        print("game over! Tie!")


def play():

    board = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    printboard(board)
    loadBoard()

if __name__ == '__main__':
    tictac_multi_X, tictac_multi_y = main.load_dataset('tictac_multi.txt')
    MLP = main.MLP_reg_get_model(tictac_multi_X, tictac_multi_y, 1e-5, (27,27,27), 5000)
    play()
