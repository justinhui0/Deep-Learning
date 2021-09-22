#check if any player won
def get_state(arr):
    if arr[0] == arr[1] == arr[2] and arr[0] != 0:
        return arr[0]
    elif arr[3] == arr[4] == arr[5] and arr[3] != 0:
        return arr[3]
    elif arr[6] == arr[7] == arr[8] and arr[6] != 0:
        return arr[6]

    elif arr[0] == arr[3] == arr[6] and arr[0] != 0:
        return arr[0]
    elif arr[1] == arr[4] == arr[7] and arr[1] != 0:
        return arr[1]
    elif arr[2] == arr[5] == arr[8] and arr[2] != 0:
        return arr[2]

    elif arr[0] == arr[4] == arr[8] and arr[0] != 0:
        return arr[0]
    elif arr[2] == arr[4] == arr[6] and arr[2] != 0:
        return arr[2]

    else:
        return 0


#prints board, uses array to print 1s or -1s
def printboard(arr):
    print('{:2}'.format(arr[0]), " | ", '{:2}'.format(arr[1]), " | ", '{:2}'.format(arr[2]), "\n",
          '{:2}'.format(arr[3]), " | ", '{:2}'.format(arr[4]), " | ", '{:2}'.format(arr[5]), "\n",
          '{:2}'.format(arr[6]), " | ", '{:2}'.format(arr[7]), " | ", '{:2}'.format(arr[8]), "\n", sep='')


def loadBoard():
    # array that represents board
    board = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    # determines who's turn it is
    turn = input("0 to go first, 1 to go second:\n")
    #determines which regresser to play against
    regressor = input("Enter KNN/linear/MLP")

    while not get_state(board):
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
            if(regressor == "KNN"):

            elif(regressor == "linear"):
            
            elif(regressor == "MLP"):
            
            turn = "0"

        printboard(board)

    print("game over!")


def play():
    board = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    printboard(board)
    loadBoard()

if __name__ == '__main__':
    play()