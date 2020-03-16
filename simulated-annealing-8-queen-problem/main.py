from random import randrange, sample

board_size = 8

queens = [[i, randrange(board_size)] for i in range(board_size)]

print(queens)