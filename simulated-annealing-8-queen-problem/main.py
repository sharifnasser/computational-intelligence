from random import randrange, sample

board_size = 8

queens = [[i, randrange(board_size)] for i in range(board_size)]

def count_crossing_queens(positions):
        crossing_values = 0
        for position in positions:
                crossing_values += positions.count(position) - 1
        return crossing_values

def count_attacking_queens(queens):
        columns = []
        diagonals = []
        reverse_diagonals = []
        for queen in queens:
                columns.append(queen[1])
                diagonals.append(queen[1] - queen[0])
                reverse_diagonals.append(queen[1] + queen[0] - (board_size + 1))

        attacks = count_crossing_queens(columns) + count_crossing_queens(diagonals) + count_crossing_queens(reverse_diagonals)
        
        return attacks

print(queens)

threats = count_attacking_queens(queens)
print(threats)

