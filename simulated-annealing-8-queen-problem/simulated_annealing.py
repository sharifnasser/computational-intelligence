board_size = 8

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

# Initialize parameters

cadena_intentos_aceptados = 40
cadena_intentos_maximo = 80
cadena_maximo_sin_mejora = 1
frecuencia_impresion = 5
alfa = 0.80
beta = 1.2
minimo_razon_aceptados = 0.90

estado_inicial = queens = [[i, random.randrange(board_size)] for i in range(board_size)]

intentos = 1
u = estado_inicial
f = count_attacking_queens(estado_inicial)

mejor = u
mejor_intentos = intentos

c = 0.1