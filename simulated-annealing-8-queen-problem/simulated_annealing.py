import random
from math import exp

board_size = 8

# Initialize parameters
cadena_intentos_aceptados = 40
cadena_intentos_maximo = 80
cadena_maximo_sin_mejora = 1
frecuencia_impresion = 5
alfa = 0.80
beta = 1.2
minimo_razon_aceptados = 0.90

c = 0.1

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

def crear_vecino(queens):
        row_selection = random.randrange(board_size)
        queen_selection = queens[row_selection]
        new_column = random.choice(list(range(0, queen_selection[1])) + list(range(queen_selection[1]+1, board_size)))
        
        neighbor = queens
        neighbor[row_selection][1] = new_column
        return neighbor

def acepta_intento(u, v):
        evaluacion_v = count_attacking_queens(v)
        evaluacion_u = count_attacking_queens(u)
        if evaluacion_v < count_attacking_queens(u):
                return True
        else:
                return (random.random() < exp( -1 * (evaluacion_v - evaluacion_u) / c ))
                

def cadena_markov(u):
        global c
        intentos = 0
        intentos_aceptados = 0
        print(u)
        while (intentos_aceptados < cadena_intentos_aceptados) and (intentos < cadena_intentos_maximo):
                v = crear_vecino(u)
                evaluacion_v = count_attacking_queens(v)

                intentos += 1
                if acepta_intento(u, v):
                        c = c * (evaluacion_v / evaluacion_u)

                        u = v
                        intentos_aceptados += 1

        razon_aceptacion = 1.0 * (intentos_aceptados / intentos)
        u_final = u

        return u_final, razon_aceptacion

def calcular_temperatura_inicial(u):
        global c
        razon_aceptacion = 0
        while razon_aceptacion < minimo_razon_aceptados:
                [u, razon_aceptacion] = cadena_markov(u)
                c = c * beta

        u_inicial = u

        return u_inicial

def recocido(u):
        global c
        cadena_sin_mejora = 0
        anterior = u
        while (cadena_sin_mejora < cadena_maximo_sin_mejora):
                u, razon_aceptacion = cadena_markov(u)
                evaluacion_anterior = count_attacking_queens(anterior)
                evaluacion_u = count_attacking_queens(u)

                if evaluacion_u >= evaluacion_anterior:
                        cadena_sin_mejora += 1
                else:
                        cadena_sin_mejora = 0

                anterior = u
                c = c * alfa

        return u

estado_inicial = queens = [[i, random.randrange(board_size)] for i in range(board_size)]

intentos = 1
u = estado_inicial
evaluacion_u = count_attacking_queens(estado_inicial)

mejor = u
mejor_intentos = intentos

u = calcular_temperatura_inicial(estado_inicial)
print('temp incial', c)

print(recocido(u))
print(count_attacking_queens(u))