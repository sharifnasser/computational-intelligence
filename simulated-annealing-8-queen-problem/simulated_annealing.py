import random
import time
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from math import exp

def count_crossing_queens(positions):
        """ Count queens being threatened directly or indirectly """
        crossing_values = 0

        for position in positions:
                crossing_values += positions.count(position) - 1 # count number of duplicate positions

        return crossing_values

def count_attacking_queens(queens):
        """ Count number of threats directly or indirectly, either in columns, diagonals and reverse diagonals """
        columns = []
        diagonals = []
        reverse_diagonals = []
        for row, column in enumerate(queens):
                columns.append(column) # column position
                diagonals.append(column - row) # diagonal position
                reverse_diagonals.append(column + row - (board_size + 1)) # reverse diagonal position

        # sum number of threats in columns, diagonals and reverse_diagonals
        attacks = count_crossing_queens(columns) + count_crossing_queens(diagonals) + count_crossing_queens(reverse_diagonals)
        
        return attacks

def generate_neighbor(queens):
        """ Generate a neighbor board from a previous board """
        neighbor = deepcopy(queens) # copy queens in neighbor
        
        row_selection = random.randrange(board_size) # choose a random row
        queen_selection = neighbor[row_selection] # choose the queen in that row
        new_column = random.choice(list(range(0, queen_selection)) + \
                                   list(range(queen_selection+1, board_size))) # generate a random column different from original
        
        neighbor[row_selection] = new_column # move selected queen to new column to generate neighbor

        return neighbor

def print_board(queens):
        for queen in queens:
                for column in range(board_size):
                        if column == queen: print('Q ', end='')
                        else: print('. ', end='')
                print()

def accept_state(evaluation_old, evaluation_new):
        """ Indicate if the new state must be accepted to minimize the cost function """
        if evaluation_new < evaluation_old:
                return True # accept if new state minimizes cost function
        else:
                return ( random.random() < exp( - (evaluation_new - evaluation_old) / temperature )) # apply metropolis algorithm                

def markov_chain(state):
        """ Run markov chain with old state. Return newest state and acceptance rate of attempts """
        global best_found_list
        attempts = 0
        accepted_attempts = 0

        while (accepted_attempts < len_accepted_attempts_markov) and (attempts < len_attempts_markov):
                new_state = generate_neighbor(state) # generate neighbor
                evaluation_old = count_attacking_queens(state) # evaluate old state with cost function
                evaluation_new = count_attacking_queens(new_state) # evaluate new state with cost function
                attempts += 1 # count attempts

                if accept_state(evaluation_old, evaluation_new):
                        state = new_state # change state
                        accepted_attempts += 1 # count accepted attempts
                        # temperature = temperature * (evaluation_new / evaluation_old) # algorithm improvement

                best_found_list.append(count_attacking_queens(state))

        acceptance_rate = 1.0 * (accepted_attempts / attempts) # calculate attempts acceptance rate

        return state, acceptance_rate

def init_temperature(state):
        """ Initialize temperature according to miminum acceptance rate """
        global temperature

        _, acceptance_rate = markov_chain(state) # get initial acceptance rate

        while acceptance_rate < min_acceptance_rate:
                temperature = temperature * beta # increase temperature
                _, acceptance_rate = markov_chain(state) # get acceptance rate with current temperature

def simulated_annealing(state):
        """ Run simulated annealing algorithm """
        global temperature
        chains_no_improve = 0

        while (chains_no_improve < max_chains_no_improve):
                new_state, _ = markov_chain(state) # run markov chain
                evaluation_old = count_attacking_queens(state) # evaluate old state with cost function
                evaluation_new = count_attacking_queens(new_state) # evaluate new state with cost function

                # count consecutive markvov chains without improvement in cost function
                if evaluation_new >= evaluation_old:
                        chains_no_improve += 1
                else:
                        chains_no_improve = 0

                state = new_state # update state
                temperature = temperature * alfa # decrease temperature

        return state

# Initialize parameters
board_size = 8
len_accepted_attempts_markov = 40 # maximum markov chain length in accepted attempts
len_attempts_markov = 80 # maximum markov chain length
max_chains_no_improve = 20 # maximum number of markov chains without improvement
alfa = 0.80
beta = 1.2
min_acceptance_rate = 0.90
temperature = 0.1
best_found_list = []
len_algorithm_evaluation = 10
best_found_algorithm_evaluation = []

for algorithm_evaluation in range(len_algorithm_evaluation):
        start = time.time()

        original_board = [random.randrange(board_size) for i in range(board_size)] # initialize queens positions

        init_temperature(original_board)

        print('Initial temperature:', temperature)

        best_found_list = []
        final_board = simulated_annealing(original_board)
        final_evaluation = count_attacking_queens(final_board)

        print_board(final_board)
        print(final_evaluation)

        print('time = ', time.time() - start)

        best_found_algorithm_evaluation.append(best_found_list.copy())

best_founds = pd.DataFrame.from_records(best_found_algorithm_evaluation).transpose()
best_founds['mean'] = best_founds.mean(axis=1)
best_founds['std'] = best_founds.std(axis=1)
best_founds['mean+std'] = best_founds['mean'] + best_founds['std']
best_founds['mean-std'] = best_founds['mean'] - best_founds['std']

plt.plot(best_founds['mean'])
plt.plot(best_founds['mean+std'], linestyle='dashed', alpha=0.5)
plt.plot(best_founds['mean-std'], linestyle='dashed', alpha=0.5)

plt.show()