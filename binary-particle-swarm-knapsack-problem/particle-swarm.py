import random
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from math import exp

number_of_objects = 10
swarm_size = 10
max_weight = 165
max_velocity = 1
alpha = 2
beta = 2
r1 = np.random.uniform(0, 1)
r2 = np.random.uniform(0, 1)
swarm = np.array([np.random.randint(2, size=number_of_objects) for i in range(swarm_size)])
particle_best = np.array([np.zeros(number_of_objects, dtype=int) for i in range(swarm_size)])
swarm_best = np.zeros(number_of_objects, dtype=int)
velocity = np.array([np.zeros(number_of_objects, dtype=int) for i in range(swarm_size)])
weights = np.array([23, 31, 29, 44, 53, 38, 63, 85, 89, 82])
gains = np.array([92, 57, 49, 68, 60, 43, 67, 84, 87, 72])

max_runs_no_improve = 100 # maximum number of markov chains without improvement
best_found_list = [] # save best state found by number of neighbors created or attempts to move a queen
len_algorithm_evaluation = 10 # number of algorithm evaluations
best_found_algorithm_evaluation = [] # best found list for each algorithm evaluation

def evaluate_combination(particle):
    if sum(particle * weights) <= max_weight:
        return sum(particle * gains)
    return 0

def sigmoid(x):
    return 1 / (1 + exp(-x))

def limit_velocity(velocity):
    return max(min(max_velocity, velocity), -max_velocity)

for algorithm_evaluation in range(len_algorithm_evaluation):
    start_time = time.time()

    runs_no_improve = 0
    while runs_no_improve < max_runs_no_improve:
        for particle in range(swarm_size):
            for bit in range(number_of_objects):
                velocity[particle][bit] = velocity[particle][bit] + alpha * r1 * (particle_best[particle][bit] - swarm[particle][bit]) + beta * r2 * (swarm_best[bit] - swarm[particle][bit])
                velocity[particle][bit] = limit_velocity(velocity[particle][bit])
                swarm[particle][bit] = 1 if np.random.uniform(0, 1) < sigmoid(velocity[particle][bit]) else 0

            best_found_list.append(evaluate_combination(swarm_best))

            if evaluate_combination(swarm[particle]) > evaluate_combination(particle_best[particle]):
                particle_best[particle] = swarm[particle].copy()

            if evaluate_combination(particle_best[particle]) > evaluate_combination(swarm_best):
                swarm_best = particle_best[particle].copy()
            else:
                runs_no_improve += 1
                
            print("local " + str(particle) + ":", particle_best[particle])
        print("best:", swarm_best)
    print("Final combination:", swarm_best)
    print("Final evaluation:", evaluate_combination(swarm_best))

    print('Execution Time: ', time.time() - start_time) # calculate execution time

    best_found_algorithm_evaluation.append(best_found_list.copy())

# Display results in a pandas DataFrame and a line plot
best_founds = pd.DataFrame.from_records(best_found_algorithm_evaluation).transpose().ffill(axis=0)
best_founds['mean'] = best_founds.mean(axis=1)
best_founds['std'] = best_founds.std(axis=1)
best_founds['mean+std'] = best_founds['mean'] + best_founds['std']
best_founds['mean-std'] = best_founds['mean'] - best_founds['std']
print(best_founds)

plt.title('Best Found Curve')
plt.ylabel('Best Found')
plt.xlabel('# of Particles Movements')
plt.grid(True)
plt.plot(best_founds['mean'])
plt.plot(best_founds['mean+std'], linestyle='dashed', alpha=0.5)
plt.plot(best_founds['mean-std'], linestyle='dashed', alpha=0.5)
plt.show()