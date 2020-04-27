import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import exp

# initialize algorithm parameters
max_velocity = 1 # maximum velocity for particles
alpha = 2 # cognitive  scaling
beta = 2 # social scaling
swarm_size = 1 # number of particles in the swarm
max_runs_no_improve = 100 # maximum number of runs without improvement

# initialize knapsack problem
number_of_objects = 10 # number of objects available
max_weight = 165 # maximum weight possible
weights = np.array([23, 31, 29, 44, 53, 38, 63, 85, 89, 82]) # weights of objects
gains = np.array([92, 57, 49, 68, 60, 43, 67, 84, 87, 72]) # gains of objects

# initialize algorithm evaluation
len_algorithm_evaluation = 10 # number of algorithm evaluations
best_found_algorithm_evaluation = [] # best found list for each algorithm evaluation

def evaluate_combination(particle):
    """ Return the sum of gains of the selected objects if the sum 
    of weights does not exceed the maximum weight, otherwise return 0 """
    if sum(particle * weights) <= max_weight:
        return sum(particle * gains)
    return 0

def sigmoid(x):
    """ Return evaluation of sigmoid function for x """
    return 1 / (1 + exp(-x))

def limit_velocity(velocity):
    """ Returns +/- velocity if it does not exceed +/- maximum velocity,
    otherwise returns +/- maximum velocity """
    return max(min(max_velocity, velocity), -max_velocity)

for algorithm_evaluation in range(len_algorithm_evaluation):
    swarm = np.array([np.zeros(number_of_objects, dtype=int) for i in range(swarm_size)]) # list of particles initialized in 0s
    particle_best = np.array([np.zeros(number_of_objects, dtype=int) for i in range(swarm_size)]) # initialize array of best local of each particle
    swarm_best = np.zeros(number_of_objects, dtype=int) # initialize best particle in the swarm
    velocity = np.array([np.zeros(number_of_objects, dtype=int) for i in range(swarm_size)]) # initialize array with velocities of particles to zero

    start_time = time.time()
    runs_no_improve = 0

    best_found_list = [] # save best combination found by number of particles movements
    while runs_no_improve < max_runs_no_improve:
        for particle in range(swarm_size):
            for bit in range(number_of_objects):
                # calculate velocity for each bit of every particle
                velocity[particle][bit] = velocity[particle][bit] \
                                        + alpha * np.random.uniform(0, 1) * (particle_best[particle][bit] - swarm[particle][bit]) \
                                        + beta * np.random.uniform(0, 1) * (swarm_best[bit] - swarm[particle][bit])

                velocity[particle][bit] = limit_velocity(velocity[particle][bit]) # limit the velocity between +/- maximum velocity
                
                swarm[particle][bit] = 1 if np.random.uniform(0, 1) < sigmoid(velocity[particle][bit]) else 0 # select value of bit with sigmoid function

            best_found_list.append(evaluate_combination(swarm_best)) # saved best found evaluation for every particle changed

            # save particle as its best local if it has a better evaluation
            if evaluate_combination(swarm[particle]) > evaluate_combination(particle_best[particle]):
                particle_best[particle] = swarm[particle].copy()

            # save best local of particle as the best global if it has a better evaluation
            if evaluate_combination(particle_best[particle]) > evaluate_combination(swarm_best):
                swarm_best = particle_best[particle].copy()
            else:
                runs_no_improve += 1 # count movements with no improvement for evaluation
                
            print("local " + str(particle) + ":", particle_best[particle])
        print("best:", swarm_best)
    print("Final combination:", swarm_best)
    print("Final evaluation:", evaluate_combination(swarm_best)) # calculate execution time

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