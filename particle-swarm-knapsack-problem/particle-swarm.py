import random
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from math import exp

number_of_objects = 10
swarm_size = 100
max_weight = 165
max_velocity = 4
alpha = 2
beta = 2
r1 = np.random.uniform(0, 1)
r2 = np.random.uniform(0, 1)
swarm = np.array([np.random.randint(2, size=number_of_objects) for i in range(swarm_size)])
particle_best = np.array([np.zeros(number_of_objects, dtype=int) for i in range(swarm_size)])
swarm_best = np.zeros(number_of_objects, dtype=int)
velocity = np.zeros(number_of_objects, dtype=int)
weights = np.array([23, 31, 29, 44, 53, 38, 63, 85, 89, 82])
gains = np.array([92, 57, 49, 68, 60, 43, 67, 84, 87, 72])

def evaluate_solution(particle):
    if sum(particle * weights) <= max_weight:
        return sum(particle * gains)
    return 0

def sigmoid(x):
    return 1 / (1 + exp(-x))

def limit_velocity(velocity):
    return max(min(max_velocity, velocity), -max_velocity)

while True:
    for particle in range(swarm_size):
        for bit in range(number_of_objects):
            velocity[bit] = velocity[bit] + alpha * r1 * (particle_best[particle][bit] - swarm[particle][bit]) + beta * r2 * (swarm_best[bit] - swarm[particle][bit])
            velocity[bit] = limit_velocity(velocity[bit])
            swarm[particle][bit] = 1 if np.random.uniform(0, 1) < sigmoid(velocity[bit]) else 0
        if evaluate_solution(swarm[particle]) > evaluate_solution(particle_best[particle]):
            particle_best[particle] = swarm[particle]   
    if evaluate_solution(swarm[particle]) > evaluate_solution(swarm_best):
        swarm_best = swarm[particle]
    print(swarm_best)