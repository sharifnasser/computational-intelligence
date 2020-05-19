import numpy as np

# initialize knapsack problem
number_of_objects = 10 # number of objects available
max_weight = 165 # maximum weight possible
weights = np.array([23, 31, 29, 44, 53, 38, 63, 85, 89, 82]) # weights of objects
gains = np.array([92, 57, 49, 68, 60, 43, 67, 84, 87, 72]) # gains of objects

def evaluate(particle):
    """ Return the sum of gains of the selected objects if the sum 
    of weights does not exceed the maximum weight, otherwise return 0 """
    if sum(particle * weights) <= max_weight:
        return sum(particle * gains),
    return 0