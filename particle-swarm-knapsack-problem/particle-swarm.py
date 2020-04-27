import random
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from math import exp

number_of_objects = 10
swarm_size = 3
swarm = np.array([np.random.randint(2, size=number_of_objects) for i in range(swarm_size)])
weights = np.array([23, 31, 29, 44, 53, 38, 63, 85, 89, 82])
gains = np.array([92, 57, 49, 68, 60, 43, 67, 84, 87, 72])

print(swarm)

for particle in swarm:
    print(particle)