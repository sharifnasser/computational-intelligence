import random
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from math import exp

swarm_size = 1
swarm = np.array([np.random.randint(2, size=10) for i in range(swarm_size)])

print(swarm)