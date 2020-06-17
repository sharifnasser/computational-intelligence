# Genetic Algorithm for solving Traveling Salesman Problems
# The problem being used was found in: http://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsp/
# The problem is named bayg29.tsp and the solution is named bayg29.opt.tour

import csv
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from deap import algorithms, base, creator, tools

def load_data():
    """ Reads the desired TSP files, extracts the city coordinates from problem.tsp, calculates and saves the distances
        between every two cities in a distance matrix (two-dimensional ndarray). Also, extracts the known optimal solution
        from solution.tsp and saves it in a list.
        :return: list with x,y coordinates of the cities (locations)
        :return: distance matrix of the cities (distances)
        :return: number of cities (number_of_cities)
        :return: known optimal solution (optimal_solution)
    """
    with open("problem.tsp") as f:
        reader = csv.reader(f, delimiter=" ", skipinitialspace=True)

        locations = []
        # read data lines until 'EOF' found:
        for row in reader:
            if row[0] != 'EOF':
                # remove index at beginning of line:
                del row[0]

                # convert x,y coordinates to ndarray:
                locations.append(np.asarray(row, dtype=np.float32))
            else:
                break
        # calculate number of cities
        number_of_cities = len(locations)

        # initialize the distance matrix with 0s:
        distances = np.zeros((number_of_cities, number_of_cities))

        # fill the distance matrix with calculated distances:
        for i in range(number_of_cities):
            for j in range(i + 1, number_of_cities):
                # calculate euclidean distance between two ndarrays:
                distance = np.linalg.norm(locations[j] - locations[i])
                # distance from city i to city j is the same for city j to city i
                distances[i, j] = distance
                distances[j, i] = distance
                # print calculated distance
                print("{}, {}: location1 = {}, location2 = {} => distance = {}".format(i, j, locations[i], locations[j], distance))
    
    with open("solution.tsp") as f:
        reader = csv.reader(f, delimiter=" ", skipinitialspace=True)

        optimal_solution = []

        # read solution file until the -1 before the EOF is found:
        for row in reader:
            if row[0] != '-1':
                # save optimal solution in in list:
                optimal_solution.append(int(row[0])-1)
            else:
                break

    return locations, distances, number_of_cities, optimal_solution

def get_total_distance(individual):
    """Calculates the total distance of the path described by the given individual of cities
    :param individual: list of ordered city indices describing the given path.
    :return: total distance of the path described by the given indices
    """
    # distance between th elast and first city:
    distance = distances[individual[-1], individual[0]]

    # add the distance between each pair of consequtive cities:
    for i in range(len(individual) - 1):
        distance += distances[individual[i], individual[i + 1]]

    return distance,

def plot_individual(individual):
    """plots the path described by the given indices of the cities
    :param individual: list of ordered city indices describing the given path.
    :return: resulting plot
    """

    # plot the dots representing the cities:
    plt.scatter(*zip(*locations), marker='.', color='red')

    # create a list of the corresponding city locations:
    locs = [locations[i] for i in individual]
    locs.append(locs[0])

    # plot a line between each pair of consequtive cities:
    plt.plot(*zip(*locs), linestyle='-', color='blue')

    return plt

# set the random seed for repeatable results
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# create the desired traveling salesman problem instace:
locations, distances, number_of_cities, optimal_solution = load_data()

# Genetic Algorithm constants:
POPULATION_SIZE = 1000
MAX_GENERATIONS = 300
HALL_OF_FAME_SIZE = 30
P_CROSSOVER = 0.5  # probability for crossover
P_MUTATION = 0.5  # probability for mutating an individual

toolbox = base.Toolbox()
# define a single objective, minimizing fitness strategy:
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# create the Individual class based on list of integers:
creator.create("Individual", list, typecode='i', fitness=creator.FitnessMin)
# define how the element of an individual is determined
# create an operator that generates randomly shuffled indices:
toolbox.register("attribute", random.sample, range(number_of_cities), number_of_cities)
# create the individual creation operator to fill up an Individual instance with shuffled indices:
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attribute)
# create the population creation operator to generate a list of individuals:
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# fitness calculation - compute the total distance of the list of cities represented by individual:
toolbox.register("evaluate", get_total_distance)

# Genetic operators:
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1.0/number_of_cities)

# create initial population (generation 0):
population = toolbox.population(n=POPULATION_SIZE)

# determine statistics to be calulated:
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", np.min)
stats.register("avg", np.mean)
stats.register("max", np.max)
stats.register("std", np.std)

# define the hall-of-fame:
hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

# run eaSimple algorithm and save stats in logbook
population, logbook = algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                            ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

# print best individual info:
best = hof.items[0]
print("Best Ever Individual = ", best)
print("Best Ever Fitness = ", best.fitness.values[0])

# print optimal solution info:
print("Known Optimal Solution = ", optimal_solution)
print("Optimal Distance = ", get_total_distance(optimal_solution)[0])

# plot best solution:
plt.figure(1)
plt.title('Best Found Solution')
plot_individual(best)

# plot optimal solution
plt.figure(2)
plt.title('Optimal Solution')
plot_individual(optimal_solution)

# plot statistics:
max_stats, min_stats, avg_stats, std_stats = logbook.select("max", "min", "avg", "std")
plt.figure(3)
sns.set_style("whitegrid")
plt.plot(min_stats, color='green')
plt.xlabel('Generation')
plt.ylabel('Best Found')
plt.title('Best Found over Generations')

# show both plots:
plt.show()

