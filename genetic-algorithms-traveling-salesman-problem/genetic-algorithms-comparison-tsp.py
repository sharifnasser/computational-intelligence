# Genetic Algorithm for solving Traveling Salesman Problems
# The problem being used was found in: http://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsp/
# The problem is named bayg29.tsp and the solution is named bayg29.opt.tour

import csv
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
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

# create the desired traveling salesman problem instace:
locations, distances, number_of_cities, optimal_solution = load_data()

# Genetic Algorithm constants:
# define number of iterations for each size of population
NUM_ITERATIONS = 2
# initialize list of sizes of populations to run
POPULATION_SIZES = [250, 500, 1000]
MAX_GENERATIONS = 300
HALL_OF_FAME_SIZE = 10

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

# determine statistics to be calulated:
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", np.min)
stats.register("avg", np.mean)
stats.register("max", np.max)
stats.register("std", np.std)

# define the hall-of-fame:
hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

# initialize a DataFrame in blank
stats_df = pd.DataFrame(data=None, columns=["Algorithm", "Iteration", "Population", "Generation", "BestFound"])

for iteration in range(NUM_ITERATIONS):
    for population_size in POPULATION_SIZES:

        # create initial population
        population = toolbox.population(n=population_size)

        # run eaSimple algorithm and save stats logbook in log_simple
        # deepcopy() is use to create a copy of the population
        _, log_simple = algorithms.eaSimple(population=deepcopy(population), toolbox=toolbox, halloffame=hof, cxpb=1.0, mutpb=1.0, 
                                            ngen=MAX_GENERATIONS, stats=stats, verbose=True)
        # print best individual info:
        best = hof.items[0]
        print("eaSimple Solution Best Ever Individual = ", best)
        print("eaSimple Solution Best Ever Fitness = ", best.fitness.values[0])

        # plot best solution:
        plt.figure(1)
        plt.title('eaSimple Solution')
        plot_individual(best)
        
        # run eaMuPlusLambda algorithm and save stats logbook in log_mupluslambda
        _, log_mupluslambda = algorithms.eaMuPlusLambda(population=deepcopy(population), toolbox=toolbox, mu=population_size, lambda_=population_size, 
                                                        halloffame=hof, cxpb=0.5, mutpb=0.5, ngen=MAX_GENERATIONS, stats=stats, verbose=True)
        # print best individual info:
        best = hof.items[0]
        print("eaMuPlusLambda Best Ever Individual = ", best)
        print("eaMuPlusLambda Best Ever Fitness = ", best.fitness.values[0])

        # plot best solution:
        plt.figure(2)
        plt.title('eaMuPlusLambda Solution')
        plot_individual(best)
        
        # run eaMuCommaLambda algorithm and save stats logbook in log_mucommalambda
        _, log_mucommalambda = algorithms.eaMuCommaLambda(population=deepcopy(population), toolbox=toolbox, mu=population_size, lambda_=population_size,
                                                        halloffame=hof, cxpb=0.5, mutpb=0.5, ngen=MAX_GENERATIONS, stats=stats, verbose=True)
        # print best individual info:
        best = hof.items[0]
        print("eaMuCommaLambda Best Ever Individual = ", best)
        print("eaMuCommaLambda Best Ever Fitness = ", best.fitness.values[0])

        # plot best solution:
        plt.figure(3)
        plt.title('eaMuCommaLambda Solution')
        plot_individual(best)

        # create column of Algorithm for DataFrame (hard way)
        algorithm_column = ["eaSimple"]*(MAX_GENERATIONS+1) + ["eaMuPlusLambda"]*(MAX_GENERATIONS+1) + ["eaMuCommaLambda"]*(MAX_GENERATIONS+1)
        # create column of Iteration for DataFrame (hard way)
        iteration_column = [iteration]*(MAX_GENERATIONS+1)*3
        # create column of Population sizes for DataFrame
        population_size_column = log_simple.select("nevals") + log_mupluslambda.select("nevals") + log_mucommalambda.select("nevals")
        # create column of Generation for DataFrame
        generation_column = log_simple.select("gen") + log_mupluslambda.select("gen") + log_mucommalambda.select("gen")
        # create column of Best Found for DataFrame
        best_found_column = log_simple.select("max") + log_mupluslambda.select("max") + log_mucommalambda.select("max")

        # Append all columns in the DataFrame 
        stats_df = stats_df.append(pd.DataFrame.from_dict({"Algorithm": algorithm_column, "Iteration": iteration_column, "Population": population_size_column,
                                "Generation": generation_column, "BestFound": best_found_column}),
                            ignore_index=True)

# print optimal solution info:
print("Known Optimal Solution = ", optimal_solution)
print("Optimal Distance = ", get_total_distance(optimal_solution))
plt.figure(4)
plt.title('Optimal Solution')
plot_individual(optimal_solution)

# Calculate average of Best Found by Population size, and then its average by Iteration
# The result is a DataFrame grouped by Algorithm and with the mean of the Best Found by Generation
stats_df = (stats_df.groupby(["Algorithm", "Population", "Generation"]).mean()
            .groupby(["Algorithm", "Generation"]).agg(['mean', 'std']))

# Remove multiindex columns
stats_df.columns = ['_'.join(col) for col in stats_df.columns.values]

# Remove multiindex
stats_df = stats_df.reset_index()

# Calculate mean +/- std
stats_df["BestFound_mean-std"] = stats_df["BestFound_mean"] - stats_df["BestFound_std"]
stats_df["BestFound_mean+std"] = stats_df["BestFound_mean"] + stats_df["BestFound_std"]

# Print DataFrame
print(stats_df)

# Plot Best Found Curve for each Algorithm, including mean +/- std as dashed curves
generations = list(range(MAX_GENERATIONS+1))
plt.figure(5)
plt.title('Best Found Curve')
plt.ylabel('Best Found')
plt.xlabel('# Generations')
plt.xlim((0, MAX_GENERATIONS))
plt.grid(True)
plt.plot(generations, stats_df[stats_df["Algorithm"] == "eaSimple"]["BestFound_mean"], color='b')
plt.plot(generations, stats_df[stats_df["Algorithm"] == "eaMuPlusLambda"]["BestFound_mean"], color='g')
plt.plot(generations, stats_df[stats_df["Algorithm"] == "eaMuCommaLambda"]["BestFound_mean"], color='r')
plt.legend(labels=["eaSimple", "eaMuPlusLambda", "eaMuCommaLambda"])

plt.plot(generations, stats_df[stats_df["Algorithm"] == "eaSimple"]["BestFound_mean-std"], color='b', linestyle='dashed', alpha=0.5)
plt.plot(generations, stats_df[stats_df["Algorithm"] == "eaMuPlusLambda"]["BestFound_mean-std"], color='g', linestyle='dashed', alpha=0.5)
plt.plot(generations, stats_df[stats_df["Algorithm"] == "eaMuCommaLambda"]["BestFound_mean-std"], color='r', linestyle='dashed', alpha=0.5)

plt.plot(generations, stats_df[stats_df["Algorithm"] == "eaSimple"]["BestFound_mean+std"], color='b', linestyle='dashed', alpha=0.5)
plt.plot(generations, stats_df[stats_df["Algorithm"] == "eaMuPlusLambda"]["BestFound_mean+std"], color='g', linestyle='dashed', alpha=0.5)
plt.plot(generations, stats_df[stats_df["Algorithm"] == "eaMuCommaLambda"]["BestFound_mean+std"], color='r', linestyle='dashed', alpha=0.5)
plt.show()


# run eaSimple algorithm and save stats in logbook
#population, logbook = algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
#                                            ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

# plot statistics:
#max_stats, min_stats, avg_stats, std_stats = logbook.select("max", "min", "avg", "std")
#plt.figure(2)
#sns.set_style("whitegrid")
#plt.plot(min_stats, color='red')
#plt.plot(avg_stats, color='green')
#plt.xlabel('Generation')
#plt.ylabel('Best Found')
#plt.title('Best Found over Generations')
#
## show both plots:
#plt.show()

