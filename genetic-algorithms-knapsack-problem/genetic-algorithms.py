from deap import base, creator
from deap import algorithms
from deap import tools
from copy import deepcopy
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# initialize knapsack problem
number_of_objects = 10 # number of objects available
max_weight = 165 # maximum weight possible
weights = np.array([23, 31, 29, 44, 53, 38, 63, 85, 89, 82]) # weights of objects
gains = np.array([92, 57, 49, 68, 60, 43, 67, 84, 87, 72]) # gains of objects

def evaluate(individual):
    """ Return the sum of gains of the selected objects if the sum 
    of weights does not exceed the maximum weight """
    return sum(individual * gains),

def feasible(individual):
    """ Return True the sum 
    of weights does not exceed the maximum weight """
    return (sum(individual * weights) <= max_weight)

def distance(individual):
    """ Distance Function for DeltaPenalty """
    return sum(individual * gains)

# define as a maximization problem
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# define individuals as lists to be maximized
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
# determine selection method
toolbox.register("select", tools.selStochasticUniversalSampling)
# select mutation method
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
# select reproduction method
toolbox.register("mate", tools.cxTwoPoint)
# determine evaluation function
toolbox.register("evaluate", evaluate)
# determine pentalty
toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, 0, distance))
# define how the element of an individual is determined:
# randomly chosen 0s and 1s
toolbox.register("attribute", random.randint, a=0, b=1) 
# define how an individual is created:
# a list of n elements
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=10)
# define how a population is created
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# determine statistics to be calulated
stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("max", np.max)
stats.register("min", np.min)
stats.register("avg", np.mean)
stats.register("std", np.std)

# define Hall of Fame of best 3 individuals
hof = tools.HallOfFame(3)

# define number of generations
num_generations = 100

# initialize a DataFrame in blank
stats_df = pd.DataFrame(data=None, columns=["Algorithm", "Iteration", "Population", "Generation", "BestFound"])
# initialize list of sizes of populations to run
population_sizes = [25, 50, 100]
# define number of iterations for each size of population
num_iterations = 10

for iteration in range(num_iterations):
    for population_size in population_sizes:

        # create a population
        population = toolbox.population(n=population_size)

        # run eaSimple algorithm and save stats logbook in log_simple
        # deepcopy() is use to create a copy of the population
        _, log_simple = algorithms.eaSimple(population=deepcopy(population), toolbox=toolbox, halloffame=hof, cxpb=1.0, mutpb=1.0, 
                                            ngen=num_generations, stats=stats, verbose=True)
        # hall of fame is printed for each run of algorithm
        print(hof)
        
        # run eaMuPlusLambda algorithm and save stats logbook in log_mupluslambda
        _, log_mupluslambda = algorithms.eaMuPlusLambda(population=deepcopy(population), toolbox=toolbox, mu=population_size, lambda_=population_size, 
                                                        halloffame=hof, cxpb=0.5, mutpb=0.5, ngen=num_generations, stats=stats, verbose=True)
        print(hof)
        
        # run eaMuCommaLambda algorithm and save stats logbook in log_mucommalambda
        _, log_mucommalambda = algorithms.eaMuCommaLambda(population=deepcopy(population), toolbox=toolbox, mu=population_size, lambda_=population_size,
                                                        halloffame=hof, cxpb=0.5, mutpb=0.5, ngen=num_generations, stats=stats, verbose=True)
        print(hof)

        # create column of Algorithm for DataFrame (hard way)
        algorithm_column = ["eaSimple"]*(num_generations+1) + ["eaMuPlusLambda"]*(num_generations+1) + ["eaMuCommaLambda"]*(num_generations+1)
        # create column of Iteration for DataFrame (hard way)
        iteration_column = [iteration]*(num_generations+1)*3
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
generations = list(range(num_generations+1))
plt.title('Best Found Curve')
plt.ylabel('Best Found')
plt.xlabel('# Generations')
plt.xlim((0, num_generations))
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