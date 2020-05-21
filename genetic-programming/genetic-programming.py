from deap import base, creator, algorithms, tools, gp
import multiprocessing
import operator
import numpy as np
import random

# define evaluation function
def evaluate(individual, inputs, outputs):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function: outputs
    mse = sum([(func(a[0], a[1]) - b)**2 for a, b in zip(inputs, outputs)]) / len(outputs)
    return mse,

# define inputs and outputs (points) for the symbolic regression
inputs = [(0, 10), (1, 9), (2, 8), (3, 7), (4, 6), (5, 5), (6, 4), (7, 3), (8, 2), (9, 1)]
outputs = [90, 82, 74, 66, 58, 50, 42, 34, 26, 18]

pset = gp.PrimitiveSet("MAIN", 2)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.sub, 2)
pset.addEphemeralConstant('R', lambda: random.randint(-10,10))
pset.renameArguments(ARG0='x')
pset.renameArguments(ARG1='y')

# define as a minimization problem
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=5)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutNodeReplacement, pset=pset)

toolbox.register("compile", gp.compile, pset=pset)

toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate, inputs=inputs, outputs=outputs)

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register('min', np.min)
stats.register('max', np.max)
stats.register('avg', np.mean)
stats.register('std', np.std)

hof = tools.HallOfFame(5)

if __name__ == '__main__':
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    pop = toolbox.population(n=100)
    pop2, log = algorithms.eaMuPlusLambda(population=pop, mu=len(pop), lambda_=len(pop),
                                          toolbox=toolbox, halloffame=hof,
                                          cxpb=0.5, mutpb=0.1, ngen=100, stats=stats)

    for ind in hof:
        print(ind)