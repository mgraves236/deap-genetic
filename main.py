import math
import random
from deap import algorithms, base, creator, tools
import RandomNumberGenerator

rng = RandomNumberGenerator.RandomNumberGenerator(5546568)


# Function to generate instance
def generate_instance(n: int, m: int):
    matrix = [[0 for x in range(m)] for y in range(n)]
    A = 0
    for i in range(0, n):
        for j in range(0, m):
            matrix[i][j] = rng.nextInt(1, 99)
            A = A + matrix[i][j]
    B = math.floor(1 / 2 * A)
    A = math.floor(1 / 6 * A)

    matrix2 = [0 for x in range(n)]
    for i in range(0, n):
        matrix2[i] = rng.nextInt(A, B)
    return (matrix, matrix2)

# Function to evaluate individuals
def evaluate(individual):
    a = sum(individual)
    return (a,)

# def evaluate(individual):
#     objective1 = calculate_objective1(individual)
#     objective2 = calculate_objective2(individual)
#
#     return objective1, objective2

def genetic_algorithm():
    # main algorithm
    population = toolbox.population(n=10)
    # algorithms.eaSimple(population, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN)
    fitnesses = map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        print(ind)
        ind.fitness.values = fit
    for g in range(NGEN):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        population[:] = offspring

    fitnesses = map(toolbox.evaluate, population)
    return (population, fitnesses)


m = 3
n = 5
p_ij, d_j = generate_instance(n, m)
# print(p_ij)
# print(d_j)
# algorithm parameters
CXPB, MUTPB, NGEN = 0.5, 0.2, 40  # probability of performing a crssover, muatatiom probability, number of genrations
IND_SIZE = n
# a minimizing fitness is built using negatives weights
# create a fitness function with a few objectives
# creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
# creator.create("Individual", list, fitness=creator.FitnessMulti)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(IND_SIZE), IND_SIZE)
toolbox.register("individual", tools.initIterate, creator.Individual,
                 toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1) # indpb probability for each attribute to be exchanged to another position
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

result, res = genetic_algorithm()
print(result)
print(list(res))
