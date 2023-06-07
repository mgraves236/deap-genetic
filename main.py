import math
import random
from deap import algorithms, base, creator, tools
import RandomNumberGenerator
from matplotlib import pyplot as plt

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


def makespan(individual):
    machine_time = [0 for x in range(n)]
    for j in range(0, len(machine_time)):
        current = individual[j]
        machine_time[0] += p_ij[current][0]
        for i in range(1, m):
            if machine_time[i] < machine_time[i - 1]:
                machine_time[i] = (machine_time[i - 1] - machine_time[i]) + machine_time[i] + p_ij[current][i]
            else:
                machine_time[i] += p_ij[current][i]
    return machine_time[m - 1]


def total_flow_time(individual):
    machine_time = [0 for x in range(n)]
    sum = 0
    for j in range(0, len(machine_time)):
        current = individual[j]
        machine_time[0] += p_ij[current][0]
        for i in range(1, m):
            if machine_time[i] < machine_time[i - 1]:
                machine_time[i] = (machine_time[i - 1] - machine_time[i]) + machine_time[i] + p_ij[current][i]
            else:
                machine_time[i] += p_ij[current][i]

            if i == m - 1:
                sum = sum + machine_time[i]
    return sum


def max_tardiness(individual):
    machine_time = [0 for x in range(n)]
    tardiness = [0 for x in range(n)]
    maximum = 0
    for j in range(0, len(machine_time)):
        current = individual[j]
        machine_time[0] += p_ij[current][0]
        for i in range(1, m):
            if machine_time[i] < machine_time[i - 1]:
                machine_time[i] = (machine_time[i - 1] - machine_time[i]) + machine_time[i] + p_ij[current][i]
            else:
                machine_time[i] += p_ij[current][i]

            if i == m - 1:
                if machine_time[i] - d_j[j] > 0:
                    tardiness[j] = machine_time[i] - d_j[j]
    maximum = max(tardiness)
    return maximum


def max_lateness(individual):
    machine_time = [0 for x in range(n)]
    lateness = [0 for x in range(n)]
    maximum = 0
    for j in range(0, len(machine_time)):
        current = individual[j]
        machine_time[0] += p_ij[current][0]
        for i in range(1, m):
            if machine_time[i] < machine_time[i - 1]:
                machine_time[i] = (machine_time[i - 1] - machine_time[i]) + machine_time[i] + p_ij[current][i]
            else:
                machine_time[i] += p_ij[current][i]

            if i == m - 1:
                lateness[j] = machine_time[i] - d_j[j]
    maximum = max(lateness)
    return maximum


# Function to evaluate individuals
def evaluate(individual):
    x1 = makespan(individual)
    x2 = total_flow_time(individual)
    x3 = max_tardiness(individual)

    c1 = 12 * n / 20
    c2 = 1
    c3 = 30 * n / 20
    sx = c1 * x1 + c2 * x2 + c3 * x3
    return sx,


# def evaluate(individual):
#     objective1 = calculate_objective1(individual)
#     objective2 = calculate_objective2(individual)
#
#     return objective1, objective2

def genetic_algorithm():
    # main algorithm
    population = toolbox.population(n=20)
    # algorithms.eaSimple(population, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN)
    fitnesses = map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
        # temp, = ind.fitness.values
        # print(temp)
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
n = 20
p_ij, d_j = generate_instance(n, m)
# print(p_ij)
# print(d_j)
# algorithm parameters
CXPB, MUTPB = 0.5, 0.3  # probability of performing a crossover, mutation probability, number of generations
IND_SIZE = n
# a minimizing fitness is built using negatives weights
# create a fitness function with few objectives
# creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
# creator.create("Individual", list, fitness=creator.FitnessMulti)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(IND_SIZE), IND_SIZE)
toolbox.register("individual", tools.initIterate, creator.Individual,
                 toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes,
                 indpb=0.1)  # indpb probability for each attribute to be exchanged to another position
toolbox.register("select", tools.selTournament, tournsize=int(0.1 * n))
toolbox.register("evaluate", evaluate)

n_arr = []
sx_arr = []
rep = 10
NGEN = 10
while NGEN < 10000:
    n_arr.append(NGEN)
    sum = 0
    for i in range(0, rep):
        # Main driver code
        result, res = genetic_algorithm()
        res = list(res)
        minimum, = min(res)
        sum = sum + minimum
    sx_arr.append(sum / rep)
    NGEN = NGEN * 2

plt.figure(1)
plt.title("s(x) vs. Iterations")
plt.xlabel("iterations")
plt.ylabel("s(x)")
plt.plot(n_arr, sx_arr, color='purple')
plt.show()
