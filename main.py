import math
import random
from deap import algorithms, base, creator, tools
import RandomNumberGenerator

random = RandomNumberGenerator.RandomNumberGenerator(5546568)


# Function to generate instance
def generate_instance(n: int, m: int):
    matrix = [[0 for x in range(m)] for y in range(n)]
    A = 0
    for i in range(0, n):
        for j in range(0, m):
            matrix[i][j] = random.nextInt(1, 99)
            A = A + matrix[i][j]
    B = math.floor(1 / 2 * A)
    A = math.floor(1 / 6 * A)

    matrix2 = [0 for x in range(n)]
    for i in range(0, n):
        matrix2[i] = random.nextInt(A, B)
    return (matrix, matrix2)







if __name__ == "__main__":
    m = 3
    n = 5
    p_ij, d_j = generate_instance(n, m)
    print(p_ij)
    print(d_j)
