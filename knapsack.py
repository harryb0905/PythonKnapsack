from selection import selection
import numpy as np
import random

space_units = [3,   5,   6,   6,   7,   3,   1,   6,   2,   9,   3,   7,   3,   4,   6,  9]
amounts     = [1.1, 5.1, 0.2, 4.4, 9.1, 0.4, 9.5, 1.8, 3.7, 2.2, 0.9, 4.3, 2.8, 1.1, 4.9, 2.4]

max_space = 40
chromosome_len = len(space_units)

gens = 10 # No of generations/iterations

crossover_prob = 0.8
mutation_prob = 0.2

population_size = 20 # No of chromosomes in population

population = np.zeros((population_size, chromosome_len)) # Generate random population of "population_size" chromosomes

for i in range(population_size):
    temp_chromosome = np.random.randint(2, size=chromosome_len)
    while (sum(np.multiply(temp_chromosome, space_units)) > max_space):
        temp_chromosome = np.random.randint(2, size=chromosome_len)

    population[i, :] = temp_chromosome

# Add extra column at the end for fitness scores
fitness_cols = np.zeros((population.shape[0], 1))
population = np.concatenate((population,fitness_cols), axis = 1)
population_new_num = 2
fittest = np.array([])

# Repeat for each new generation
for k in range(gens):
    # Calculate fitness
    for j in range(population_size):
        population[j, 16] = sum(np.multiply(population[j, 0:16], amounts))

    # Elitism - sort population on fitness and keep best 2
    population = population[population[:, 16].argsort()]
    new_population = np.zeros((population_size, chromosome_len))
    new_population[0:2, :] = population[population_size-2:population_size, 0:16]
    population_new_num = 2

    # Repeat until new population is full
    if (population_new_num < population_size):
        # Weights = fitness of each chromosome / sum of total fitness of all chromosomes
        weights = population[:, 16] / sum(population[:, 16])

        # Use a selection method and pick two chromosomes
        choice_1 = selection(weights)
        choice_2 = selection(weights)
        parent_chromosome_1 = population[choice_1, 0:16]
        parent_chromosome_2 = population[choice_2, 0:16]

        offspring_1 = parent_chromosome_1
        offspring_2 = parent_chromosome_2

        # Crossover prob and random pick cross point
        if (random.uniform(0, 1) < crossover_prob):
            cross_point = random.randint(0, 16)
            # Create offspring using parents and crossover point
            offspring_1 = np.concatenate((parent_chromosome_1[0:cross_point], parent_chromosome_2[cross_point::]), axis=0)
            offspring_2 = np.concatenate((parent_chromosome_2[0:cross_point], parent_chromosome_1[cross_point::]), axis=0)

        # Mutation prob and random pick bit to switch (bit flip)
        if (random.uniform(0, 1) < mutation_prob):
            # Mutate first offspring
            point_1 = random.randint(0,2)
            offspring_1[point_1] = 1 - offspring_1[point_1]

        if (random.uniform(0, 1) < mutation_prob):
            # Mutate second offspring
            point_2 = random.randint(0, 2)
            offspring_2[point_2] = 1 - offspring_2[point_2]

        # Put in new population if within max space
        if (sum(np.multiply(offspring_1, space_units)) <= max_space):
            population_new_num += 1
            new_population[population_new_num, :] = offspring_1

        if (sum(np.multiply(offspring_2, space_units)) <= max_space):
            if (population_new_num < 20):
                population_new_num += 1
                new_population[population_new_num, :] = offspring_2

        # Replace, last column not updated yet
        population[:, 0:16] = new_population
        np.concatenate((fittest, population[-1]))

# Evaluate fitness scores and rank them
for i in range(population_size):
    population[i,16] = sum(np.multiply(population[i, 0:16], amounts))

population = population[population[:, 16].argsort()]
best = population[-1]
print('Fittest solution:', best[:-1], 'with amount of £%.2f' % best[-1])