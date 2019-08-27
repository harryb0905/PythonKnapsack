import numpy as np
import random

class GA_Knapsack:
    def __init__(self, space_units, amounts, max_space=40, gens=50, crossover_prob=0.8, mutation_prob=0.2):
        assert len(space_units) > 0
        assert len(amounts) > 0
        assert max_space > 0

        self.pop_size = 50
        self.space_units = space_units
        self.amounts = amounts
        self.max_space = max_space

        # No of generations/iterations
        self.gens = gens
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

        self.chromosome_len = len(self.space_units)

    def _init_population(self):
        # Generate random population of population_size chromosomes
        self.population = np.zeros((self.pop_size, self.chromosome_len))

        for i in range(self.pop_size):
            temp_chromosome = np.random.randint(2, size=self.chromosome_len)
            while (sum(np.multiply(temp_chromosome, self.space_units)) > self.max_space):
                temp_chromosome = np.random.randint(2, size=self.chromosome_len)
            self.population[i, :] = temp_chromosome

        fitness_cols = np.zeros((self.pop_size, 1))
        self.population = np.concatenate((self.population, fitness_cols), axis=1)

    def _calculate_pop_fitness(self):
        for j in range(self.pop_size):
            self.population[j, 16] = sum(np.multiply(self.population[j, 0:16], self.amounts))

    def _selection(self, weights):
        accumulation = np.cumsum(weights)
        p = random.uniform(0.0, 1.0)
        chosen_index = -1
        for i in range(len(accumulation)):
            if (accumulation[i] > p):
                chosen_index = i
                break

        return chosen_index

    def _crossover(self, chrom_1, chrom_2):
        # Create offspring using parents and crossover point
        cross_point = random.randint(0, 16)
        offspring_1 = np.concatenate((chrom_1[0:cross_point], chrom_2[cross_point:]), axis=0)
        offspring_2 = np.concatenate((chrom_2[0:cross_point], chrom_1[cross_point:]), axis=0)
        return offspring_1, offspring_2

    def _mutate(self, chrom):
        point_1 = random.randint(0, len(chrom)-1)
        chrom[point_1] = 1 - chrom[point_1]
        return chrom

    def run(self):
        # Create initial population
        self._init_population()

        fittest = []
        population_new_num = 2

        for gen in range(self.gens):
            # Calculate fitness
            self._calculate_pop_fitness()

            # Elitism - sort population on fitness and keep best 2
            population = self.population[self.population[:, 16].argsort()]
            new_population = np.zeros((self.pop_size, self.chromosome_len))
            new_population[0:2, :] = self.population[self.pop_size-2:self.pop_size, 0:16]
            fittest.append(self.population[-1][-1])

            population_new_num = 2

            # Repeat until new population is full
            while (population_new_num < self.pop_size-1):
                # Weights = fitness of each chromosome / sum of total fitness of all chromosomes
                weights = self.population[:, 16] / sum(self.population[:, 16])

                # Use a selection method and pick two chromosomes
                choice_1 = self._selection(weights)
                choice_2 = self._selection(weights)
                parent_chromosome_1 = self.population[choice_1, 0:16]
                parent_chromosome_2 = self.population[choice_2, 0:16]

                offspring_1 = parent_chromosome_1
                offspring_2 = parent_chromosome_2

                # Crossover prob and random pick cross point
                if (random.uniform(0, 1) < self.crossover_prob):
                    offspring_1, offspring_2 = self._crossover(parent_chromosome_1, parent_chromosome_2)

                # Mutation prob and random pick bit to switch (bit flip)
                if (random.uniform(0, 1) < self.mutation_prob):
                    offspring_1 = self._mutate(offspring_1)

                if (random.uniform(0, 1) < self.mutation_prob):
                    offspring_2 = self._mutate(offspring_2)

                # Put in new population if within max space
                if (sum(np.multiply(offspring_1, self.space_units)) <= self.max_space):
                    population_new_num += 1
                    new_population[population_new_num, :] = offspring_1

                if (sum(np.multiply(offspring_2, self.space_units)) <= self.max_space):
                    if (population_new_num < self.pop_size-1):
                        population_new_num += 1
                        new_population[population_new_num, :] = offspring_2

            # Replace, last column not updated yet
            self.population[:, 0:16] = new_population

        # Evaluate fitness scores and rank them
        for i in range(self.pop_size):
            population[i,16] = sum(np.multiply(population[i, 0:16], amounts))

        self.population = self.population[self.population[:, 16].argsort()]
        best = self.population[-1]
        fittest[-1] = best[-1]

        print('Fittest solution:', best[:-1], 'with amount of £%.2f' % best[-1])
        return fittest

if __name__ == "__main__":
    space_units = [3,   5,   6,   6,   7,   3,   1,   6,   2,   9,   3,   7,   3,   4,   6,   9]
    amounts     = [1.1, 5.1, 0.2, 4.4, 9.1, 0.4, 9.5, 1.8, 3.7, 2.2, 0.9, 4.3, 2.8, 1.1, 4.9, 2.4]
    max_space = 40
    ga = GA_Knapsack(space_units, amounts, max_space)
    ga.run()
