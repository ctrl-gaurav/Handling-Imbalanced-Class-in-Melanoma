"""
Paper Title: Handling imbalanced class in melanoma: Kemeny--Young rule based optimal rank aggregation and Self-Adaptive Differential Evolution Optimization
Journal: Engineering Applications of Artificial Intelligence
Authors: Gaurav Srivastava and Nitesh Pradhan
Date: July 2023
"""

import numpy as np

class SADE:
    def __init__(self, func, bounds, population_size=50, max_iterations=1000, mutation_factor=0.5, crossover_probability=0.7):
        """
        Initializes the SADE optimizer.

        Parameters:
            func (function): The objective function to be minimized.
            bounds (list): A list of tuples representing the lower and upper bounds for each dimension of the search space.
            population_size (int): The size of the population (default: 50).
            max_iterations (int): The maximum number of iterations (default: 1000).
            mutation_factor (float): The scaling factor for mutation (default: 0.5).
            crossover_probability (float): The probability of crossover (default: 0.7).
        """
        self.func = func
        self.bounds = bounds
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.mutation_factor = mutation_factor
        self.crossover_probability = crossover_probability
        
    def optimize(self):
        """
        Runs the SADE optimization algorithm.

        Returns:
            tuple: A tuple containing the best solution found and its fitness value.
        """
        dim = len(self.bounds)
        lower_bound, upper_bound = np.array(self.bounds).T

        # Step 1: Initialize population
        population = np.random.uniform(low=lower_bound, high=upper_bound, size=(self.population_size, dim))

        # Step 2: Evaluate fitness of the initial population
        fitness = np.array([self.func(individual) for individual in population])

        # Step 3: Main loop
        for iteration in range(self.max_iterations):
            new_population = np.empty_like(population)

            for i in range(self.population_size):
                # Step 3.1: Mutation
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = population[i] + self.mutation_factor * (a - b) + self.mutation_factor * (c - population[i])

                # Step 3.2: Crossover
                trial = np.where(np.random.uniform(0, 1, dim) < self.crossover_probability, mutant, population[i])

                # Step 3.3: Bound handling
                trial = np.clip(trial, lower_bound, upper_bound)

                # Step 3.4: Selection
                trial_fitness = self.func(trial)
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                else:
                    new_population[i] = population[i]

            # Step 3.5: Update population
            population = new_population

        # Step 4: Return the best solution found
        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]

        return best_solution, best_fitness


def sphere(individual):
    """
    Sphere function: sum of squares of individual components.

    Parameters:
        individual (numpy.ndarray): The individual (solution) to evaluate.

    Returns:
        float: The fitness value.
    """
    return np.sum(individual**2)


def main():
    # Define bounds for each dimension
    bounds = [(-5, 5), (-5, 5), (-5, 5)]

    # Create SADE optimizer
    optimizer = SADE(func=sphere, bounds=bounds, population_size=50, max_iterations=1000)

    # Run optimization
    best_solution, best_fitness = optimizer.optimize()

    # Print the result
    print("Best solution:", best_solution)
    print("Best fitness:", best_fitness)


if __name__ == "__main__":
    main()
