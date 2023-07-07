"""
Paper Title: Handling imbalanced class in melanoma: Kemeny--Young rule based optimal rank aggregation and Self-Adaptive Differential Evolution Optimization
Journal: Engineering Applications of Artificial Intelligence
Authors: Gaurav Srivastava and Nitesh Pradhan
Date: July 2023
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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


def cost_sensitive_objective(class_weights):
    """
    Cost-sensitive objective function for optimizing class weights.

    Parameters:
        class_weights (numpy.ndarray): The class weights to evaluate.

    Returns:
        float: The fitness value.
    """
    # Load the melanoma image dataset
    data_dir = '/path/to/melanoma/dataset'  # Replace with the actual path to your dataset
    batch_size = 32
    input_size = (224, 224)  # Input size for ResNet50

    # Split the dataset into training and testing sets
    train_data_dir = data_dir + '/train'
    validation_data_dir = data_dir + '/validation'

    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=input_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )

    validation_generator = val_datagen.flow_from_directory(
        validation_data_dir,
        target_size=input_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    # Create a ResNet50 model with a custom output layer
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_size + (3,))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=x)

    # Set the class weights
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    class_weight = {0: class_weights[0], 1: class_weights[1]}

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        class_weight=class_weight,
        verbose=1
    )

    # Make predictions on the validation set
    y_pred = model.predict(validation_generator)
    y_pred = np.where(y_pred > 0.5, 1, 0)

    # Calculate the F1 score as the fitness value
    y_true = validation_generator.classes
    fitness = f1_score(y_true, y_pred)

    return -fitness  # Minimize the negative F1 score


def main():
    # Define bounds for each class weight (assuming binary classification)
    bounds = [(0.01, 100), (0.01, 100)]

    # Create SADE optimizer
    optimizer = SADE(func=cost_sensitive_objective, bounds=bounds, population_size=50, max_iterations=1000)

    # Run optimization
    best_solution, best_fitness = optimizer.optimize()

    # Print the optimized class weights
    print("Optimized Class Weights:", best_solution)
    print("Best F1 Score:", -best_fitness)  # Convert back to positive F1 score


if __name__ == "__main__":
    main()
