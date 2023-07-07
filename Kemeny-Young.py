"""
Paper Title: Handling imbalanced class in melanoma: Kemeny--Young rule based optimal rank aggregation and Self-Adaptive Differential Evolution Optimization
Journal: Engineering Applications of Artificial Intelligence
Authors: Gaurav Srivastava and Nitesh Pradhan
Date: July 2023
"""

import numpy as np
from itertools import permutations


class KemenyYoungEnsemble:
    def __init__(self, models):
        """
        Initializes the Kemeny-Young ensemble.

        Parameters:
            models (list): A list of deep learning models to be aggregated.
        """
        self.models = models
        self.num_models = len(models)

    def aggregate(self, X):
        """
        Aggregates predictions from the models using the Kemeny-Young rule.

        Parameters:
            X (numpy.ndarray): Input data for aggregation.

        Returns:
            numpy.ndarray: Aggregated predictions.
        """
        predictions = [model.predict(X) for model in self.models]
        num_samples = X.shape[0]
        num_classes = predictions[0].shape[1]

        # Compute pairwise ranking matrix
        ranking_matrix = np.zeros((num_samples, self.num_models, num_classes))
        for i in range(num_samples):
            for j in range(self.num_models):
                for k in range(num_classes):
                    count = 0
                    for l in range(self.num_models):
                        if predictions[l][i, k] >= predictions[j][i, k]:
                            count += 1
                    ranking_matrix[i, j, k] = count

        # Find the optimal permutation using Kemeny-Young rule
        best_permutation = None
        best_score = float('inf')
        for permutation in permutations(range(self.num_models)):
            score = np.sum(ranking_matrix[:, permutation, :])
            if score < best_score:
                best_score = score
                best_permutation = permutation

        # Aggregate predictions based on the optimal permutation
        aggregated_predictions = np.zeros((num_samples, num_classes))
        for i, permutation_index in enumerate(best_permutation):
            aggregated_predictions += predictions[permutation_index]

        # Normalize aggregated predictions
        aggregated_predictions /= self.num_models

        return aggregated_predictions