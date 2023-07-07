"""
Paper Title: Handling imbalanced class in melanoma: Kemeny--Young rule based optimal rank aggregation and Self-Adaptive Differential Evolution Optimization
Journal: Engineering Applications of Artificial Intelligence
Authors: Gaurav Srivastava and Nitesh Pradhan
Date: July 2023
"""

import numpy as np
from itertools import permutations
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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


def main():
    # Load the melanoma test data
    data_dir = '/path/to/melanoma/test_data'  # Replace with the actual path to your test data directory
    batch_size = 32
    input_size = (224, 224)  # Input size for the models

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        data_dir,
        target_size=input_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    # Load the 8 models
    models = ['model1.h5', 'model2.h5', 'model3.h5', 'model4.h5', 'model5.h5', 'model6.h5', 'model7.h5', 'model8.h5']
    for i in range(1, 9):
        model_path = f'model{i}.h5'
        model = load_model(model_path)
        models.append(model)

    ensemble = KemenyYoungEnsemble(models)

    # Aggregate predictions using Kemeny-Young rule
    aggregated_predictions = ensemble.aggregate(test_generator)

if __name__ == "__main__":
    main()
