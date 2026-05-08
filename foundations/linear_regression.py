import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.round(np.dot(X, weights), 5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        n = len(model_prediction)
        return np.round(np.sum(np.array([(model_prediction[i] - ground_truth[i])**2 for i in range(n)]))/n, 5)
