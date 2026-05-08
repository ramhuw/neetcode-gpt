import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        y = self.sigmoid(np.dot(x, w) + b)
        L = 0.5 * (np.dot(y, y_true))
        a1 = np.round(np.dot(y - y_true, y * (1 - y)) * x, 5)
        a2 = round(np.dot(y - y_true, y*(1 - y)), 5)
        return (a1, a2)
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        return np.round(1/(1 + np.exp(-z)), 5)
