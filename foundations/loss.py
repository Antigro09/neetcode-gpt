import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        result = z - np.max(z)
        result2 = np.exp(result)
        result3 = result2 / np.sum(result2)
        return np.round(result3, 4)
