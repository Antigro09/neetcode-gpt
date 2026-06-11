import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        x, W1, b1, W2, b2, y_true = map(np.array, [x, W1, b1, W2, b2, y_true])
        z1 = W1 @ x + b1
        a1 = np.maximum(0, z1)
        z2 = W2 @ a1 + b2
        pred = z2
        outGrad = (2 * (z2 - y_true)) / np.size(y_true)
        dW2 = np.outer(outGrad, a1)
        db2 = outGrad
        reluGrad = W2.T @ outGrad
        mask = (z1 > 0).astype(float)
        outGrad1 = reluGrad * mask
        dW1 = np.outer(outGrad1, x)
        db1 = outGrad1

        loss = (1/len(y_true) * np.sum((pred - y_true)**2)) 

        return {'loss': round(float(loss), 4), 'dW1': np.round(dW1, 4).tolist(), 'db1': np.round(db1, 4).tolist(), 'dW2': np.round(dW2, 4).tolist(), 'db2': np.round(db2, 4).tolist()}