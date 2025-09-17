import numpy as np
from typing import List

from beras.core import Diffable, Tensor


class Loss(Diffable):
    @property
    def weights(self) -> List[Tensor]:
        return []

    def get_weight_gradients(self) -> List[Tensor]:
        return []


class MeanSquaredError(Loss):
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return NotImplementedError

    def get_input_gradients(self) -> List[Tensor]:
        return NotImplementedError


class CategoricalCrossEntropy(Loss):
    def __init__(self, epsilon: float = 1e-12) -> None:
        self.epsilon = epsilon
        
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Categorical cross entropy forward pass!"""
        return NotImplementedError

    def get_input_gradients(self) -> List[Tensor]:
        """Categorical cross entropy input gradient method!"""
        return NotImplementedError
