from collections import defaultdict
import numpy as np
from typing import List

from beras.core import Tensor

class BasicOptimizer:
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate

    def apply_gradients(self, trainable_params: List[Tensor], grads: List[Tensor]) -> None:
        return NotImplementedError


class RMSProp:
    def __init__(self, learning_rate: float, beta: float = 0.9, epsilon: float = 1e-6) -> None:
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.v = defaultdict(lambda: 0)

    def apply_gradients(self, trainable_params: List[Tensor], grads: List[Tensor]) -> None:
        return NotImplementedError


class Adam:
    def __init__(
        self, learning_rate: float, beta_1: float = 0.9, beta_2: float = 0.999, epsilon: float = 1e-7, amsgrad: bool = False
    ) -> None:

        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.m = defaultdict(lambda: 0)         # First moment zero vector
        self.v = defaultdict(lambda: 0)         # Second moment zero vector.
        self.t = 0                              # Time counter

    def apply_gradients(self, trainable_params: List[Tensor], grads: List[Tensor]) -> None:
        return NotImplementedError
