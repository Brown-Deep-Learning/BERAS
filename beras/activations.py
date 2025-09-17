import numpy as np
from typing import List

from beras.core import Diffable, Tensor

class Activation(Diffable):
    @property
    def weights(self) -> List[Tensor]: 
        return []

    def get_weight_gradients(self) -> List[Tensor]: 
        return []


################################################################################
## Intermediate Activations To Put Between Layers

class LeakyReLU(Activation):

    ## TODO: Implement for default intermediate activation.

    def __init__(self, alpha: float = 0.3) -> None:
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        """Leaky ReLu forward propagation!"""
        return NotImplementedError

    def get_input_gradients(self) -> List[Tensor]:
        """
        Leaky ReLu backpropagation!
        To see what methods/variables you have access to, refer to the cheat sheet.
        Hint: Make sure not to mutate any instance variables. Return a new list[tensor(s)]
        """
        raise NotImplementedError

    def compose_input_gradients(self, J: Tensor) -> Tensor:
        return self.get_input_gradients()[0] * J

class ReLU(LeakyReLU):
    ## GIVEN: Just shows that relu is a degenerate case of the LeakyReLU
    def __init__(self) -> None:
        super().__init__(alpha=0)


################################################################################
## Output Activations For Probability-Space Outputs

class Sigmoid(Activation):
    
    ## TODO: Implement for default output activation to bind output to 0-1
    
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def get_input_gradients(self) -> List[Tensor]:
        """
        To see what methods/variables you have access to, refer to the cheat sheet.
        Hint: Make sure not to mutate any instance variables. Return a new list[tensor(s)]
        """
        raise NotImplementedError

    def compose_input_gradients(self, J: Tensor) -> Tensor:
        return self.get_input_gradients()[0] * J


class Softmax(Activation):

    ## TODO: Implement for default output activation to bind output to 0-1

    def forward(self, x: Tensor) -> Tensor:
        """Softmax forward propagation!"""
        ## Not stable version
        ## exps = np.exp(inputs)
        ## outs = exps / np.sum(exps, axis=-1, keepdims=True)

        ## HINT: Use stable softmax, which subtracts maximum from
        ## all entries to prevent overflow/underflow issues
        raise NotImplementedError

    def get_input_gradients(self) -> List[Tensor]:
        """Softmax input gradients!"""
        x, y = self.inputs + self.outputs
        bn, n = x.shape
        grad = np.zeros(shape=(bn, n, n), dtype=x.dtype)
        
        # TODO: Implement softmax gradient
        # If you are stuck, refer to these links and the cheat sheet:w
        # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
        # https://stackoverflow.com/questions/48633288/how-to-assign-elements-into-the-diagonal-of-a-3d-matrix-efficiently
        raise NotImplementedError