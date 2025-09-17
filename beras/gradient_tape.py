from collections import defaultdict
from typing import List, Optional, Dict, Any

from beras.core import Diffable, Tensor

class GradientTape:

    def __init__(self) -> None:
        # Dictionary mapping the object id of an output Tensor to the Diffable layer it was produced from.
        self.previous_layers: Dict[int, Optional[Diffable]] = defaultdict(lambda: None)

    def __enter__(self) -> 'GradientTape':
        # When tape scope is entered, all Diffables will point to this tape.
        if Diffable.gradient_tape is not None:
            raise RuntimeError("Cannot nest gradient tape scopes.")

        Diffable.gradient_tape = self
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # When tape scope is exited, all Diffables will no longer point to this tape.
        Diffable.gradient_tape = None

    def gradient(self, target: Tensor, sources: List[Tensor]) -> List[Tensor]:
        """
        Computes the gradient of the target tensor with respect to the sources.

        :param target: the tensor to compute the gradient of, typically loss output
        :param sources: the list of tensors to compute the gradient with respect to
        In order to use tensors as keys to the dictionary, use the python built-in ID function here: https://docs.python.org/3/library/functions.html#id.
        To find what methods are available on certain objects, reference the cheat sheet
        """

        ### TODO: Populate the grads dictionary with {weight_id, weight_gradient} pairs.

        queue = [target]                    ## Live queue; will be used to propagate backwards via breadth-first-search.
        grads = defaultdict(lambda: None)   ## Grads to be recorded. Initialize to None. Note: stores {id: list[gradients]}
        # Use id(tensor) to get the object id of a tensor object.
        # in the end, your grads dictionary should have the following structure:
        # {id(tensor): [gradient]}

        # What tensor and what gradient is for you to implement!
        # compose_input_gradients() and compose_weight_gradients() are methods that will be helpful
