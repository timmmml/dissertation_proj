"""This module implements a custom GRU for the project.

Vanilla GRU with one modification: it could be initialised with an altered
scheme, in which it recycles the input for the preparatory phase. This is so we can save some space.

"""

import torch
import torch.nn as nn

class CustomGRU(nn.Module):
    """CustomGRU model - GRU graph with the option to recycle input.

    """
    def __init__(self, **kwargs):
        super(CustomGRU, self).__init__()
        if "recycle_input" in kwargs:
            self.recycle_input = kwargs["recycle_input"]
            del kwargs["recycle_input"]
        self.gru =...
        # TODO
        # Decide if this is even a good idea. You would need the space anyway if you want to use matrix multiplication.

