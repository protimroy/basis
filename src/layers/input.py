# -*- coding: utf-8 -*-
"""
Input Layer Class for Neural Networks
Created by Protim Roy
"""

import torch;
from .layer import Layer

class InputLayer( Layer ):
    """
    Input layer for the perceptron.
    """
    def __init__( self, hyperparams : dict, name: str, device='cpu'  ):
        """
        Constructor for the Input Layer class.

        Args:
        Returns:
        Attributes:
        Raises:
        """
        super( InputLayer, self ).__init__( hyperparams, name );
        self.node_no = hyperparams['input_units'];
        self.device = device
    
    def output( self, inputs: torch.Tensor ) -> torch.Tensor:
        """
        Mirror the inputs.

        Args:
        Returns:
        Attributes:
        Raises:
        """
        self.inputs = inputs.to(self.device)
        self.outputs = self.inputs;
        return self.outputs;

    def backward( self, grad_output: torch.Tensor ) -> torch.Tensor:
        # Input layer doesn't have parameters to update, nor does it propagate further back typically.
        # However, to fit the chain, it should return the grad_output.
        return grad_output.to(self.device)

    def clear_gradients(self): # Input layer has no gradients
        pass

    def get_gradients(self):
        return super().get_gradients()