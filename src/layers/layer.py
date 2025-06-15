# -*- coding: utf-8 -*-
"""
Layer Class for Neural Networks
Created by Protim Roy
"""

import torch;
from abc import ABC, abstractmethod
from .base import BaseLayer

class Layer( BaseLayer, ABC ):
    """"
    Base class for all layers in the neural network."
    """
    def __init__( self, hyperparams: dict, name: str ):
        """"
        Constructor for the Layer class.
        
        Args:
        Returns:
        Attributes:
        Raises:

        """
        super().__init__( hyperparams, name );
        self.inputs = None;
        self.outputs = None;
        self.node_no = None;

    @abstractmethod
    def output( self, inputs: torch.Tensor ):
        """"
        Abstract method to be implemented by subclasses.

        Args:
        Returns:
        Attributes:
        Raises:
        """
        pass;
    
    def get_parameters( self ) -> dict:
        """
        Returns the parameters of the layer.
        
        Args:
        Returns:
        Attributes:
        Raises:
        """
        return {};

    @abstractmethod
    def backward( self, grad_output: torch.Tensor ):
        """
        Computes gradients for parameters and propagates gradient to the previous layer.
        Args:
            grad_output (torch.Tensor): Gradient of the loss w.r.t. the output of this layer.
        Returns:
            torch.Tensor: Gradient of the loss w.r.t. the input of this layer.
        """
        pass
