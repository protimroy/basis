# -*- coding: utf-8 -*-
"""
Weight Layer Class for Neural Networks
Created by Protim Roy
"""

import torch;
from .base import BaseLayer
from .input import InputLayer
from .layer import Layer

from .dense import DenseLayer

class WeightLayer( BaseLayer ):
    """
    Weight layer for the perceptron using low-level PyTorch.
    """
    def __init__( self, src: Layer, dest: Layer, device='cpu' ):
        """
        Constructor for the weight layer.

        Args:
            params : dict
            src : ( Layer ) source layer.
            dest : ( Layer ) destination layer.
            device : device to place tensors on

        Returns:
            None

        Attributes:
            self.src : ( Layer ) source layer.
            self.dest : ( Layer ) destination layer.
            self.input_size : ( tuple )
            self.output_size : ( tuple )
            self.name : ( str ) name of the layer.
            self.weights : ( array ) weights of the layer.
            self.weighted_output : ( array ) weighted output of the layer.

        Raises: 
            None
        """
        if not isinstance( src, ( InputLayer, DenseLayer ) ):
            raise TypeError( f"Source layer for WeightLayer must be InputLayer or DenseLayer, not {type(src).__name__}" );
        if not isinstance( dest, DenseLayer ):
            raise TypeError( f"Destination layer for WeightLayer must be DenseLayer, not {type(dest).__name__}" );
        
        self.src = src;
        self.dest = dest;
        self.device = device;

        fan_in = src.node_no;
        fan_out = dest.node_no;
        #limit = math.sqrt( 2. / ( fan_in + fan_out ) );
        limit = torch.sqrt( torch.tensor( 2 ) / ( fan_in + fan_out ) ).item();
        #self.weights = torch.randn(fan_out, fan_in).float() * limit;
        self.weights = torch.randn( fan_out, fan_in, dtype=torch.float32, requires_grad=True, device=self.device ) * limit;
        self.initial_weights = self.weights.data.clone();

        self.name = f"W_{self.src.name}_{self.dest.name}_layer";

    def output( self, inputs: torch.Tensor ) -> torch.Tensor:
        """
        Matrix multiplication between the inputs and the weights.

        Args:
        Returns:
        Attributes:
        Raises:
        """
        inputs = inputs.to(self.device)
        self.weighted_output = torch.matmul(self.weights, inputs)
        return self.weighted_output;

    def get_parameters( self ) -> dict:
        """
        Get the parameters of the layer.

        Args:
        Returns:
        Attributes:
        Raises:
        """
        return { f"{self.name}": self.weights };

    def backward( self, grad_output_wrt_Z: torch.Tensor ) -> torch.Tensor:
        """
        Computes gradients for weights and propagates gradient to the previous layer.
        Args:
            grad_output_wrt_Z (torch.Tensor): Gradient of the loss w.r.t. the output of this WeightLayer (dL/dZ).
                                              Shape: (fan_out, num_samples) or (fan_out, 1) for single sample.
        Returns:
            torch.Tensor: Gradient of the loss w.r.t. the input of this WeightLayer (dL/dX).
                          Shape: (fan_in, num_samples) or (fan_in, 1).
        """
        grad_output_wrt_Z = grad_output_wrt_Z.to(self.device)
        
        # dL/dW = dL/dZ @ X.T
        dL_dW = torch.matmul( grad_output_wrt_Z, self.src.outputs.transpose(-2, -1) );

        if self.weights.grad is None:
            self.weights.grad = torch.zeros_like( self.weights );
        self.weights.grad += dL_dW; # Accumulate gradient

        # dL/dX = W.T @ dL/dZ
        grad_input_wrt_X = torch.matmul( self.weights.transpose(-2, -1), grad_output_wrt_Z );
        
        return grad_input_wrt_X;

    def get_gradients( self ) -> dict:
        """
        Returns the gradients of the layer.
        
        Args:
        Returns:
        Attributes:
        Raises:
        """
        return { f"{self.name}": self.weights.grad };

    def clear_gradients(self):
        """
        Clears the gradients of the layer.
        
        Args:
        Returns:
        Attributes:
        Raises:
        """
        if self.weights.grad is not None:
            self.weights.grad.zero_();

    def reset( self):
        """
        Resets the weights to their initial values.
        
        Args:
        Returns:
        Attributes:
        Raises:
        """
        self.weights.data = self.initial_weights.clone();
        self.clear_gradients();