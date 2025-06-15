# -*- coding: utf-8 -*-
"""
Dense Layer Class for Neural Networks
Created by Protim Roy
"""

import torch;
from .layer import Layer

class DenseLayer( Layer ):
    """
    Low-level PyTorch layer class for the perceptron.
    """
    def __init__( self, hyperparams: dict, name: str, transfer: str, device='cpu' ):
        """
        Constructor for the Layer class.

        Args:
        Returns:
        Attributes:
        Raises:
        """
        #super( Layer, self ).__init__( hyperparams, name );
        super().__init__( hyperparams, name ); # Corrected super call
        self.transfer_name = transfer;
        self.inputs = None;
        self.outputs = None;
        self.device = device

        if "hidden" in name:
            self.node_no = hyperparams['hidden_units'];
        else:
            self.node_no = hyperparams['output_units'];
        
        self.bias = torch.zeros( (self.node_no, 1), dtype=torch.float32, requires_grad=True, device=self.device );
        self.initial_bias = self.bias.data.clone()


    def transfer_fx( self, inputs: torch.Tensor, deriv=False ) -> torch.Tensor:
        """
        Transfer function for the layer.

        Args:
        Returns:
        Attributes:
        Raises:
        """
        inputs = inputs.to(self.device)
        if self.transfer_name == "sigmoid":
            fx = 1 / ( 1 + torch.exp( -inputs ) );
            if deriv:
                return fx * ( 1 - fx );
            return fx;
        
        else:
            raise ValueError( f"Unknown transfer function: { self.transfer_name }" );

    def output( self, inputs: torch.Tensor, deriv=False ) -> torch.Tensor:
        """
        Calculate the output with the activation function and inputs.

        Args:
        Returns:
        Attributes:
        Raises:
        """
        self.inputs = inputs.to(self.device) # This is actually z = Wx from WeightLayer
        self.weighted_input = self.inputs + self.bias # z_with_bias = Wx + b
        self.outputs = self.transfer_fx( self.weighted_input, deriv=deriv ) # a = sigma(z_with_bias)
        return self.outputs
    
    def get_parameters( self ) -> dict:
        """
        Returns the parameters of the layer.
        Args:
            None
        Returns:
            None
        Attributes:
            None
        Raises:
            None
        """
        return { f"{self.name}": self.bias }
    
    def backward( self, deriv_loss: torch.Tensor ) -> torch.Tensor:
        """
        Computes gradients for parameters and propagates gradient to the previous layer.
        Args:
            deriv_loss (torch.Tensor): derivative of the loss w.r.t. the output of this layer (dL/da).
        Returns:
            torch.Tensor: Gradient of the loss w.r.t. the input of this layer (dL/dZ for the previous WeightLayer).
        """
        deriv_loss = deriv_loss.to(self.device)
        
        # Calculate dL / dz = dL/da * da/dz
        # where a is the activation function and z is the weighted sum of inputs, so da/dz is the derivative of the activation function
        da_dz = self.transfer_fx( self.weighted_input, deriv=True );
        
        dL_dz = deriv_loss * da_dz; # element-wise product (dL/da * da/dz)

        # Accumulate gradient for bias (dL/dbias = dL/dz * dz/dbias = dL/dz * 1 = dL/dz)
        if self.bias.grad is None:
            self.bias.grad = torch.zeros_like(self.bias)
        
        # Corrected: Directly add dL_dz, as it already has the correct shape and per-node gradients
        self.bias.grad += dL_dz # No sum needed for single sample, dL_dz is already (num_nodes, 1)

        # Gradient to pass to the previous layer (WeightLayer) is dL/dZ
        # This is exactly dL_dz
        grad_input = dL_dz;
        return grad_input;

    def get_gradients( self ) -> dict:
        """
        Returns the gradients of the layer's parameters.
        
        Args:
        Returns:
        Attributes:
        Raises:
        """
        return { f"{self.name}": self.bias.grad } if self.bias.grad is not None else {}

    def clear_gradients(self):
        """
        Clears the gradients of the layer's parameters.
        
        Args:
        Returns:
        Attributes:
        Raises:
        """
        if self.bias.grad is not None:
            self.bias.grad.zero_()
    
    def reset( self ):
        """
        Resets the layer's parameters to their initial state.
        
        Args:
        Returns:
        Attributes:
        Raises:
        """
        self.bias.data = self.initial_bias.clone();
        self.clear_gradients();