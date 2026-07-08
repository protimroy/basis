import torch;

class Loss( object ):
    def __init__( self, name ):
        """
        Args:
        Returns:
        Attributes:
        Raises:
        """
        self.name = name
    
    def binary_cross_entropy( self, output, target ):
        """
        Args:
        Returns:
        Attributes:
        Raises:
        """
        # Binary Cross-Entropy Loss
        return -torch.mean( target * torch.log( output ) + ( 1 - target ) * torch.log( 1 - output ) );

    def __call__( self, output, target ):
        """
        Args:
        Returns:
        Attributes:
        Raises:
        """
        # Ensure numerical stability by clipping output values
        epsilon = 1e-7;
        output = torch.clamp( output, epsilon, 1.0 - epsilon );

        # Store output and target for backward computation
        self.output = output;
        self.target = target;

        if self.name == "binary_cross_entropy":
            # Binary Cross-Entropy Loss
            return self.binary_cross_entropy( output, target );

    def backward( self ):
        """
        Computes the derivative of the loss with respect to its input.
        """
        if self.name == "binary_cross_entropy":
            if self.output is None or self.target is None:
                raise ValueError( "Output not set. Call forward() first." );
            # Corrected derivative for binary cross-entropy
            # dL/da = - (target/output - (1-target)/(1-output)) = (output - target) / (output * (1 - output))
            # The forward loss uses torch.mean(), so divide by the number of elements to
            # keep the backward pass consistent with the averaged forward loss. For a
            # single sample this divides by 1 and leaves the previous behaviour unchanged.
            n = self.output.numel()
            return (self.output - self.target) / (self.output * (1.0 - self.output) + 1e-9) / n # add epsilon for stability
