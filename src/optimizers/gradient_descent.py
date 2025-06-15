import torch;

class GD:
    def __init__( self, params: list, lr: float ):
        """
        Args:
        Returns:
        Attributes:
        Raises:
        """
        self.params = params # These are the actual tensor objects
        self.lr = lr
        self.name  = "Gradient Descent"  # Gradient Descent
    
    def step(self):
        """
        Update parameters using their .grad attribute.
        Assumes .grad contains the accumulated (and possibly averaged) gradient for the batch.
        """
        with torch.no_grad(): # Disable gradient tracking for the update step itself
            for param in self.params:
                if param.grad is None:
                    # This might happen if a parameter wasn't used or no gradient flowed to it.
                    # print(f"Warning: param {param.shape} has no gradient during step.")
                    continue
                param -= self.lr * param.grad # Update rule: param = param - lr * gradient

    def zero_grad(self):
        """
        Clears .grad attributes of all parameters managed by the optimizer.
        """
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()