class LayerChainMixin:
    """Mixin to provide layer chaining functionality"""
    def __rshift__(self, other):
        """Overwrite the right shift operator to connect the layers."""
        # Import here to avoid circular dependency
        from ..models.neural_network import NeuralNetwork
        
        # Get device from the first layer
        device = getattr(self, 'device', 'cpu')
        return NeuralNetwork(self, other, device=device)