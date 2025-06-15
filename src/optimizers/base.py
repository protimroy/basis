class BaseOptimizer( object ):
    def __init__( self, params, lr ):
        self.params = params
        self.lr = lr

    def step(self):
        raise NotImplementedError( f"Optimizer step method must be implemented." )

    def zero_grad( self ):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()