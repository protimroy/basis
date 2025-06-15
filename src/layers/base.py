# -*- coding: utf-8 -*-
"""
Base Layer Class for Neural Networks
Created by Protim Roy
"""

from .mixins import LayerChainMixin
class BaseLayer( LayerChainMixin ):
    """
    Base class for the Layer class.
    """
    def __init__( self, hyperparams : dict, name : str  ):
        """
        Constructor for the Base Layer class.

        Args:
        Returns:
        Attributes:
        Raises:
        """
        self.hyperparams = hyperparams;
        self.name = name;