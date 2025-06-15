###### Basis

###### Basis is a low-level PyTorch library for building neural networks from scratch. It is used to create custom layers, optimizers, and other ingredients required in crafting a neural network. One can think of it as the laboratory for exploring and experimenting about the inner workings of neural networks.

```
basis/
├── src/
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── input.py
│   │   ├── dense.py
│   │   ├── mixin.py
│   │   └── weight.py
│   ├── optimizers/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── gradient_descent.py
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── base_loss.py
│   │   └── cross_entropy.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── neural_network.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── tensorboard_logs.py
│   │   └── gpu_monitor.py
├── notebooks/
│   ├── experiments.ipynb
│   └── runs/  # TensorBoard logs directory
├── requirements.txt
├── setup.py
└── README.md
```

###### Installation

###### To install the required dependencies, run:

```
pip install -r requirements.txt
```

###### Usage

###### See `notebooks/experiments.ipynb` for examples of how to use the library.