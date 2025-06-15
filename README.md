###### Basis

###### Basis is a low-level PyTorch library for building neural networks from scratch. It is used to create custom layers, optimizers, and other components from scratch. One can think of it as the laboratory for learning about the inner workings of neural networks.

###### Project Structure

```
basis
├── src
│   ├── layers
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── input.py
│   │   ├── dense.py
│   │   └── weight.py
│   ├── optimizers
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── gradient_descent.py
│   ├── losses
│   │   ├── __init__.py
│   │   └── cross_entropy.py
│   ├── models
│   │   ├── __init__.py
│   │   └── neural_network.py
│   ├── trainers
│   │   ├── __init__.py
│   │   └── trainer.py
│   ├── utils
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   └── metrics.py
│   └── main.py
├── notebooks
│   ├── data_exploration.ipynb
│   └── experiments.ipynb
├── data
│   ├── raw
│   ├── processed
│   └── README.md
├── visualizations
│   ├── pre_training
│   │   └── data_analysis.py
│   ├── post_training
│   │   ├── loss_curves.py
│   │   ├── predictions.py
│   │   └── parameter_evolution.py
│   └── utils.py
├── tests
│   ├── test_layers.py
│   ├── test_optimizers.py
│   └── test_models.py
├── configs
│   └── config.yaml
├── requirements.txt
├── setup.py
└── README.md
```

###### Installation

###### To install the required dependencies, run:

```
pip install -r requirements.txt
```