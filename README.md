# Modular Stochastic Gradient MCMC for Jax

**[Introduction](#introduction) |
[Implemented Solvers](#quickstart-with-solvers-from-aliaspy) |
[Features](#features) | [Installation](#installation) |
[Contributing](#contributing)**

[![CI](https://github.com/tummfm/jax-sgmc/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/tummfm/jax-sgmc/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/jax-sgmc/badge/?version=latest)](https://jax-sgmc.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/jax-sgmc.svg)](https://badge.fury.io/py/jax-sgmc)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Introduction

**JaxSGMC** brings Stochastic Gradient Markov chain Monte Carlo (SGMCMC)
samplers to JAX. Inspired by [optax](https://github.com/deepmind/optax),
**JaxSGMC** is built on a modular concept to increase reusability and
accelerate research of new SGMCMC solvers. Additionally, **JaxSGMC** aims to
promote probabilistic machine learning by removing obstacles in switching
from stochastic optimizers to SGMCMC samplers.

## Quickstart with solvers from ``alias.py``

To get started quickly using SGMCMC samplers, **JaxSGMC** provides some popular
pre-built samplers in [alias.py](jax_sgmc/alias.py):

- **SGLD (rms-prop)**: <https://arxiv.org/abs/1512.07666>
- **SGHMC**: <https://arxiv.org/abs/1402.4102>
- **reSGLD**: <https://arxiv.org/abs/2008.05367v3>
- **SGGMC**: <https://arxiv.org/abs/2102.01691>
- **AMAGOLD**: <https://arxiv.org/abs/2003.00193>
- **OBABO**: <https://arxiv.org/abs/2102.01691>

## Features

### Modular SGMCMC solvers

**JaxSGMC** aims to increase reusability of SGMCMC components via a toolbox of
helper functions and a modular concept:

![](https://raw.githubusercontent.com/tummfm/jax-sgmc/main/jax-sgmc-structure.svg)

In the simplest case of employing a pre-built sampler from
[alias.py](jax_sgmc/alias.py), the user only needs to provide the computational
model, consisting of functions for Prior and Likelihood.
Schedulers allow to change sampler properies over the course of the training.
Advanced users may build custom samplers from given components.

### Data Input / Output under ``jit``

**JaxSGMC** provides a toolbox to pass reference data to the computation
and save collected samples from the Markov chain.

By combining different data loader / collector classes and general wrappers it
is possible to read data from and save samples to different data types via the
mechanisms of JAX's Host-Callback module.
It is therefore also possible to access datasets bigger than the device memory.

Saving Data:
  - HDF5
  - Numpy ``.npz``

Loading Data:
  - HDF5
  - Numpy arrays
  - Tensorflow datasets
  
### Computing the stochastic potential

Stochastic Gradient MCMC requires the evaluation of a potential function for a
batch of data.
**JaxSGMC** allows to compute this potential from likelihoods accepting only
single observations and batches them automatically with sequential, parallel or
vectorized execution. 
Moreover, **JaxSGMC** supports passing a model state between the evaluations of
the likelihood function, which is saved corresponding to the samples, speeding 
up postprocessing.

## Installation

### Basic Setup

**JaxSGMC** can be installed via pip:

```shell
pip install jax-sgmc --upgrade
```

The above command installs **Jax for CPU**. To run **JaxSGMC on the GPU**,
the GPU version of JAX has to be installed.
Further information can be found here:
[Jax Installation Instructions](https://github.com/google/jax#installation)

### Additional Packages

Some parts of **JaxSGMC** require additional packages:

- Data Loading with tensorflow:
  ```shell
  pip install jax-sgmc[tensorflow] --upgrade
  ```
- Saving Samples in the HDF5-Format:
  ```shell
  pip install jax-sgmc[hdf5] --upgrade
  ```


### Installation from Source

For development purposes, **JaxSGMC** can be installed from source in
editable mode:

```shell
git clone git@github.com:tummfm/jax-sgmc.git
pip install -e .[test,docs]
```

This command additionally installs the requirements to run the tests:

```shell
pytest tests
```

And to build the documentation (e.g. in html):

```shell
make -C docs html
```

## Contributing

Contributions are always welcome! Please open a pull request to discuss the code
additions.
