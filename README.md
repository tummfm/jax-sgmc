# Modular Stochastic Gradient MCMC for Jax

[![CI](https://github.com/tummfm/jax-sgmc/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/tummfm/jax-sgmc/actions/workflows/ci.yml)

**[Introduction](#introduction) | [Content](#content) | [Features](#features) | 
[Implemented Solvers](#quickstart-with-solvers-from-aliaspy) | 
[Installation](#installation) | [Contributing](#contributing)**

## Introduction

## Content

- **[Introduction](#introduction)**
- **[Content](#content)**
- **[Features](#features)**
- **[Implemented Solvers](#quickstart-with-solvers-from-aliaspy)**
- **[Installation](#installation)** 
- **[Contributing](#contributing)**


## Features

### Modular SGMCMC solvers

**JaxSGMC** tries to simplify the implementation of new solvers by
providing a toolbox of helper functions and a modular concept:

![Structure of JaxSGMC](/jax-sgmc-structure.svg)

### Data Input / Output under ``jit``

**JaxSGMC** provides a toolbox to pass reference data to the computation
and save collected samples from the chain.

By the combination of different
data loader / collector classes and general wrappers it is possible to read data
from and save samples to different data types via the mechanisms of Jax's
Host-Callback module.
It is therefore even possible to access datasets bigger than the device memory.

Saving Data:
  - HDF5
  - Numpy ``.npz``

Loading Data:
  - HDF5
  - Numpy arrays
  - Tensorflow datasets
  
### Compute stochastic potential from different likelihoods

Stochastic Gradient MCMC requires the evaluation of a potential function for a
batch of data.
**JaxSGMC** allows to compute this potential from likelihoods accepting only
single observations and batches them automatically with sequential, parallel or
vectorized execution. 
Moreover, **JaxSGMC** supports passing a model state between the evaluations of
the likelihood function, which is saved corresponding to the samples, speeding 
up the postprocessing of the results.

## Quickstart with solvers from ``alias.py``

To get started quickly, some popular solvers are already implemented in
**JaxSGMC** and can be found in [alias.py](jax_sgmc/alias.py):

- **SGLD (rms-prop)**: <https://arxiv.org/abs/1512.07666>
- **SGHMC**: <https://arxiv.org/abs/1402.4102>
- **reSGLD**: <https://arxiv.org/abs/2008.05367v3>
- **SGGMC**: <https://arxiv.org/abs/2102.01691>
- **AMAGOLD**: <https://arxiv.org/abs/2003.00193>
- **OBABO**: <https://arxiv.org/abs/2102.01691>


## Installation

### Basic Setup

**JaxSGMC** can be installed with pip:

```shell
pip install jax-sgmc --upgrade
```

The above command installs **Jax for CPU**.

To be able to run **JaxSGMC on the GPU**, a special version of Jax has to be
installed. Further information can be found here:

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
