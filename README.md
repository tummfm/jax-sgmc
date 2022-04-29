# Modular stochastic gradient MCMC for Jax

[![CI](https://github.com/tummfm/jax-sgmc/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/tummfm/jax-sgmc/actions/workflows/ci.yml)

**[Introduction](#introduction) | [Content](#content) | [Features](#features) | 
[Quickstart](#quickstart-aliaspy) | [Installation](#installation) |
[Contributing](#contributing)**

## Introduction

## Content

- **[Introduction](#introduction)**
- **[Content](#content)**
- **[Features](#features)**
- **[Implemted Solvers](#implemented-solvers)**
- **[Installation](#installation)** 
- **[Contributing](#contributing)**


## Features

### Data Loading under ``jit``

### Saving PyTrees under ``jit``

## Implemented Solvers

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
installed. Further information can be found in
[Jax Installation Instructions](https://github.com/google/jax#installation).

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

Contributions are very welcome, but take a look at 
[how to contribute](CONTRIBUTING.md).

