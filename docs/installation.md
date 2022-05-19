# Installation

## Basic Setup

**JaxSGMC** can be installed with pip:

```shell
pip install jax-sgmc --upgrade
```

The above command installs **Jax for CPU**.

To be able to run **JaxSGMC on the GPU**, a special version of Jax has to be
installed. Further information can be found in
[Jax Installation Instructions](https://github.com/google/jax#installation).

(additional_requirements)=
## Additional Packages

Some parts of **JaxSGMC** require additional packages:

- Data Loading with tensorflow:
  ```shell
  pip install jax-sgmc[tensorflow] --upgrade
  ```
- Saving Samples in the HDF5-Format:
  ```shell
  pip install jax-sgmc[hdf5] --upgrade
  ```


## Installation from Source

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
