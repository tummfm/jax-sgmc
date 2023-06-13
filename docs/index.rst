Jax SGMC
========

JaxSGMC brings Stochastic Gradient Markov chain Monte Carlo (SGMCMC) samplers to JAX. Inspired by `optax <https://github.com/deepmind/optax>`_, JaxSGMC is built on a modular concept to increase reusability and accelerate research of new SGMCMC solvers. Additionally, JaxSGMC aims to promote probabilistic machine learning by removing obstacles in switching from stochastic optimizers to SGMCMC samplers.


To get started quickly using SGMCMC samplers, JaxSGMC provides some popular pre-built samplers in :doc:`./api/jax_sgmc.alias`:

- `SGLD (rms-prop) <https://arxiv.org/abs/1512.07666>`_
- `SGHMC <https://arxiv.org/abs/1402.4102>`_
- `reSGLD <https://arxiv.org/abs/2008.05367v3>`_
- `SGGMC <https://arxiv.org/abs/2102.01691>`_
- `AMAGOLD <https://arxiv.org/abs/2003.00193>`_
- `OBABO <https://arxiv.org/abs/2102.01691>`_


.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart


.. toctree::
   :maxdepth: 2
   :caption: Reference Documentation

   usage/data
   usage/potential
   usage/io
   usage/scheduler
   usage/sgld_rms

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics

   advanced/adaption
   advanced/scheduler


.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/cifar


.. toctree::
   :maxdepth: 3
   :caption: API Documentation

   api/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
