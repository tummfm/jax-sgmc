jax_sgmc.io.rst
================

Overview
---------

.. automodule:: jax_sgmc.io


Data Collectors
----------------

Data Collector Interface
_________________________

.. autoclass:: jax_sgmc.io.DataCollector
    :members:

Collectors
___________

.. autoclass:: jax_sgmc.io.MemoryCollector
    :members:

.. autoclass:: jax_sgmc.io.HDF5Collector
    :members:


Saving Strategies
------------------


.. autofunction:: jax_sgmc.io.save

.. autofunction:: jax_sgmc.io.no_save


Pytree to Dict Transformation
------------------------------

.. autofunction:: jax_sgmc.io.pytree_to_dict

.. autofunction:: jax_sgmc.io.dict_to_pytree

.. autofunction:: jax_sgmc.io.pytree_dict_keys

Adding Tranformable Pytree-Nodes
_________________________________

By default, transformations are defined for some default types:

- list
- dict
- (named)tuple

Additionaly, transformation for the following optional libraries are
implemented:

- haiku._src.data_structures.FlatMapping

New rules can be implemented by:

.. autofunction:: jax_sgmc.io.register_dictionize_rule

