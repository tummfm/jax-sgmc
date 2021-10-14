jax_sgmc.data
==============

.. automodule:: jax_sgmc.data


API
----

States
________

.. autoclass:: jax_sgmc.data.MiniBatchInformation
.. autoclass:: jax_sgmc.data.CacheState

Data Loaders
______________


.. autoclass:: jax_sgmc.data.DataLoader
    :members:

.. autoclass:: jax_sgmc.data.TensorflowDataLoader
    :members:

.. autoclass:: jax_sgmc.data.NumpyDataLoader
    :members:

Host Callback Wrappers
________________________

.. autofunction:: jax_sgmc.data.random_reference_data
.. autofunction:: jax_sgmc.data.full_reference_data

Utility Functions
__________________

.. autosummary::
   :toctree: _autosummary

    tree_index
    tree_dtype_struct

