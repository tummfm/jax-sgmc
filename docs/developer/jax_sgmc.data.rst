jax_sgmc.data
==============

Numpy Data Loader
------------------
.. automodule:: jax_sgmc.data.numpy_loader

Tensorflow Data Loader
-----------------------
.. automodule:: jax_sgmc.data.tensorflow_loader


API
----

.. automodule:: jax_sgmc.data.core

States
________

.. autoclass:: jax_sgmc.data.MiniBatchInformation
.. autoclass:: jax_sgmc.data.CacheState

Data Loaders
______________


.. autoclass:: jax_sgmc.data.DataLoader
    :members:

.. autoclass:: jax_sgmc.data.DeviceDataLoader

.. autoclass:: jax_sgmc.data.HostDataLoader

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

