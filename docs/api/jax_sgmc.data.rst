jax_sgmc.data
==============

jax_sgmc.data.core
-------------------

.. automodule:: jax_sgmc.data.core

Host Callback Wrappers
________________________

.. autofunction:: jax_sgmc.data.core.random_reference_data

.. autofunction:: jax_sgmc.data.core.full_reference_data

.. autoclass:: jax_sgmc.data.core.GetBatchFunction

   .. automethod:: __call__

.. autoclass:: jax_sgmc.data.core.FullDataMapFunction

    .. automethod:: __call__

.. autoclass:: jax_sgmc.data.core.MaskedMappedFunction

    .. automethod:: __call__

.. autoclass:: jax_sgmc.data.core.UnmaskedMappedFunction

    .. automethod:: __call__


States
_______

.. autoclass:: jax_sgmc.data.core.MiniBatchInformation

.. autoclass:: jax_sgmc.data.core.CacheState

Base Classes
_____________

.. autoclass:: jax_sgmc.data.core.DataLoader
   :members:

.. autoclass:: jax_sgmc.data.core.DeviceDataLoader
   :members:

.. autoclass:: jax_sgmc.data.core.HostDataLoader
   :members:

Utility Functions
__________________

.. autosummary::
   :toctree: _autosummary

    tree_index
    tree_dtype_struct

jax_sgmc.data.numpy_loader
---------------------------

.. automodule:: jax_sgmc.data.numpy_loader

.. autoclass:: jax_sgmc.data.numpy_loader.NumpyBase
   :members:

.. autoclass:: jax_sgmc.data.numpy_loader.NumpyDataLoader
   :members:

.. autoclass:: jax_sgmc.data.numpy_loader.DeviceNumpyDataLoader
   :members:

jax_sgmc.data.tensorflow_loader
--------------------------------

.. automodule:: jax_sgmc.data.tensorflow_loader

.. autoclass:: jax_sgmc.data.tensorflow_loader.TensorflowDataLoader
   :members:
