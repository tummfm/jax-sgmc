jax_sgmc.data
==============


.. automodule:: jax_sgmc.data


Data Loaders
------------


.. autoclass:: jax_sgmc.data.DataLoader
    :members:

.. autoclass:: jax_sgmc.data.TensorflowDataLoader
    :members:


Data Passing
-------------

Big Data but limited device memory disallows to store all reference data on the
computing device. With the following functions, data mini-batches of data can be
requested just as the data would be fully loaded on the device and thus enables
to jit-compile or vmap the entiere function. In the bachground, the data is
passed sequentially via host_callback.call().


.. autoclass:: jax_sgmc.data.mini_batch_format

Random mini-batch access
__________________________

.. autoclass:: jax_sgmc.data.random_data_state

.. autofunction:: jax_sgmc.data.random_reference_data




Ordered mini-batch access
___________________________

Comming soon....

Utility Functions
-------------------

.. autosummary::
   :toctree: _autosummary

    tree_index
    tree_dtype_struct

