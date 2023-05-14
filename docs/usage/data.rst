Data Loading
=============

Steps for Setting up a Data Chain
---------------------------------

Access of (random) data in **JaxSGMC** consists of two steps:

- Setup Data Loader
- Setup Callback Wrappers

The Data Loaders determines where the data is stored and how it is passed
to the device (e. g. shuffled in epochs).

The Callback Wrappers requests new batches from the Data Loader and passes them
to the device via Jax's Host-Callback module. Therefore, only a subset of the
data is stored in the device memory.

The combination of a Data Loader and Callback Wrappers determines how the data is
passed to the computation. Therefore, this guide presents different methods of
data access with ``NumpyDataLoader`` and ``TensorflowDataLoader``.

.. note::
    When using multiple Data Loaders sequentially in a single script, the
    release function should be called after the Callback Wrapper has been used in
    the last computation. After this, the reference to the Data Loader will be
    discarded and the Data Loader can be deleted.

Important Notes
----------------

Getting shape and dtype of the data
____________________________________

Some models need to know the shape and dtype of the reference data. Therefore,
an all-zero batch can be drawn from every Data Loader.

  ::

    print(data_loader.initializer_batch(3))
    {'x_r': Array([0, 0, 0], dtype=int32), 'y_r': Array([[0., 0.],
                 [0., 0.],
                 [0., 0.]], dtype=float32)}

If no batch size is specified a single observation is returned (all leaves'
shapes are reduced by the first axis).

  ::

    print(data_loader.initializer_batch())
    {'x_r': Array(0, dtype=int32), 'y_r': Array([0., 0.], dtype=float32)}

Combining ``pmap`` and ``jit``
______________________________

.. warning::
   Jit-compiling a function including pmap requires adjustments of the Callback
   Wrapper functions, which can lead to memory leaks if not done correctly.

   Additionally, combining ``jit`` and ``pmap`` can lead to inefficient data
   movement. See `<https://github.com/google/jax/issues/2926>`_.

When jitting a function f including a pmap function g, also the parts of f
outside of g are run on all involved devices. This causes all devices to request
the same cache state (verified by a token) from a single chain.

For example, if g is pmapped to 5 devices, f is also going to run on 5 devices
and hence 5 times the same cache state is requested from a chain.

**JaxSGMC** resolved this issue by caching all requested states on the host for a
specified number of requests.

In the above example, the Callback Wrapper used in f should be called like this:

  ::

  ... = full_data_map(to_map_fn, data_state, carry, device_count=5)


It is important to note that providing a device count larger than the actual
number of calling devices causes a memory leak, as all requested cache states
will remain on the host until the program has finished.

Numpy Data Loader
------------------

.. doctest::

  >>> import numpy as onp
  >>> from jax_sgmc import data
  >>> from jax_sgmc.data.numpy_loader import NumpyDataLoader

First we set up the dataset. This is very simple, as each array can be assigned
as a keyword argument to the dataloader. The keywords of the single arrays form
the keys of the pytree-dict, bundling all observations.

  >>> # The arrays must have the same length along the first dimension,
  >>> # corresponding to the total observation count
  >>> x = onp.arange(10)
  >>> y = onp.zeros((10, 2))
  >>>
  >>> data_loader = NumpyDataLoader(x_r=x, y_r=y)

The host callback wrappers cache some data in the device memory to reduce the
number of calls to the host. The cache size equals the number of batches stored
on the device. A bigger cache size is more effective in computation time, but
has an increased device memory consumption.

  >>> rd_init, rd_batch, _ = data.random_reference_data(data_loader, 100, 2)

The ``NumpyDataLoader`` accepts keyword arguments in
the init function to determine the starting points of the chains.

  >>> rd_state = rd_init(seed=0)
  >>> new_state, (rd_batch, info) = rd_batch(rd_state, information=True)
  >>> print(rd_batch)
  {'x_r': Array([8, 9], dtype=int32), 'y_r': Array([[0., 0.],
         [0., 0.]], dtype=float32)}
  >>> # If necessary, information about the total sample count can be passed
  >>> print(info)
  MiniBatchInformation(observation_count=10, mask=Array([ True,  True], dtype=bool), batch_size=2)


Random Data Access
___________________

The ``NumpyDataLoader`` provides three different methods to randomly select
observations:

- Independent draw (default): Draw from all samples with replacement.
- Shuffling: Draw from all samples without replacement and immediately reshuffle
  if all samples have been drawn.
- Shuffling in epochs: Draw from all samples without replacement and return mask
  to mark invalid samples at the end of the epoch.

This is illustrated at a small toy-dataset, where the observation count is not a
multiple of the batch size:

.. doctest::

  >>> import numpy as onp
  >>> from jax_sgmc import data
  >>> from jax_sgmc.data.numpy_loader import NumpyDataLoader

  >>> x = onp.arange(10)
  >>> data_loader = NumpyDataLoader(x=x)
  >>> init_fn, batch_fn, _ = data.random_reference_data(data_loader, 2, 3)

The preferred method has to be passed when initializing the different chains:

  >>> random_chain = init_fn()
  >>> shuffle_chain = init_fn(shuffle=True)
  >>> epoch_chain = init_fn(shuffle=True, in_epochs=True)

In the fourth draw, the epoch chain should return a mask with invalid samples:

  >>> def eval_fn(chain):
  ...   for _ in range(4):
  ...     chain, batch = batch_fn(chain, information=True)
  ...   print(batch)
  >>>
  >>> eval_fn(random_chain)
  ({'x': Array([4, 6, 6], dtype=int32)}, MiniBatchInformation(observation_count=10, mask=Array([ True,  True,  True], dtype=bool), batch_size=3))
  >>> eval_fn(shuffle_chain)
  ({'x': Array([0, 4, 7], dtype=int32)}, MiniBatchInformation(observation_count=10, mask=Array([ True,  True,  True], dtype=bool), batch_size=3))
  >>> eval_fn(epoch_chain)
  ({'x': Array([5, 0, 0], dtype=int32)}, MiniBatchInformation(observation_count=10, mask=Array([ True, False, False], dtype=bool), batch_size=3))


Mapping over Full Dataset
__________________________

It is also possible to map a function over the complete dataset provided by a
``DataLoader``. In each iteration, the function is mapped over a batch of data to
speed up the calculation but limit the memory consumption.

In this toy example, the dataset consists of the potential bases
:math:`\mathcal{D} = \left\{i \mid i = 0, \ldots, 10 \right\}`. In a scan loop,
the sum of the potentials with given exponents is calculated:

.. math::

  f_e = \sum_{i=0}^{9}d_i^e \mid d_i \in \mathcal{D}, k = 0,\ldots, 2.

.. doctest::

  >>> from functools import partial
  >>> import jax.numpy as jnp
  >>> from jax.lax import scan
  >>> from jax_sgmc import data
  >>> from jax_sgmc.data.numpy_loader import NumpyDataLoader

First, the data loader must be set up. The mini batch size is not required to
truly divide the total observation count. This is realized by filling up the
last batch with some values, which are sorted out either automatically or
directly by the user with the provided mask.

  >>> base = jnp.arange(10)
  >>>
  >>> data_loader = NumpyDataLoader(base=base)

The mask is an boolean array with ``True`` if the value is valid and ``False``
if it is just a filler.
If set to ``masking=False`` (default), no positional argument mask is expected
in the function signature.

  >>> def sum_potentials(exp, data, mask, unused_state):
  ...   # Mask out the invalid samples (filler values, already mapped over)
  ...   sum = jnp.sum(mask * jnp.power(data['base'], exp))
  ...   return sum, unused_state
  >>>
  >>> init_fun, map_fun, _ = data.full_reference_data(data_loader,
  ...                                                 cached_batches_count=3,
  ...                                                 mb_size=4)

The results per batch must be post-processed. If ``masking=False``, a result for
each observation is returned. Therefore, using the masking option improves the
memory consumption.

  >>> # The exponential value is fixed during the mapping, therefore add it via
  >>> # functools.partial to the mapped function.
  >>> map_results = map_fun(partial(sum_potentials, 2),
  ...                       init_fun(),
  ...                       None,
  ...                       masking=True)
  >>>
  >>> data_state, (batch_sums, unused_state) = map_results
  >>>
  >>> # As we used the masking, a single result for each batch is returned.
  >>> # Now we need to postprocess those results, in this case by summing, to
  >>> # get the true result.
  >>> summed_result = jnp.sum(batch_sums)
  >>> print(f"Result: {summed_result : d}")
  Result:  285

The full data map can be used in ``jit``-compiled functions, e. g. in a scan loop,
such that it is possible to compute the results for multiple exponents in a
``lax.scan``-loop.

  >>> # Calculate for multiple exponents:
  >>> def body_fun(data_state, exp):
  ...   map_results = map_fun(partial(sum_potentials, exp), data_state, None, masking=True)
  ...   # Currently, we only summed over each mini-batch but not the whole
  ...   # dataset.
  ...   data_state, (batch_sums, unused_state) = map_results
  ...   return data_state, (jnp.sum(batch_sums), unused_state)
  >>>
  >>> init_data_state = init_fun()
  >>> _, (result, _) = scan(body_fun, init_data_state, jnp.arange(3))
  >>> print(result)
  [ 10  45 285]

It is also possible to store the ``CacheStates`` in the host memory, such that
it is not necessary to carry the ``data_state`` through all function calls.
The :func:`jax_sgmc.data.core.full_data_mapper` function does this, such that
its usage is a little bit simpler:

  >>> mapper_fn, release_fn = data.full_data_mapper(data_loader,
  ...                                               cached_batches_count=3,
  ...                                               mb_size=4)
  >>>
  >>> results, _ = mapper_fn(partial(sum_potentials, 2), None, masking=True)
  >>>
  >>> print(f"Result with exp = 2: {jnp.sum(results) : d}")
  Result with exp = 2:  285
  >>>
  >>> # Delete the reference to the Data Loader (optional)
  >>> release_fn()


Tensorflow Data Loader
-----------------------

Random Access
_______________________

The tensorflow data loader is a great choice for many standard datasets
available on tensorflow_datasets.

.. doctest::

  >>> import tensorflow_datasets as tfds
  >>> from jax import tree_util
  >>> from jax_sgmc import data
  >>> from jax_sgmc.data.tensorflow_loader import TensorflowDataLoader
  >>>
  >>> import contextlib
  >>> import io
  >>>
  >>> # Helper function to look at the data provided
  >>> def show_data(data):
  ...   for key, item in data.items():
  ...     print(f"{key} with shape {item.shape} and dtype {item.dtype}")

The pipeline returned by tfds load can be directly passed to the data loader.
However, not all tensorflow data types can be transformed to jax data types, for
example the feature 'id', which is a string. Those keys can be simply excluded
via the keyword argument `exclude_keys`.

  >>> # The data pipeline can be used directly
  >>> with contextlib.redirect_stdout(io.StringIO()):
  ...   pipeline, info = tfds.load("cifar10", split="train", with_info=True)
  >>> print(info.features)
  FeaturesDict({
      'id': Text(shape=(), dtype=string),
      'image': Image(shape=(32, 32, 3), dtype=uint8),
      'label': ClassLabel(shape=(), dtype=int64, num_classes=10),
  })
  >>>
  >>> data_loader = TensorflowDataLoader(pipeline, shuffle_cache=10, exclude_keys=['id'])
  >>>
  >>> # If the model needs data for initialization, an all zero batch can be
  >>> # drawn with the correct shapes and dtypes
  >>> show_data(data_loader.initializer_batch(mb_size=1000))
  image with shape (1000, 32, 32, 3) and dtype uint8
  label with shape (1000,) and dtype int32

The host callback wrappers cache some data in the device memory to reduce the
number of calls to the host. The cache size equals the number of batches stored
on the device. A bigger cache size is more effective in computation time, but
has an increased device memory consumption.

  >>> data_init, data_batch, _ = data.random_reference_data(data_loader, 100, 1000)
  >>>
  >>> init_state = data_init()
  >>> new_state, batch = data_batch(init_state)
  >>> show_data(batch)
  image with shape (1000, 32, 32, 3) and dtype uint8
  label with shape (1000,) and dtype int32
