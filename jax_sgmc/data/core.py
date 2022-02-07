# Copyright 2021 Multiscale Modeling of Fluid Materials, TU Munich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data input in jit-compiled functions

Big Data but limited device memory disallows to store all reference data on the
computing device. With the following functions, data mini-batches of data can be
requested just as the data would be fully loaded on the device and thus enables
to jit-compile or vmap the entiere function.

In the background, a cache of mini-batches is sequentially requested by the
Host Callback Wrappers from the Data Loaders and loaded on the device via the
:mod:`jax_sgmc.util.host_callback` module.

Random Data Access
--------------------

Get a batch of randomly selected observations.

Numpy Data Loader
__________________

The numpy data loader is easy to use if the whole dataset fits into RAM and is
already present as numpy-arrays.

.. doctest::

  >>> import numpy as onp
  >>> from jax_sgmc import data

First we set up the dataset. This is very simply, as each array can be asigned
as a keyword argument to the dataloader. The keywords of the single arrays form
the keys of the pytree-dict, bundling all observations.

  >>> # The arrays must have the same length algong the first dimension,
  >>> # corresponding to the total observation count
  >>> x = onp.arange(10)
  >>> y = onp.zeros((10, 2))
  >>>
  >>> data_loader = data.NumpyDataLoader(x_r=x, y_r=y)

Some models needs to now the shape and dtype of the reference data. Therefore,
a all-zero batch can be drawn from the dataloader.

  >>> print(data_loader.initializer_batch(3))
  {'x_r': DeviceArray([0, 0, 0], dtype=int32), 'y_r': DeviceArray([[0., 0.],
               [0., 0.],
               [0., 0.]], dtype=float32)}

The host callback wrappers cache some data in the device memory to reduce the
number of calls to the host. The cache size equals the number of batches stored
on the device. A bigger cache size is more effective in computation time, but
has an increased device memory consumption.

  >>> rd_init, rd_batch = data.random_reference_data(data_loader, 100, 2)

Some data loaders, such as the Numpy Data Loader, accept keyword arguments in
the init function to determnine the starting points of the chains.

  >>> rd_state = rd_init(seed=0)
  >>> new_state, (rd_batch, info) = rd_batch(rd_state, information=True)
  >>> print(rd_batch)
  {'x_r': DeviceArray([9, 7], dtype=int32), 'y_r': DeviceArray([[0., 0.],
               [0., 0.]], dtype=float32)}
  >>> # If necessary, information about the total sample count can be passed
  >>> print(info)
  MiniBatchInformation(observation_count=10, batch_size=2)


Tensorflow Data Loader
_______________________

The tensorflow data loader is a great choice for many standard datasets
available on tensorflow_datasets.

.. doctest::

  >>> import tensorflow_datasets as tfds
  >>> from jax import tree_util
  >>> from jax_sgmc import data
  >>>
  >>> # Helper function to look at the data provided
  >>> def show_data(data):
  ...   for key, item in data.items():
  ...     print(f"{key} with shape {item.shape} and dtype {item.dtype}")

The pipeline returned by tfds load can be directly passet to the data loader.
However, not all tensorflow data types can be transformed to jax data types, for
eample the feature 'id', which is a string. Those keys can be simply excluded
by passing the keyword argument `exclude_keys`.

  >>> # The data pipeline can be used directly
  >>> pipeline, info = tfds.load("cifar10", split="train", with_info=True)
  >>> print(info.features)
  FeaturesDict({
      'id': Text(shape=(), dtype=tf.string),
      'image': Image(shape=(32, 32, 3), dtype=tf.uint8),
      'label': ClassLabel(shape=(), dtype=tf.int64, num_classes=10),
  })
  >>>
  >>> data_loader = data.TensorflowDataLoader(pipeline, shuffle_cache=10, exclude_keys=['id'])
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

  >>> data_init, data_batch = data.random_reference_data(data_loader, 100, 1000)
  >>>
  >>> init_state = data_init()
  >>> new_state, batch = data_batch(init_state)
  >>> show_data(batch)
  image with shape (1000, 32, 32, 3) and dtype uint8
  label with shape (1000,) and dtype int32

Mapping over Full Dataset
--------------------------

It is also possible to map a function over the complete dataset provided by a
data loader. In each iteration, the function is mapped over a batch of data to
speed up the calculation but limit the memory consumption.

In this toy example, the dataset consits of the potential bases
:math:`\mathcal{D} = \left\{i \mid i = 0, \ldots, 10 \\right\}`. In a scan loop,
the sum of the potentials with given exponents is calculated:

.. math::

  f_e = \sum_{i=0}^{9}d_i^e \mid d_i \in \mathcal{D}, k = 0,\ldots, 2.

.. doctest::

  >>> from functools import partial
  >>> import jax.numpy as jnp
  >>> from jax.lax import scan
  >>> from jax_sgmc import data

First, the data loader must be set up. The mini batch size is not required to
truly divide the total observation count. This is realized by filling up the
last batch with some values, which are sorted out either automatically or
directly by the user with a provided mask.

  >>> base = jnp.arange(10)
  >>>
  >>> data_loader = data.NumpyDataLoader(base=base)

The mask is an boolean array with `True` if the value is valid and `False` if it
is just a filler. If set to `maksing=False` (default), no positional argument
mask is expected in the function signature.

  >>> def sum_potentials(exp, data, mask, unused_state):
  ...   # Mask out the invalid samples (filler values, already mapped over)
  ...   sum = jnp.sum(mask * jnp.power(data['base'], exp))
  ...   return sum, unused_state
  >>>
  >>> init_fun, map_fun = data.full_reference_data(data_loader,
  ...                                              cached_batches_count=3,
  ...                                              mb_size=4)

The results per batch must be post-processed. If `masking=False`, a result for
each observation is returned. Therefore, using the masking option improves the
memory consumption.

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

"""

import abc

import warnings

from functools import partial
import itertools

from typing import Tuple, Any, Callable, List, Union, NamedTuple, Dict, Optional

import jax
from jax import tree_util, lax, random
import jax.numpy as jnp
from jax.experimental import host_callback as hcb

import numpy as onp

# Tensorflow is only required if the tensorflow dataLoader ist used.
try:
  from tensorflow import data as tfd
  import tensorflow_datasets as tfds
  TFDataSet = tfd.Dataset
except ModuleNotFoundError:
  TFDataSet = None
  tfds = None

from jax_sgmc.util import Array, stop_vmap

class MiniBatchInformation(NamedTuple):
  """Bundles all information about the reference data.

  Args:
    observation_count: Total number of observatins
    effective_observation_count: The number of observations without the
      discarded samples remaining after e.g. shuffling.
    mini_batch: List of tuples, tuples consist of ``(observations, parameters)``

  """
  observation_count: Array
  mask: Array
  batch_size: Array


# Todo: Rework occurences
mini_batch_information = MiniBatchInformation

PyTree = Any
MiniBatch = Union[Tuple[PyTree],
                  Tuple[PyTree, mini_batch_information],
                  Tuple[PyTree, mini_batch_information, Array]]
RandomBatch = Tuple[Callable, Callable]
OrderedBatch = Tuple[Callable, Callable]
PyTree = Any

# Definition of the data loader class

# Todo: Register DataLoader pipes with (optional) the parent chain JaxUUID.

class DataLoader(metaclass=abc.ABCMeta):
  """Abstract class to define required methods of a DataLoader.

  This class defines common methods of a DataLoader, such as returning an all-
  zero batch with correct shape to initialize the model.
  """

  @property
  @abc.abstractmethod
  def static_information(self) -> Dict:
    """Information about the dataset such as the total observation count. """

  @property
  @abc.abstractmethod
  def _format(self):
    """dtype and shape of a mini-batch. """

  def initializer_batch(self, mb_size: int = None) -> PyTree:
    """Returns a zero-like mini-batch.

    Args:
      mb_size: Number of observations in a batch. If ``None``, the returned
        pytree has the shape of a single observation.

    """
    obs_format = self._format

    # Append the cache size to the batch_format
    def append_cache_size(leaf):
      if mb_size is None:
        new_shape = tuple(int(s) for s in leaf.shape)
      else:
        new_shape = tuple(int(s) for s in
                          itertools.chain([mb_size], leaf.shape))
      return jnp.zeros(
        dtype=leaf.dtype,
        shape=new_shape
      )

    batch = tree_util.tree_map(append_cache_size, obs_format)
    return batch


class DeviceDataLoader(DataLoader, metaclass=abc.ABCMeta):
  """Abstract class to define required methods of a DeviceDataLoader.

  A class implementing the data loader must have the functionality to return the
  complete dataset as a dictionary of arrays.
  """

  @abc.abstractmethod
  def init_random_data(self, *args, **kwargs) -> PyTree:
    """Initializes the state necessary to randomly draw data. """

  @abc.abstractmethod
  def get_random_data(self,
                      state,
                      batch_size
                      ) ->Tuple[PyTree, Tuple[PyTree, mini_batch_information]]:
    """Returns a random batch of the data.

    This function must be jit-able and free of side effects.

    """

  @abc.abstractmethod
  def get_full_data(self) -> Dict:
    """Returns the whole dataset as dictionary of arrays."""

class HostDataLoader(DataLoader, metaclass=abc.ABCMeta):
  """Abstract class to define required methods of a HostDataLoader.

  A class implementing the data loader must have the functionality to load data
  from storage in an ordered and a random fashion.
  """

  def __init__(self, mb_size: int = None, data_collector: Any = None):
    if mb_size is not None:
      assert False, "Set the batch size when registering chain. "
    if data_collector is not None:
      # Allows the data collector to save and restore the dataloader state
      data_collector.register_data_loader(self)

  # Todo: Add a checkpoint function returning the current state
  #       def checkpoint(self, ...):

  def save_state(self, chain_id: int):
    """Returns all necessary information to restore the dataloader state.

    Args:
      chain_id: Each chain can be checkpointed independently.

    Returns:
      Returns necessary information to restore the state of the chain via
      :func:`load_state`.

    """
    raise NotImplementedError("This method must be overwritten to allow "
                              "checkpointing of the data loader.")

  def load_state(self, chain_id: int, data):
    """Restores dataloader state from previously computed checkpoint.

    Args:
      chain_id: The chain to restore the state.
      data: Data from :func:`save_state` to restore state of the chain.

    """
    raise NotImplementedError("This method must be overwritten to allow "
                              "checkpointing of the data loader.")

  @abc.abstractmethod
  def register_random_pipeline(self,
                               cache_size: int = 1,
                               mb_size: int = None,
                               **kwargs
                               ) -> int:
    """Register a new chain which draw samples randomly.

    Args:
      cache_size: The number of drawn batches.
      mb_size: The number of observations per batch.
      seed: Set the random seed to start the chain at a well defined state.

    Returns:
      Returns the id of the new chain.

    """

  @abc.abstractmethod
  def register_ordered_pipeline(self,
                                cache_size: int = 1,
                                mb_size: int = None,
                                **kwargs
                                ) -> int:
    """Register a chain which assembles batches in an ordered manor.

    Args:
      cache_size: The number of drawn batches.
      mb_size: The number of observations per batch.

    Returns:
      Returns the id of the new chain.

    """


  @abc.abstractmethod
  def get_batches(self, chain_id: int) -> Tuple[PyTree, Optional[Array]]:
    """Return batches from an ordered or random chain. """

  def batch_format(self,
                   cache_size: int,
                   mb_size: int,
                   ) -> PyTree:
    """Returns dtype and shape of cached mini-batches.

    Args:
      cache_size: number of cached mini-batches

    Returns:
      Returns a pytree with the same tree structure as the random data cache but
      with ``jax.ShapedDtypeStruct``` as leaves.
    """
    obs_format = self._format

    # Append the cache size to the batch_format
    def append_cache_size(leaf):
      new_shape = tuple(int(s) for s in
                        itertools.chain([cache_size, mb_size], leaf.shape))
      return jax.ShapeDtypeStruct(
        dtype=leaf.dtype,
        shape=new_shape
      )
    format = tree_util.tree_map(append_cache_size, obs_format)
    mb_info = MiniBatchInformation(
      observation_count=self.static_information["observation_count"],
      batch_size=mb_size)
    return format, mb_info

# Todo: Add variable to keept track of already returned batches to implement
#       masking

class CacheState(NamedTuple):
  """Caches several batches of randomly batched reference data.

  Args:
    cached_batches: An array of mini-batches
    cached_batches_count: Number of cached mini-batches. Equals the first
      dimension of the cached batches
    current_line: Marks the next batch to be returned.
    chain_id: Identifier of the chain to associate random state
    state: Additional information
    valid: Array containing information about the validity of individual samples
  """
  cached_batches: PyTree = None
  cached_batches_count: Array = None
  current_line: Array = None
  chain_id: Array = None
  state: PyTree = None
  valid: Array = None

random_data_state = CacheState


def random_reference_data(data_loader: DeviceDataLoader,
                          cached_batches_count: int,
                          mb_size: int,
                          drop_remainder: bool = True
                          ) -> RandomBatch:
  """Initializes reference data access in jit-compiled functions.

  Randomly draw batches from a given dataset on the host or the device.

  Args:
    data_loader: Reads data from storage.
    cached_batches_count: Number of batches in the cache. A larger number is
      faster, but requires more memory.
    mb_size: Size of the data batch.
    drop_remainder: When shuffling, i.e. drawing each sample once before drawing
      a sample twice, the last samples after every shuffle are discarded. If
      false, a mask is returned to mark invalid samples.

  Returns:
    Returns a tuple of functions to initialize a new reference data state and
    get a minibatch from the reference data state

  """
  # Check batch size is not bigger than total observation count
  observation_count = data_loader.static_information["observation_count"]
  if observation_count > mb_size:
    raise ValueError(f"Batch size cannot be bigger than the number of total "
                     f"observations. Got {observation_count} and {mb_size}.")

  if isinstance(data_loader, HostDataLoader):
    return _random_reference_data_host(
      data_loader, cached_batches_count, mb_size, drop_remainder)
  elif isinstance(data_loader, DeviceDataLoader):
    if not cached_batches_count == 1:
      raise ValueError(f"No caching on device.")
    return _random_reference_data_device(
      data_loader, mb_size)
  else:
    raise TypeError("The DataLoader must inherit from HostDataLoader or "
                    "DeviceDataLoader")

def full_reference_data(data_loader: DataLoader,
                        cached_batches_count: int = 100,
                        mb_size: int = None
                        ) -> OrderedBatch:
  """Initializes reference data access in jit-compiled functions.

  Map a function batch-wise over a dataset on the host or the device.

  Args:
    data_loader: Reads data from storage.
    cached_batches_count: Number of batches in the cache. A larger number is
      faster, but requires more memory.
        mb_size: Size of the data batch.

  Returns:
    Returns a tuple of functions to initialize a new reference data state and
    get a minibatch from the reference data state

  """
  # Check batch size is not bigger than total observation count
  observation_count = data_loader.static_information["observation_count"]
  if observation_count > mb_size:
    raise ValueError(f"Batch size cannot be bigger than the number of total "
                     f"observations. Got {observation_count} and {mb_size}.")

  if isinstance(data_loader, HostDataLoader):
    init_fn, (_batch_fn, mb_information) = _full_reference_data_host(
      data_loader, cached_batches_count, mb_size)
  elif isinstance(data_loader, DeviceDataLoader):
    init_fn, (_batch_fn, mb_information) = _full_reference_data_device(
      data_loader, cached_batches_count)
  else:
    raise TypeError("The DataLoader must inherit from HostDataLoader or "
                    "DeviceDataLoader")

  num_iterations = int(onp.ceil(
    mb_information.observation_count / mb_information.batch_size))

  batch_size = mb_information.batch_size

  def _uninitialized_body_fn(fun,
                             state,
                             iteration,
                             information=False,
                             masking=False):
    # The mask has is 1 if the observation is valid and 0 otherwise. This is
    # necessary to ensure, that fun is always called with the same tree shape.
    observations = iteration * batch_size + jnp.arange(batch_size)
    mask = observations < mb_information.observation_count

    data_state, fun_state = state
    data_state, batch = _batch_fn(data_state, information=information)

    if masking:
      result, fun_state = fun(batch, mask, fun_state)
    else:
      result, fun_state = fun(batch, fun_state)

    return (data_state, fun_state), result

  def batch_scan(fun: Callable[[PyTree, Array, PyTree], Tuple[PyTree, PyTree]],
                 data_state: CacheState,
                 carry: PyTree,
                 masking: bool = False,
                 information: bool = False):
    """Map the function over all data and return the results.

    Args:
      fun: Function accepting a batch of data, a mask and a state.
      data_state: Reference data state.
      carry: A argument that is carried over between iterations
      masking: If set to true, the mapped function is called with a positional
        argument mask and expected to return the results with a reduced dimension.
        Setting to true changes the signature from `fun(data, carry)` to
        `fun(data, mask, carry)`.
      information: Provide the minibatch information in addition to the data
        batch.

    """
    _body_fn = partial(_uninitialized_body_fn,
                       fun,
                       information=information,
                       masking=masking)
    (data_state, carry), results = lax.scan(
      _body_fn, (data_state, carry), onp.arange(num_iterations))

    if masking:
      true_results = results
    else:
      # The results must be concatenated
      concat_results = tree_util.tree_map(
        partial(jnp.concatenate, axis=0),
        results)
      # Invalid results (fillers) must be thrown away
      true_results = tree_util.tree_map(
        lambda leaf: leaf[0:mb_information.observation_count],
        concat_results)

    return data_state, (true_results, carry)

  return init_fn, batch_scan


def tree_dtype_struct(pytree: PyTree):
  """Returns a tree with leaves only representing shape and type."""
  @partial(partial, tree_util.tree_map)
  def concrete_to_shape_struct(leaf):
    shape_struct = jax.ShapeDtypeStruct(
      dtype=leaf.dtype,
      shape=leaf.shape)
    return shape_struct
  return concrete_to_shape_struct(pytree)

def tree_index(pytree: PyTree, index):
  """Indexes the leaves of the tree in the first dimension.

  Args:
    PyTree: Tree to index with array-like leaves
    index: Selects which slice to return

  Returns:
    Returns a tree with the same structure as pytree, but the leaves have a
    dimension reduced by 1.

  """
  @partial(partial, tree_util.tree_map)
  def split_tree_imp(leaf):
    if leaf.ndim == 1:
      return leaf[index]
    else:
      return leaf[index, ::]
  return split_tree_imp(pytree)


# Callback is independent of assembling of the batches
def _hcb_wrapper(data_loader: HostDataLoader,
                 cached_batches_count: int,
                 mb_size: int):
  # These are helper function which keep a reference to the stateful data object
  # and can be called via the host_callback.call function
  # The format of the mini batch is static.

  hcb_format, mb_information = data_loader.batch_format(
    cached_batches_count, mb_size=mb_size)
  mask_shape = (cached_batches_count, mb_size)

  # The definition requires passing an argument to the host function. The shape
  # of the returned data must be known before the first call. The chain id
  # determines whether the data is collected randomly or sequentially.

  def get_data(chain_id):
    new_data, *opt_args = data_loader.get_batches(chain_id)
    if len(opt_args) == 0:
      # Assume all samples to be valid
      mask = jnp.ones(mask_shape, dtype=jnp.bool_)
    elif len(opt_args) == 1 and opt_args.shape == mask_shape:
      mask = opt_args[0]
    else:
      raise TypeError(f"Expecting only on optional return value (mask).")
    return new_data, mask

  def _new_cache_fn(state: CacheState) -> CacheState:
    """This function is called if the cache must be refreshed."""
    new_data, masks = hcb.call(
      get_data,
      state.chain_id,
      result_shape=(hcb_format, mask_shape))
    new_state = CacheState(
      cached_batches_count=state.cached_batches_count,
      cached_batches=new_data,
      current_line=jnp.array(0),
      chain_id=state.chain_id,
      valid=masks)
    return new_state

  def _old_cache_fn(state: CacheState) -> CacheState:
    """This function is called if the cache must not be refreshed."""
    return state

  # Necessary, because cond is replaced by select under vmap, but the cond
  # branches have side effects.
  @stop_vmap.stop_vmap
  def _data_state_helper(data_state):
    return lax.cond(data_state.current_line == data_state.cached_batches_count,
                    _new_cache_fn,
                    _old_cache_fn,
                    data_state)

  def batch_fn(data_state: CacheState,
               information: bool = False
               ) -> Union[Tuple[CacheState, MiniBatch],
                          Tuple[CacheState,
                                Tuple[MiniBatch, mini_batch_information]]]:
    """Draws a new random batch (hides data transfer between devices).

    Args:
      data_state: State with cached samples
      information: Whether to return batch information

    Returns:
      Returns the new data state and the next batch. Optionally also also a
      struct containing information about the batch can be returned.

    """

    # Todo: Implement mask in mb_information.

    # Refresh the cache if necessary, after all cached batches have been used.
    data_state = _data_state_helper(data_state)
    current_line = jnp.mod(data_state.current_line,
                           data_state.cached_batches_count)

    # Read the current line from the cache and add the mask containing
    # information about the validity of the individual samples
    mini_batch = tree_index(data_state.cached_batches, current_line)
    mask = data_state.valid[current_line, :]

    current_line = current_line + 1

    new_state = CacheState(
      cached_batches=data_state.cached_batches,
      cached_batches_count=data_state.cached_batches_count,
      current_line=current_line,
      chain_id=data_state.chain_id,
      valid=mask)

    if information:
      return new_state, (mini_batch, mb_information)
    else:
      return new_state, mini_batch

  return batch_fn, mb_information


def _random_reference_data_host(data_loader: HostDataLoader,
                                cached_batches_count: int = 100,
                                mb_size: int = 1,
                                drop_remainder: bool = True
                                ) -> RandomBatch:
  """Random reference data access via host-callback. """
  # Warn if cached_batches are bigger than total dataset
  observation_count = data_loader.static_information["observation_count"]
  if observation_count < cached_batches_count * mb_size:
    warnings.warn("Cached batches are bigger than the total dataset. Consider "
                  "using a DeviceDataLoader.")

  batch_fn, _ = _hcb_wrapper(data_loader, cached_batches_count, mb_size)

  def init_fn(**kwargs) -> CacheState:
    # Pass the data loader the information about the number of cached
    # mini-batches. The data loader returns an unique id for reproducibility
    # Todo: Pass drop_remainder here -> how to deal with that is task of the
    #   data loader. It might have no effect, e.g in case of random drawing.
    chain_id = data_loader.register_random_pipeline(
      cached_batches_count,
      mb_size=mb_size,
      **kwargs)
    initial_state = data_loader.get_batches(chain_id)
    inital_cache_state = CacheState(
      cached_batches=initial_state,
      cached_batches_count=jnp.array(cached_batches_count),
      current_line=jnp.array(0),
      chain_id=jnp.array(chain_id))
    return inital_cache_state

  return init_fn, batch_fn


def _random_reference_data_device(data_loader: DeviceDataLoader,
                                  mb_size: int
                                  ) -> RandomBatch:
  """Random reference data on device. """

  def init_fn(*args, **kwargs) -> CacheState:
    state = data_loader.init_random_data(*args, **kwargs)
    return CacheState(state=state)

  def batch_fn(state: CacheState,
               information: bool = False
               ) -> Union[Tuple[CacheState, MiniBatch],
                          Tuple[CacheState,
                                Tuple[MiniBatch, mini_batch_information]]]:

    state, (batch, mb_info) = data_loader.get_random_data(
      state.state, batch_size=mb_size)
    new_state = CacheState(state=state)

    if information:
      return new_state, (batch, mb_info)
    else:
      return new_state, batch

  return init_fn, batch_fn

def _full_reference_data_host(data_loader: HostDataLoader,
                              cached_batches_count: int = 100,
                              mb_size: int = None
                              ) -> Tuple[Callable, Tuple[Callable, mini_batch_information]]:
  """Sequentially load batches of reference data via host-callback. """
  # Warn if cached_batches are bigger than total dataset
  observation_count = data_loader.static_information["observation_count"]
  if observation_count < cached_batches_count * mb_size:
    warnings.warn("Cached batches are bigger than the total dataset. Consider "
                  "using a DeviceDataLoader.")

  batch_fn = _hcb_wrapper(
    data_loader,
    cached_batches_count,
    mb_size)

  def init_fn(**kwargs) -> CacheState:
    # Pass the data loader the information about the number of cached
    # mini-batches. The data loader returns an unique id for reproducibility
    chain_id = data_loader.register_ordered_pipeline(
      cached_batches_count,
      mb_size=mb_size,
      **kwargs)
    initial_state = data_loader.get_batches(chain_id)
    inital_cache_state=CacheState(
      cached_batches=initial_state,
      cached_batches_count=jnp.array(cached_batches_count),
      current_line=jnp.array(0),
      chain_id=jnp.array(chain_id)
    )
    return inital_cache_state

  return init_fn, batch_fn


def _full_reference_data_device(data_loader: DeviceDataLoader,
                                mb_size: int = None
                                ) -> Tuple[Callable,
                                           Tuple[Callable,
                                                 mini_batch_information]]:
  """Batches the dataset on the device. """

  reference_data = data_loader.get_full_data()
  total_observations = data_loader.static_information["observation_count"]


  # The information about the batches need to be static.
  mb_info = mini_batch_information(
    observation_count=total_observations,
    batch_size=mb_size)

  def init_fn(offset: jnp.ndarray = 0):
    if offset >= total_observations:
      raise ValueError(f"The offset cannot be greater than the total "
                       f"observation count. Given {offset} and "
                       f"{total_observations}.")

    init_state = CacheState(current_line=offset)
    return init_state

  def batch_fn(data_state: CacheState,
               information: bool = False):
    indices = jnp.mod(jnp.arange(mb_size) + data_state.current_line,
                      total_observations)

    # Update the offset, where to start slicing in the next iteration.
    new_state = CacheState(
      current_line=jnp.mod(indices[-1] + 1, total_observations))

    selected_data = tree_index(reference_data, indices)

    if information:
      return new_state, (selected_data, mb_info)
    else:
      return new_state, selected_data

  return init_fn, (batch_fn, mb_info)
