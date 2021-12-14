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

from functools import partial
import itertools

from typing import Tuple, Any, Callable, List, Union, NamedTuple, Dict

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
    mini_batch: List of tuples, tuples consist of ``(observations, parameters)``

  """
  observation_count: Array
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

# Todo: State übergeben bei random batch statt chain id. Damit einfacher
#       checkpointing

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


class DeviceNumpyDataLoader(DeviceDataLoader):
  """Load complete dataset into memory from multiple numpy arrays.

  This data loader supports checkpointing, starting chains from a well defined
  state and true random access.

  The pipeline can be constructed directly from numpy arrays:

  .. doctest::

    >>> import numpy as onp
    >>> from jax_sgmc import data
    >>>
    >>> x, y = onp.arange(10), onp.zeros((10, 4, 3))
    >>>
    >>> data_loader = data.DeviceNumpyDataLoader(name_for_x=x, name_for_y=y)
    >>>
    >>> zero_batch = data_loader.initializer_batch(4)
    >>> for key, value in zero_batch.items():
    ...   print(f"{key}: shape={value.shape}, dtype={value.dtype}")
    name_for_x: shape=(4,), dtype=int32
    name_for_y: shape=(4, 4, 3), dtype=float32

  Args:
    reference_data: Each kwarg-pair is an entry in the returned data-dict.

  """

  def __init__(self, **reference_data):
    super().__init__()

    observation_counts = []
    self._reference_data = dict()
    for name, array in reference_data.items():
      observation_counts.append(len(array))
      # Transform to jax arrays
      self._reference_data[name] = jnp.array(array)

    # Check same number of observations
    if onp.any(onp.array(observation_counts) != observation_counts[0]):
      raise TypeError("All reference_data arrays must have the same length in"
                      " the first dimension.")

    self._observation_count = observation_counts[0]

  def init_random_data(self, *args, **kwargs) -> PyTree:
    del args
    key = kwargs.get("key", random.PRNGKey(0))
    return key

  def get_random_data(self,
                      state,
                      batch_size
                      ) ->Tuple[PyTree, Tuple[PyTree, mini_batch_information]]:
    if self._observation_count < batch_size:
      raise ValueError(f"The batch size ({batch_size}) cannot be greater than "
                       f"the observation count ({self._observation_count}).")

    key, split = random.split(state)
    selection_indices = random.randint(
      split, shape=(batch_size,), minval=0, maxval=self._observation_count)

    selected_observations = tree_index(self._reference_data, selection_indices)
    info = mini_batch_information(observation_count=self._observation_count,
                                  batch_size=batch_size)

    return key, (selected_observations, info)

  def get_full_data(self) -> Dict:
    return self._reference_data

  @property
  def static_information(self) -> Dict:
    information = {
      "observation_count": self._observation_count
    }
    return information

  @property
  def _format(self):
    """Returns shape and dtype of a single observation. """
    mb_format = dict()
    for name, array in self._reference_data.items():
      # Get the format and dtype of the data
      mb_format[name] = jax.ShapeDtypeStruct(
        dtype=self._reference_data[name].dtype,
        shape=tuple(int(s) for s in array.shape[1:]))
    return mb_format


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
  def get_batches(self, chain_id: int) -> PyTree:
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


class TensorflowDataLoader(HostDataLoader):
  """Load data from a tensorflow dataset object.

  The tensorflow datasets package provides a high number of ready to go
  datasets, which can be provided directly to the Tensorflow Data Loader.

  .. doctest::

    >>> import tensorflow_datasets as tdf
    >>> from jax_sgmc import data
    >>>
    >>> pipeline = tfds.load("cifar10", split="train")
    >>> data_loader = data.TensorflowDataLoader(pipeline, shuffle_cache=100, exclude_keys=['id'])

  Args:
    pipeline: A tensorflow data pipeline, which can be obtained from the
      tensorflow dataset package

  """

  def __init__(self,
               pipeline: TFDataSet,
               mini_batch_size: int = None,
               shuffle_cache: int = 100,
               exclude_keys: List = None):
    super().__init__()
    # Tensorflow is in general not required to use the library
    assert mini_batch_size is None, "Depreceated"
    assert TFDataSet is not None, "Tensorflow must be installed to use this " \
                                  "feature."
    assert tfds is not None, "Tensorflow datasets must be installed to use " \
                             "this feature."

    self._observation_count = jnp.int32(pipeline.cardinality().numpy())
    # Basic pipeline, from which all other pipelines are constructed
    self._pipeline = pipeline
    self._exclude_keys = [] if exclude_keys is None else exclude_keys
    self._shuffle_cache = shuffle_cache

    self._pipelines: List[TFDataSet] = []

  def register_random_pipeline(self,
                               cache_size: int = 1,
                               mb_size: int = None,
                               **kwargs) -> int:
    """Register a new chain which draw samples randomly.

    Args:
      cache_size: The number of drawn batches.
      mb_size: The number of observations per batch.

    Returns:
      Returns the id of the new chain.

    """

    # Assert that not kwargs are passed with the intention to control the
    # initial state of the tensorflow data loader
    assert kwargs == {}, "Tensorflow data loader does not accept additional "\
                         "kwargs"
    new_chain_id = len(self._pipelines)

    # Randomly draw a number of cache_size mini_batches, where each mini_batch
    # contains self.mini_batch_size elements.
    random_data = self._pipeline.repeat()
    random_data = random_data.shuffle(self._shuffle_cache)
    random_data = random_data.batch(mb_size)
    random_data = random_data.batch(cache_size)

    # The data must be transformed to numpy arrays, as most numpy arrays can
    # be transformed to the duck-typed jax array form
    random_data = tfds.as_numpy(random_data)
    random_data = iter(random_data)

    self._pipelines.append(random_data)

    return new_chain_id

  def register_ordered_pipeline(self,
                                cache_size: int = 1,
                                mb_size: int = None,
                                **kwargs
                                ) -> int:
    """Register a chain which assembles batches in an ordered manner.

    Args:
      cache_size: The number of drawn batches.
      mb_size: The number of observations per batch.

    Returns:
      Returns the id of the new chain.

    """
    raise NotImplementedError

  def get_batches(self, chain_id: int) -> PyTree:
    """Draws a batch from a chain.

    Args:
      chain_id: ID of the chain, which holds the information about the form of
        the batch and the process of assembling.

    Returns:
      Returns a superbatch as registered by :func:`register_random_pipeline` or
      :func:`register_ordered_pipeline` with `cache_size` batches holding
      `mb_size` observations.

    """

    # Not supported data types, such as strings, must be excluded before
    # transformation to jax types.
    numpy_batch = next(self._pipelines[chain_id])
    if self._exclude_keys is not None:
      for key in self._exclude_keys:
        del numpy_batch[key]

    return tree_util.tree_map(jnp.array, numpy_batch)

  @property
  def _format(self):
    data_spec = self._pipeline.element_spec
    if self._exclude_keys is not None:
      not_excluded_elements = {id: elem for id, elem in data_spec.items()
                               if id not in self._exclude_keys}
    else:
      not_excluded_elements = data_spec

    def leaf_dtype_struct(leaf):
      shape = tuple(int(s) for s in leaf.shape if s is not None)
      dtype = leaf.dtype.as_numpy_dtype
      return jax.ShapeDtypeStruct(
        dtype=dtype,
        shape=shape)

    return tree_util.tree_map(leaf_dtype_struct, not_excluded_elements)

  @property
  def static_information(self):
    """Returns information about total samples count and batch size. """
    information = {
      "observation_count" : self._observation_count
    }
    return information


class NumpyDataLoader(HostDataLoader):
  """Load complete dataset into memory from multiple numpy arrays.

  This data loader supports checkpointing, starting chains from a well defined
  state and true random access.

  The pipeline can be constructed directly from numpy arrays:

  .. doctest::

    >>> import numpy as onp
    >>> from jax_sgmc import data
    >>>
    >>> x, y = onp.arange(10), onp.zeros((10, 4, 3))
    >>>
    >>> data_loader = data.NumpyDataLoader(name_for_x=x, name_for_y=y)
    >>>
    >>> zero_batch = data_loader.initializer_batch(4)
    >>> for key, value in zero_batch.items():
    ...   print(f"{key}: shape={value.shape}, dtype={value.dtype}")
    name_for_x: shape=(4,), dtype=int32
    name_for_y: shape=(4, 4, 3), dtype=float32

  Args:
    reference_data: Each kwarg-pair is an entry in the returned data-dict.

  """

  def __init__(self, mini_batch_size: int = None, **reference_data):
    super().__init__()
    assert mini_batch_size is None, "Global mini batch size depreceated. "
    assert len(reference_data) > 0, "Observations are required."

    first_key = list(reference_data.keys())[0]
    observation_count = reference_data[first_key].shape[0]

    self._reference_data = dict()
    for name, array in reference_data.items():
      assert array.shape[0] == observation_count, "Number of observations is" \
                                                  "ambiguous."
      # Transform if jax arrays are passed by mistake
      self._reference_data[name] = onp.array(array)

    self._chains: List[Dict[str, Any]] = []
    self._observation_count = observation_count

  def save_state(self, chain_id: int) -> PyTree:
    """Returns all necessary information to restore the dataloader state.

    Args:
      chain_id: Each chain can be checkpointed independently.

    Returns:
      Returns necessary information to restore the state of the chain via
      :func:`load_state`.

    """
    # Get the state of all random data generators. All other information will be
    # set by initializing the generator on the same way as before

    chain_data = self._chains[chain_id]
    if chain_data['type'] == 'random':
      return {'random': chain_data['rng'].bit_generator.state}
    elif chain_data['type'] == 'ordered':
      return {'ordered': chain_data['idx_offset']}
    else:
      raise ValueError(f"Chain type {chain_data['type']} is unknown.")

  def load_state(self, chain_id: int, data) -> None:
    """Restores dataloader state from previously computed checkpoint.

    Args:
      chain_id: The chain to restore the state.
      data: Data from :func:`save_state` to restore state of the chain.

    """
    # Restore the state by setting the random number generators to the
    # checkpointed state
    type, value = data.popitem()
    if type == 'random':
      self._chains[chain_id]['rng'].bit_generator.state = value
    elif type == 'ordered':
      self._chains[chain_id]['idx_offset'] = value
    else:
      raise ValueError(f"Chain type {type} is unknown.")

  def register_random_pipeline(self,
                               cache_size: int = 1,
                               mb_size: int = None,
                               **kwargs: Any) -> int:
    """Register a new chain which draw samples randomly.

    Args:
      cache_size: The number of drawn batches.
      mb_size: The number of observations per batch.
      seed: Set the random seed to start the chain at a well defined state.

    Returns:
      Returns the id of the new chain.

    """
    # The random state of each chain can be defined unambiguously via the
    # PRNGKey
    assert mb_size <= self._observation_count, \
      (f"The batch size cannot be bigger than the observation count. Provided "
       f"{mb_size} and {self._observation_count}")

    chain_id = len(self._chains)

    seed = kwargs.get("seed", chain_id)
    rng = onp.random.default_rng(
      onp.random.SeedSequence(seed).spawn(1)[0])

    new_chain = {'type': 'random',
                 'rng': rng,
                 'idx_offset': None,
                 'mb_size': mb_size,
                 'cache_size': cache_size}

    self._chains.append(new_chain)
    return chain_id

  def register_ordered_pipeline(self,
                                cache_size: int = 1,
                                mb_size: int = None,
                                **kwargs
                                ) -> int:
    """Register a chain which assembles batches in an ordered manner.

    Args:
      cache_size: The number of drawn batches.
      mb_size: The number of observations per batch.
      seed: Set the random seed to start the chain at a well defined state.

    Returns:
      Returns the id of the new chain.

    """
    assert mb_size <= self._observation_count, \
      (f"The batch size cannot be bigger than the observation count. Provided "
       f"{mb_size} and {self._observation_count}")
    chain_id = len(self._chains)

    new_chain = {'type': 'ordered',
                 'rng': None,
                 'idx_offset': 0,
                 'mb_size': mb_size,
                 'cache_size': cache_size}

    self._chains.append(new_chain)
    return chain_id

  def get_batches(self, chain_id: int) -> PyTree:
    """Draws a batch from a chain.

    Args:
      chain_id: ID of the chain, which holds the information about the form of
        the batch and the process of assembling.

    Returns:
      Returns a superbatch as registered by :func:`register_random_pipeline` or
      :func:`register_ordered_pipeline` with `cache_size` batches holding
      `mb_size` observations.

    """
    if self._chains[chain_id]['type'] == 'random':
      return self._random_batches(chain_id)
    elif self._chains[chain_id]['type'] == 'ordered':
      return self._ordered_batches(chain_id)
    else:
      return None

  def _random_batches(self, chain_id: int) -> PyTree:
    # Get the random state of the chain, do some random operations and then save
    # the random state of the chain.
    def generate_selections():
      for _ in range(self._chains[chain_id]['cache_size']):
        mb_idx = self._chains[chain_id]['rng'].choice(
          onp.arange(0, self._observation_count),
          size=self._chains[chain_id]['mb_size'],
          replace=False
        )
        yield mb_idx
    selected_observations_index = onp.array(list(generate_selections()))
    selected_observations = dict()
    for key, data in self._reference_data.items():
      if data.ndim == 1:
        selection = data[selected_observations_index,]
      else:
        selection = data[selected_observations_index,::]
      selected_observations[key] = selection
    mini_batch_pytree = tree_util.tree_map(jnp.array, selected_observations)
    return mini_batch_pytree

  def _ordered_batches(self, chain_id: int) -> PyTree:
    cache_size = self._chains[chain_id]['cache_size']
    mini_batch_size = self._chains[chain_id]['mb_size']
    sample_count = self._observation_count

    def select_mini_batch():
      for _ in range(cache_size):
        idcs = onp.arange(mini_batch_size) + self._chains[chain_id]['idx_offset']
        # Start again at the first sample if all samples have been returned
        if self._chains[chain_id]['idx_offset'] + mini_batch_size > sample_count:
          self._chains[chain_id]['idx_offset'] = 0
        else:
          self._chains[chain_id]['idx_offset'] += mini_batch_size
        # Simply return the first samples again if less samples remain than
        # necessary to fill the cache
        yield onp.mod(idcs, sample_count)

    selected_observations_index = onp.array(list(select_mini_batch()))
    selected_observations = dict()
    for key, data in self._reference_data.items():
      if data.ndim == 1:
        selection = data[selected_observations_index,]
      else:
        selection = data[selected_observations_index, ::]
      selected_observations[key] = selection

    mini_batch_pytree = tree_util.tree_map(jnp.array, selected_observations)
    return mini_batch_pytree

  @property
  def _format(self):
    """Returns shape and dtype of a single observation. """
    mb_format = dict()
    for name, array in self._reference_data.items():
      # Get the format and dtype of the data
      mb_format[name] = jax.ShapeDtypeStruct(
        dtype=self._reference_data[name].dtype,
        shape=tuple(int(s) for s in array.shape[1:]))
    return mb_format

  @property
  def static_information(self):
    """Returns information about total samples count and batch size. """
    information = {
      "observation_count" : self._observation_count
    }
    return information

class CacheState(NamedTuple):
  """Caches several batches of randomly batched reference data.

  Args:
    cached_batches: An array of mini-batches
    cached_batches_count: Number of cached mini-batches. Equals the first
    dimension of the cached batches
    current_line: Marks the next batch to be returned.
    chain_id: Identifier of the chain to associate random state
    key: PRNGKey for random on-device access

  """
  cached_batches: PyTree = None
  cached_batches_count: Array = None
  current_line: Array = None
  chain_id: Array = None
  state: PyTree = None

random_data_state = CacheState


def random_reference_data(data_loader: DeviceDataLoader,
                          cached_batches_count: int,
                          mb_size: int
                          ) -> RandomBatch:
  """Initializes reference data access in jit-compiled functions.

  Randomly draw batches from a given dataset on the host or the device.

  Args:
    data_loader: Reads data from storage.
    cached_batches_count: Number of batches in the cache. A larger number is
      faster, but requires more memory.
    mb_size: Size of the data batch.

  Returns:
    Returns a tuple of functions to initialize a new reference data state and
    get a minibatch from the reference data state

  """
  if isinstance(data_loader, HostDataLoader):
    return _random_reference_data_host(
      data_loader, cached_batches_count, mb_size)
  elif isinstance(data_loader, DeviceDataLoader):
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

  Returns:
    Returns a tuple of functions to initialize a new reference data state and
    get a minibatch from the reference data state

  """

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

  # The definition requires passing an argument to the host function. The shape
  # of the returned data must be known before the first call. The chain id
  # determines whether the data is collected randomly or sequentially.

  def get_data(chain_id):
    data = hcb.call(lambda id: data_loader.get_batches(id),
                    chain_id,
                    result_shape=hcb_format)
    return data

  def _new_cache_fn(state: CacheState) -> CacheState:
    """This function is called if the cache must be refreshed."""
    new_data = get_data(state.chain_id)
    new_state = CacheState(
      cached_batches_count=state.cached_batches_count,
      cached_batches=new_data,
      current_line=jnp.array(0),
      chain_id=state.chain_id)
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

    # Refresh the cache if necessary, after all cached batches have been used.
    data_state = _data_state_helper(data_state)
    current_line = jnp.mod(data_state.current_line,
                           data_state.cached_batches_count)

    # Read the current line from the cache and
    mini_batch = tree_index(data_state.cached_batches, current_line)
    current_line = current_line + 1

    new_state = CacheState(
      cached_batches=data_state.cached_batches,
      cached_batches_count=data_state.cached_batches_count,
      current_line=current_line,
      chain_id=data_state.chain_id)

    if information:
      return new_state, (mini_batch, mb_information)
    else:
      return new_state, mini_batch

  return batch_fn, mb_information


def _random_reference_data_host(data_loader: HostDataLoader,
                                cached_batches_count: int = 100,
                                mb_size: int = 1
                                ) -> RandomBatch:
  """Random reference data access via host-callback. """
  batch_fn, _ = _hcb_wrapper(data_loader, cached_batches_count, mb_size)

  def init_fn(**kwargs) -> CacheState:
    # Pass the data loader the information about the number of cached
    # mini-batches. The data loader returns an unique id for reproducibility
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

  # Check that the batch size is not greater than the total dataset size
  if mb_size > total_observations:
    raise ValueError(f"The batch size cannot be greater than the total "
                     f"observation count. Given {mb_size} and "
                     f"{total_observations}.")

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
