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

"""Data input and output.

Accessing data inside a jit-compiled function works

Random Data Access
-----------------



.. doctest::




"""

import abc

from collections import namedtuple
from functools import partial

from typing import Tuple, Any, Callable, List, Optional, Union

import jax
from jax import tree_util, lax
import jax.numpy as jnp

import numpy as onp

# Tensorflow is only required if the tensorflow dataLoader ist used.
try:
  from tensorflow import data as tfd
  import tensorflow_datasets as tfds
  TFDataSet = tfd.Dataset
except ModuleNotFoundError:
  TFDataSet = None
  tfds = None

from jax.experimental import host_callback as hcb
from jax_sgmc.util import Array, scan_vmap

mini_batch_information = namedtuple(
  "mini_batch_information",
  ["observation_count",
   "batch_size"])
""" Bundling all information about the reference data:

Attributes:
  observation_count: Total number of observatins
  mini_batch: List of tuples, tuples consist of ``(observations, parameters)``
"""

PyTree = Any
MiniBatch = Union[Tuple[PyTree],
                  Tuple[PyTree, mini_batch_information],
                  Tuple[PyTree, mini_batch_information, Array]]
RandomBatch = Tuple[Callable[[Optional[Any], Optional[Any]], PyTree],
                    Callable[[PyTree, Optional[bool]], MiniBatch]]
OrderedBatch = Tuple[Callable[[Optional[Any], Optional[Any]], PyTree],
                     Callable[[PyTree, Optional[bool]], MiniBatch]]
PyTree = Any

# Definition of the data loader class

# Todo: State übergeben bei random batch statt chain id. Damit einfacher
#       checkpointing

class DataLoader(metaclass=abc.ABCMeta):
  """Abstract class to define required methods of the DataLoader.

  A class implementing the data loader must have the functionality to load data
  from storage in an ordered and a random fashion.
  """

  def __init__(self, data_collector: Any = None):
    if data_collector is not None:
      # Allows the data collector to save and restore the dataloader state
      data_collector.register_data_loader(self)

  # Todo: Add a checkpoint function returning the current state
  #       def checkpoint(self, ...):

  def save_state(self):
    """Returns all necessary information to restore the dataloader state."""
    raise NotImplementedError("This method must be overwritten to allow "
                              "checkpointing of the data loader.")

  def load_state(self, data):
    """Restores dataloader state from previously computed checkpoint."""
    raise NotImplementedError("This method must be overwritten to allow "
                              "checkpointing of the data loader.")

  @abc.abstractmethod
  def register_random_pipeline(self,
                               cache_size: int=1,
                               **kwargs
                               ) -> int:
    """Registers a data pipeline for random data access.

    Args:
      cache_size: Number of mini_batches in device memory on the same time
      **kwargs: Additional information depending on data loader

    Returns:
      Returns an identifier of the chain.

    """

  @abc.abstractmethod
  def register_ordered_pipeline(self,
                                cache_size: int=1,
                                **kwargs
                                ) -> int:
    """Registers a data pipeline for ordered data access.

    Ordered data access is required for AMAGOLD to evaluate the true potential
    over all reference data.

    Args:
      cache_size: Number of mini_batches in device memory on the same time
      **kwargs: Additional information depending on data loader

    Returns:
      Returns an identifier of the chain.

    """

  @property
  @abc.abstractmethod
  def _batch_format(self):
    """dtype and shape of a mini-batch. """

  @property
  @abc.abstractmethod
  def _mini_batch_format(self):
    """namedtuple with information about mini-batch."""

  def batch_format(self,
                   cache_size:int,
                   ) -> PyTree:
    """Returns dtype and shape of cached mini-batches.

    Args:
      cache_size: number of cached mini-batches

    Returns:
      Returns a pytree with the same tree structure as the random data cache but
      with ``jax.ShapedDtypeStruct``` as leaves.
    """
    # Append the cache size to the batch_format
    def append_cache_size(leaf):
      new_shape = tuple(onp.append(cache_size, leaf.shape))
      return jax.ShapeDtypeStruct(
        dtype=leaf.dtype,
        shape=new_shape
      )
    format = tree_util.tree_map(append_cache_size, self._batch_format)
    return format, self._mini_batch_format

  def initializer_batch(self) -> PyTree:
    """Returns a zero-like mini-batch. """
    batch = tree_util.tree_map(
      lambda leaf: jnp.zeros(leaf.shape, dtype=leaf.dtype),
      self._batch_format
    )
    return batch

  @abc.abstractmethod
  def random_batches(self, chain_id: int) -> PyTree:
    """Return random batches."""

  @abc.abstractmethod
  def ordered_batches(self, chain_id: int) -> PyTree:
    """Return ordered batches."""

class TensorflowDataLoader(DataLoader):
  """Load data from a tensorflow dataset object."""

  def __init__(self,
               pipeline: TFDataSet,
               mini_batch_size: int = 1,
               shuffle_cache: int = 100,
               exclude_keys: List = []):
    super().__init__()
    # Tensorflow is in general not required to use the library
    assert TFDataSet is not None, "Tensorflow must be installed to use this " \
                                  "feature."
    assert tfds is not None, "Tensorflow datasets must be installed to use " \
                             "this feature."

    self._observation_count = jnp.int32(pipeline.cardinality().numpy())
    self._pipeline = pipeline
    self._exclude_keys = exclude_keys
    self._mini_batch_size = mini_batch_size
    self._shuffle_cache = shuffle_cache
    self._random_pipelines = []
    self._full_data_pipelines = []

  @property
  def _batch_format(self):
    """Returns pytree with information about shape and dtype of a minibatch. """
    data_spec = self._pipeline.element_spec
    if self._exclude_keys is not None:
      not_excluded_elements = {id: elem for id, elem in data_spec.items() if id not in self._exclude_keys}
    else:
      not_excluded_elements = data_spec
    def leaf_dtype_struct(leaf):
      shape = tuple(s for s in leaf.shape if s is not None)
      mb_shape = tuple(onp.append(self._mini_batch_size, shape))
      mb_shape = tree_util.tree_map(int, mb_shape)
      dtype = leaf.dtype.as_numpy_dtype
      return jax.ShapeDtypeStruct(
        dtype=dtype,
        shape=mb_shape)
    return tree_util.tree_map(leaf_dtype_struct, not_excluded_elements)

  @property
  def _mini_batch_format(self) -> mini_batch_information:
    """Returns information about total samples count and batch size. """
    return mini_batch_information(observation_count=self._observation_count,
                                  batch_size=self._mini_batch_size)

  def register_random_pipeline(self, cache_size: int=1, **kwargs) -> int:
    # Assert that not kwargs are passed with the intention to control the
    # initial state of the tensorflow data loader
    assert kwargs == {}, "Tensorflow data loader does not accept additional "\
                         "kwargs"
    new_chain_id = len(self._random_pipelines)

    # Randomly draw a number of cache_size mini_batches, where each mini_batch
    # contains self.mini_batch_size elements.
    random_data = self._pipeline.repeat()
    random_data = random_data.shuffle(self._shuffle_cache)
    random_data = random_data.batch(self._mini_batch_size)
    random_data = random_data.batch(cache_size)

    # The data must be transformed to numpy arrays, as most numpy arrays can
    # be transformed to the duck-typed jax array form
    random_data = tfds.as_numpy(random_data)
    random_data = iter(random_data)

    self._random_pipelines.append(random_data)

    return new_chain_id

  def register_ordered_pipeline(self,
                                 cache_size: int=1,
                                 **kwargs
                                 ) -> int:
    raise NotImplementedError

  def random_batches(self, chain_id: int) -> PyTree:
    assert chain_id < len(self._random_pipelines), f"Pipe {chain_id} does not" \
                                                   f"exist."

    # Not supportet data types, such as strings, must be excluded before
    # transformation to jax types.
    numpy_batch = next(self._random_pipelines[chain_id])
    if self._exclude_keys is not None:
      for key in self._exclude_keys:
        del numpy_batch[key]

    jax_batch = tree_util.tree_map(jnp.array, numpy_batch)
    return jax_batch


class NumpyDataLoader(DataLoader):
  """Load complete dataset into memory from multiple numpy arrays."""

  def __init__(self, mini_batch_size: int, **reference_data):
    super().__init__()
    assert len(reference_data) > 0, "Observations are required."

    first_key = list(reference_data.keys())[0]
    observation_count = reference_data[first_key].shape[0]

    assert mini_batch_size <= observation_count, "There cannot be more samples"\
                                                 "in the minibatch than there"\
                                                 "are observations."

    self._reference_data = dict()
    self._format = dict()
    for name, array in reference_data.items():
      assert array.shape[0] == observation_count, "Number of observations is" \
                                                  "ambiguous."
      # Transform if jax arrays are passed by mistake
      self._reference_data[name] = onp.array(array)
      # Get the format and dtype of the data
      self._batch_format[name] = jax.ShapeDtypeStruct(
        dtype=self._reference_data[name].dtype,
        shape=tuple(onp.append(mini_batch_size, array.shape[1:])))
      print(self._batch_format)

    self._idx_offset = []
    self._rng = []
    self._ordered_cache_sizes = []
    self._random_cache_sizes = []
    self._observation_count = observation_count
    self._mini_batch_size = mini_batch_size

  @property
  def _batch_format(self):
    """Returns pytree with information about shape and dtype of a minibatch. """
    return self._format

  @property
  def _mini_batch_format(self):
    """Returns information about total samples count and batch size. """
    mb_info = mini_batch_information(
      observation_count=self._observation_count,
      batch_size=self._mini_batch_size)
    return mb_info

  def save_state(self):
    """Returns all necessary information to restore the dataloader state.

    Returns:
      Returns list of current rng states. Can be passed to load state to restore
      the internal state.

    """
    # Get the state of all random data generators. All other information will be
    # set by initializing the generator on the same way as before
    rng_states = [rng.bit_generator.state for rng in self._rng]
    return rng_states

  def load_state(self, data):
    """Restores dataloader state from previously computed checkpoint.

    Args:
      data: list of rng_states

    """
    # Restore the state by setting the random number generators to the
    # checkpointed state
    for chain_id, state in enumerate(data):
      self._rng[chain_id].bit_generator.state = state

  def register_random_pipeline(self, cache_size: int=1, **kwargs) -> int:
    # The random state of each chain can be defined unambiguously via the
    # PRNGKey
    seed = kwargs.get("seed", len(self._rng))
    rng = onp.random.default_rng(
      onp.random.SeedSequence(seed).spawn(1)[0])

    chain_id = len(self._rng)
    self._rng.append(rng)
    self._random_cache_sizes.append(cache_size)
    return chain_id

  def random_batches(self, chain_id: int) -> PyTree:
    assert chain_id < len(self._rng), f"Chain {chain_id} does not exist."

    # Get the random state of the chain, do some random operations and then save
    # the random state of the chain.
    def generate_selections():
      for _ in range(self._random_cache_sizes[chain_id]):
        mb_idx = self._rng[chain_id].choice(
          onp.arange(0, self._observation_count),
          size=self._mini_batch_size,
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

  def register_ordered_pipeline(self,
                                 cache_size: int = 1,
                                 **kwargs
                                 ) -> int:
    chain_id = len(self._idx_offset)
    self._idx_offset.append(0)
    self._ordered_cache_sizes.append(cache_size)
    return chain_id

  def ordered_batches(self, chain_id: int) -> PyTree:
    cache_size = self._ordered_cache_sizes[chain_id]
    mini_batch_size = self._mini_batch_size
    sample_count = self._observation_count

    def select_mini_batch():
      for _ in range(cache_size):
        idcs = onp.arange(mini_batch_size) + self._idx_offset[chain_id]
        # Start again at the first sample if all samples have been returned
        if self._idx_offset[chain_id] + mini_batch_size > sample_count:
          idx_offset = 0
        else:
          self._idx_offset[chain_id] += mini_batch_size
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

random_data_state = namedtuple("random_data_state",
                               ["cached_batches",
                                "cached_batches_count",
                                "current_line",
                                "chain_id"])
"""Caches several batches of randomly batched reference data.

Attributes:
  cached_batches: An array of mini-batches
  cached_batches_count: Number of cached mini-batches. Equals the first
  dimension of the cached batches
  current_line: Marks the next batch to be returned.
  chain_id: Identifier of the chain to associate random state

"""

full_data_state = namedtuple("full_data_state",
                               ["cached_batches",
                                "cached_batches_count",
                                "current_line",
                                "batch_information",
                                "chain_id"])
"""Caches several batches of sequentially batched reference data.

Attributes:
  cached_batches: An array of mini-batches
  cached_batches_count: Number of cached mini-batches. Equals the first
    dimension of the cached batches
  current_line: Marks the next batch to be returned.
  current_observation: Keep track 
  chain_id: Indentifier of the chain
"""

def random_reference_data(data_loader: DataLoader,
                          cached_batches_count: int=100
                          ) -> RandomBatch:
  """Initializes reference data access from jit-compiled functions.

  Utility to sequentially load reference data into a jit-compiled function. The
  mini_batches are gathered randomly by the data loader.

  Args:
    data_loader: Reads data from storage.
    cached_batches_count: Number of batches in the cache. A larger number is
      faster, but requires more memory.

  Returns:
    Returns a tuple of functions to initialize a new reference data state and
    get a minibatch from the reference data state

  """

  # These are helper function which keep a reference to the stateful data object
  # and can be called via the host_callback.call function
  # The format of the mini batch is static, thus it must not be passed
  # in form of a state.

  hcb_format, mb_information = data_loader.batch_format(cached_batches_count)

  # The definition requires passing an argument to the host function. The shape
  # of the returned data must be known before the first call

  def host_function(chain_id):
    return data_loader.random_batches(chain_id)

  def get_data(chain_id):
    data =  hcb.call(host_function,
                     chain_id,
                     result_shape=hcb_format)
    return data

  def new_cache_fn(state: random_data_state) -> random_data_state:
    """This function is called if the cache must be refreshed."""
    new_data = get_data(state.chain_id)
    new_state = random_data_state(
      cached_batches_count=state.cached_batches_count,
      cached_batches=new_data,
      current_line=0,
      chain_id=state.chain_id)
    return new_state

  def old_cache_fn(state: random_data_state) -> random_data_state:
    """This function is called if the cache must not be refreshed."""
    return state

  # Thes following functions are passed back

  def init_fn(**kwargs) -> random_data_state:
    # Pass the data loader the information about the number of cached
    # mini-batches. The data loader returns an unique id for reproducibility
    chain_id = data_loader.register_random_pipeline(
      cached_batches_count,
      **kwargs)
    initial_state = data_loader.random_batches(chain_id)
    inital_cache_state=random_data_state(
      cached_batches=initial_state,
      cached_batches_count=cached_batches_count,
      current_line=0,
      chain_id=chain_id
    )
    return inital_cache_state

  @scan_vmap.stop_vmap_decorator
  def _data_state_helper(data_state):
    return lax.cond(data_state.current_line == data_state.cached_batches_count,
                    new_cache_fn,
                    old_cache_fn,
                    data_state)

  def batch_fn(data_state: random_data_state,
               information: bool = False
               ) -> Union[Tuple[random_data_state, MiniBatch],
                          Tuple[random_data_state,
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
    random_mini_batch = tree_index(data_state.cached_batches, current_line)
    current_line = current_line + 1

    new_state = random_data_state(
      cached_batches=data_state.cached_batches,
      cached_batches_count=data_state.cached_batches_count,
      current_line=current_line,
      chain_id=data_state.chain_id)

    # The static_information contains information such as total observation
    # count and must be a valid jax type.

    if information:
      return new_state, (random_mini_batch, mb_information)
    else:
      return new_state, random_mini_batch

  return init_fn, batch_fn


# Todo: Implement utility to handle full dataset

def full_reference_data(data_loader: DataLoader,
                          cached_batches_count: int = 100
                          ) -> OrderedBatch:
  """Initializes reference data access from jit-compiled functions.

  Utility to sequentially load reference data into a jit-compiled function. The
  mini_batches are gathered one after another.

  Args:
    data_loader: Reads data from storage.
    cached_batches_count: Number of batches in the cache. A larger number is
      faster, but requires more memory.
  Returns:
    Returns a tuple of functions to initialize a new reference data state and
    get a minibatch from the reference data state

  """

  # These are helper function which keep a reference to the stateful data object
  # and can be called via the host_callback.call function
  # The format of the mini batch is static, thus it must not be passed
  # in form of a state.

  hcb_format, mb_information = data_loader.batch_format(cached_batches_count)

  # The definition requires passing an argument to the host function. The shape
  # of the returned data must be known before the first call

  def host_function(chain_id):
    return data_loader.ordered_batches(chain_id)

  def get_data(chain_id):
    data =  hcb.call(host_function,
                     chain_id,
                     result_shape=hcb_format)
    return data

  def new_cache_fn(state: random_data_state) -> random_data_state:
    """This function is called if the cache must be refreshed."""
    new_data = get_data(state.chain_id)
    new_state = random_data_state(
      cached_batches_count=state.cached_batches_count,
      cached_batches=new_data,
      current_line=0,
      chain_id=state.chain_id)
    return new_state

  def old_cache_fn(state: random_data_state) -> random_data_state:
    """This function is called if the cache must not be refreshed."""
    return state

  # Thes following functions are passed back

  def init_fn(**kwargs) -> random_data_state:
    # Pass the data loader the information about the number of cached
    # mini-batches. The data loader returns an unique id for reproducibility
    chain_id = data_loader.register_random_pipeline(
      cached_batches_count,
      **kwargs)
    initial_state = data_loader.random_batches(chain_id)
    inital_cache_state=random_data_state(
      cached_batches=initial_state,
      cached_batches_count=cached_batches_count,
      current_line=0,
      chain_id=chain_id
    )
    return inital_cache_state

  def batch_fn(data_state: random_data_state,
               information: bool = False
               ) -> Union[Tuple[random_data_state, MiniBatch],
                          Tuple[random_data_state,
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
    data_state = lax.cond(data_state.current_line
                          == data_state.cached_batches_count,
                          new_cache_fn,
                          old_cache_fn,
                          data_state)
    current_line = jnp.mod(data_state.current_line,
                           data_state.cached_batches_count)

    # Read the current line from the cache and
    random_mini_batch = tree_index(data_state.cached_batches, current_line)
    current_line = current_line + 1

    new_state = random_data_state(
      cached_batches=data_state.cached_batches,
      cached_batches_count=data_state.cached_batches_count,
      current_line=current_line,
      chain_id=data_state.chain_id)

    # The static_information contains information such as total observation
    # count and must be a valid jax type.

    if information:
      return new_state, (random_mini_batch, mb_information)
    else:
      return new_state, random_mini_batch

  # Todo: Return static information -> number of mini_batches
  return init_fn, batch_fn

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


def vmap_helper(batch_fn):
  """Sequentially load a minibatch.

  host_callback.call() currently does not support vmap(). Therefore, this helper
  function can be used to load the reference data before a call to vmap.

  Args:
    batch_fn: Function returned by ``random_reference_data`` or
    ``ordered_reference_data```

  Returns:
    Returns a function taking a batch of reference_data states and returns a
    batch of mini-batches.

  """

  def get_fn(_, state):
    state, minibatch = batch_fn(state)
    return None, {"states": state, "minibatches": minibatch}

  def vmap_batch_fn(states):
    _, minibatches = lax.scan(get_fn, None, states)
    return minibatches["states"], minibatches["minibatches"]

  return vmap_batch_fn
