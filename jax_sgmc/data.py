"""Data input and output."""

from collections import namedtuple
from functools import partial

from typing import Tuple, Any, Type, Callable, List

import jax
from jax import tree_util, lax
import jax.numpy as jnp
from jax import random, tree_util
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

from jax_sgmc.util import Array

mini_batch_format = namedtuple(
  "mini_batch_format",
  ["observation_count"])
""" Bundling all information about the reference data:

Attributes:
  observation_count: Total number of observatins
  mini_batch: List of tuples, tuples consist of ``(observations, parameters)``
"""

MiniBatch = Tuple[Array, mini_batch_format]
PyTree = Any

# Definition of the data loader class

class DataLoader:
  """Abstract class to define required methods of the DataLoader.

  A class implementing the data loader must have the functionality to load data
  from storage in an ordered and a random fashion.
  """

  def register_random_pipeline(self, cache_size: int=1, **kwargs) -> int:
    """Registers a data pipeline for random data access."""
    assert False, "Must override"

  def register_ordered_pipeline(self, cache_size: int=1, **kwargs) -> int:
    """Registers a data pipeline for ordered data access.

    Ordered data access is required for AMAGOLD to evaluate the true potential
    over all reference data.
    """
    assert False, "Must override"

  def batch_format(self) -> mini_batch_format:
    """Return the form of each batch."""
    assert False, "Must override"

  def random_batches(self, chain_id: int) -> PyTree:
    """Return random batches"""
    assert False, "Must override"


class TensorflowDataLoader(DataLoader):
  """Load data from a tensorflow dataset object."""

  def __init__(self,
               pipeline: TFDataSet,
               mini_batch_size: int = 1,
               shuffle_cache: int = 100,
               exclude_keys: List = None):
    # Tensorflow is in general not required to use the library
    assert TFDataSet is not None, "Tensorflow must be installed to use this " \
                                  "feature."
    assert tfds is not None, "Tensorflow datasets must be installed to use " \
                             "this feature."

    self.observation_count = jnp.int32(pipeline.cardinality().numpy())
    self.pipeline = pipeline
    self.exclude_keys = exclude_keys
    self.mini_batch_size = mini_batch_size
    self.shuffle_cache = shuffle_cache
    self.random_pipelines = []
    self.full_data_pipelines = []

  def register_random_pipeline(self, cache_size: int=1, **kwargs) -> int:
    # Singleton, as the order of data for multiple pipelines is irrelevant
    new_chain_id = len(self.random_pipelines)

    # Randomly draw a number of cache_size mini_batches, where each mini_batch
    # contains self.mini_batch_size elements.
    random_data = self.pipeline.repeat()
    random_data = random_data.shuffle(self.shuffle_cache)
    random_data = random_data.batch(self.mini_batch_size)
    random_data = random_data.batch(cache_size)

    # The data must be transformed to numpy arrays, as most numpy arrays can
    # be transformed to the duck-typed jax array form
    random_data = tfds.as_numpy(random_data)
    random_data = iter(random_data)

    self.random_pipelines.append(random_data)

    return new_chain_id

  def random_batches(self, chain_id: int) -> PyTree:
    assert chain_id < len(self.random_pipelines), f"Pipe {chain_id} does not" \
                                                  f"exist."

    # Not supportet data types, such as strings, must be excluded before
    # transformation to jax types.
    numpy_batch = next(self.random_pipelines[chain_id])
    if self.exclude_keys is not None:
      for key in self.exclude_keys:
        numpy_batch.pop(key)

    jax_batch = tree_util.tree_map(lambda leaf: jnp.array(leaf),
                                   numpy_batch)

    return jax_batch

  def batch_format(self) -> mini_batch_format:
    return mini_batch_format(observation_count=self.observation_count)


class NumpyDataLoader(DataLoader):
  """Load complete dataset into memory from multiple numpy files."""

  def __init__(self, mini_batch_size: int, **reference_data):
    assert len(reference_data) > 0, "Observations are required."

    first_key = list(reference_data.keys())[0]
    observation_count = reference_data[first_key].shape[0]

    self.reference_data = dict()
    for name, array in reference_data.items():
      assert array.shape[0] == observation_count, "Number of observations is" \
                                                  "ambiguous."
      self.reference_data[name] = array

    self.PRNGKeys = []
    self.cache_sizes = []
    self.observation_count = observation_count
    self.mini_batch_size = mini_batch_size

  def register_random_pipeline(self, cache_size: int=1, **kwargs) -> int:
    # The random state of each chain can be defined unambiguously via the
    # PRNGKey
    if "key" not in kwargs:
      if len(self.PRNGKeys) == 0:
        new_key = random.PRNGKey(0)
      else:
        new_key, = random.split(self.PRNGKeys[-1], 1)
    else:
      new_key = random.split(kwargs["key"])

    chain_id = len(self.PRNGKeys)
    self.PRNGKeys.append(new_key)
    self.cache_sizes.append(cache_size)

    return chain_id

  def random_batches(self, chain_id: int) -> PyTree:
    assert chain_id < len(self.PRNGKeys), f"Chain {chain_id} does not exist."

    self.PRNGKeys[chain_id], *splits = random.split(
      self.PRNGKeys[chain_id],
      self.cache_sizes[chain_id] + 1)
    splits = jnp.array(splits)
    sample_count = self.observation_count
    @jax.vmap
    def sample_selection(split):
      sample_selection = random.choice(split,
                                       jnp.arange(sample_count),
                                       shape=(self.mini_batch_size,))
      return sample_selection
    selected_observations_index = sample_selection(splits)
    selected_observations = dict()
    for key, data in self.reference_data.items():
      if data.ndim == 1:
        selection = data[selected_observations_index,]
      else:
        selection = data[selected_observations_index,::]
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
  batch_information: Additional information for each bach, such as whether each
    sample is valid or has already been passed before.
  chain_id: Indentifier of the chain
    
"""

# Todo: Implement checkpoint function

def random_reference_data(data_loader: DataLoader,
                          cached_batches_count: int=100
                          ) -> Tuple[Callable, Callable]:
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

  init_chain_id = data_loader.register_random_pipeline(cached_batches_count)

  mini_batch_format = data_loader.batch_format()
  initial_state = data_loader.random_batches(init_chain_id)
  returned_data_format = tree_dtype_struct(initial_state)

  # The definition requires passing an argument to the host function. The shape
  # of the returned data must be known before the first call

  def host_function(chain_id):
    return data_loader.random_batches(chain_id)

  def get_data(chain_id):
    data =  hcb.call(host_function,
                     chain_id,
                     result_shape=returned_data_format)
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

    chain_id = data_loader.register_random_pipeline(cached_batches_count,
                                                    **kwargs)

    inital_cache_state=random_data_state(
      cached_batches=initial_state,
      cached_batches_count=cached_batches_count,
      current_line=0,
      chain_id=chain_id
    )
    return inital_cache_state

  def batch_fn(data_state: random_data_state
              ) -> Tuple[random_data_state, MiniBatch]:
    """Draws a new random batch (hides data transfer between devices)."""

    # Refresh the cache if necessary

    current_line = jnp.mod(data_state.current_line,
                           data_state.cached_batches_count)
    data_state = lax.cond(current_line == 0,
                          new_cache_fn,
                          old_cache_fn,
                          data_state)

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

    return new_state, (random_mini_batch, mini_batch_format)

  return init_fn, batch_fn


# Todo: Implement utility to handle full dataset

# def full_reference_data(data_loader: DataLoader, cached_batches=100):
#   """Initializes reference data access from jit-compiled functions.
#
#   Utility to sequentially load reference data into a jit-compiled function.
#   This utility allows to access every sample exactly once
#   (required by AMAGOLD).
#
#   Args:
#     data_loader: Reads data from storage.
#     cached_batches: Number of batches in the cache. A larger number is faster,
#       but requires more memory.
#
#   Returns:
#     Returns a tuple of functions to initialize a new reference data state and
#     get a minibatch from the reference data state
#
#   """
#
#   # These are helper function which keep a reference to the stateful data object
#   # and can be called via the host_callback.call function
#
#   # Pass the data loader the information about the number of cached mini-batches
#
#   data_loader.multiple_batches(cached_batches)
#
#   # The format of the mini batch is static, thus it must not be passed
#   # in form of a state.
#   mini_batch_format = data_loader.batch_format()
#
#   # The implementation is based on the experimental host_callback module.
#
#   def init_fn():
#     cached_batches = jnp.array(data_loader.random_batches())
#     cached_batches_count = cached_batches.shape[0]
#     inital_state = random_data_state(
#       cached_batches=cached_batches,
#       cached_batches_count=cached_batches_count,
#       current_line=0
#     )
#     return inital_state
#
#   def batch_fn(data_state:):
#     # 1.) Calculate current line modulo
#
#     # 2.) Check if all batches are already used. Reload if so
#
#     # 3.) Load the data from the current cache line
#
#     # 4.) Increase the cache line counter
#
#     pass

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
