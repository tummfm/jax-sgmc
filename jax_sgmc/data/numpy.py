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

from typing import Tuple, Any, Dict, List

import numpy as onp
import jax.numpy as jnp
import jax
from jax import random
from jax import tree_util

from jax_sgmc.data.core import DeviceDataLoader, HostDataLoader, mini_batch_information
from jax_sgmc.data.core import tree_index

PyTree = Any

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
    shuffle: Whether to batch by shuffling the dataset or by drawing
      observations independently (applies only to random data).

  """

  def __init__(self,
               mini_batch_size: int = None,
               shuffle: bool = False,
               **reference_data):
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
    self._shuffle = shuffle

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