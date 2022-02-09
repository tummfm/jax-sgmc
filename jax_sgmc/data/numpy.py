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

""""""
import math
import itertools
from typing import Tuple, Any, Dict, List

import numpy as onp
import jax.numpy as jnp
import jax
from jax import random
from jax import tree_util

from jax_sgmc.data.core import DeviceDataLoader, HostDataLoader, DataLoader
from jax_sgmc.data.core import mini_batch_information
from jax_sgmc.data.core import tree_index

PyTree = Any

class NumpyBase(DataLoader):

  def __init__(self, on_device: bool = True, **reference_data):
    super().__init__()

    observation_counts = []
    self._reference_data = dict()
    for name, array in reference_data.items():
      observation_counts.append(len(array))
      # Transform to jax arrays if on device
      if on_device:
        self._reference_data[name] = jnp.array(array)
      else:
        self._reference_data[name] = onp.array(array)

    # Check same number of observations
    if onp.any(onp.array(observation_counts) != observation_counts[0]):
      raise TypeError("All reference_data arrays must have the same length in"
                      " the first dimension.")

    self._observation_count = observation_counts[0]

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
      "observation_count": self._observation_count
    }
    return information


class DeviceNumpyDataLoader(NumpyBase, DeviceDataLoader):
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
    super().__init__(on_device=True, **reference_data)

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


class NumpyDataLoader(NumpyBase, HostDataLoader):
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
               **reference_data):
    super().__init__(
      on_device=False,
      **reference_data)
    self._chains: List = []

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
                               in_epochs: bool = False,
                               shuffle: bool = False,
                               **kwargs: Any) -> int:
    """Register a new chain which draw samples randomly.

    Args:
      cache_size: The number of drawn batches.
      mb_size: The number of observations per batch.
      shuffle: Shuffle dataset instead of drawing randomly from the
        observations.
      in_epochs: Samples returned twice per epoch are marked via mask = 0 (only
        if ``shuffle = True``.
      seed: Set the random seed to start the chain at a well defined state.

    Returns:
      Returns the id of the new chain.

    """
    # The random state of each chain can be defined unambiguously via the
    # PRNGKey
    if mb_size > self._observation_count:
      raise ValueError(f"The batch size cannot be bigger than the observation "
                       f"count. Provided {mb_size} and "
                       f"{self._observation_count}")
    if not shuffle and in_epochs:
      raise ValueError(f"in_epochs = True can only be used for shuffle = True.")

    chain_id = len(self._chains)

    seed = kwargs.get("seed", chain_id)
    rng = onp.random.default_rng(
      onp.random.SeedSequence(seed).spawn(1)[0])

    # The indices list must have at least the length equal to the number of
    # observations but should also be a multiple of the mb_size to simplify
    # getting new indices.
    new_chain = {'type': 'random',
                 'rng': rng,
                 'idx_offset': None,
                 'in_epochs': in_epochs,
                 'shuffle': shuffle,
                 'remaining_samples': 0,
                 'draws': math.ceil(self._observation_count / mb_size),
                 'random_indices': None,
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
    # Todo: Slicing is the same, only specify different function for generating
    #   the index and the mask.
    if self._chains[chain_id]['type'] == 'random':
      return self._random_batches(chain_id)
    elif self._chains[chain_id]['type'] == 'ordered':
      return self._ordered_batches(chain_id)
    else:
      return None

  def _random_batches(self, chain_id: int) -> PyTree:
    # Get the random state of the chain, do some random operations and then save
    # the random state of the chain.
    # Todo: Something is wrong here!
    indices, masks = list(zip(*map(
      self._random_indices,
      itertools.repeat(chain_id, self._chains[chain_id]['cache_size']))))
    selected_observations_index = onp.array(indices)
    selections_masks = onp.array(masks, dtype=onp.bool_)
    selected_observations = dict()
    for key, data in self._reference_data.items():
      if data.ndim == 1:
        selection = data[selected_observations_index,]
      else:
        selection = data[selected_observations_index,::]
      selected_observations[key] = selection
    mini_batch_pytree = tree_util.tree_map(jnp.array, selected_observations)

    return mini_batch_pytree, selections_masks

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
    return mini_batch_pytree, None

  def _random_indices(self, chain_id: int) -> Tuple[List, Any]:
    """Returns indices and mask to access random data. """
    chain = self._chains[chain_id]
    if chain['in_epochs']:
      return self._shuffle_in_epochs(chain)
    elif chain['shuffle']:
      return self._shuffle_indices(chain)
    else:
      return self._draw_indices(chain)

  def _draw_indices(self, chain):
    # Randomly choose batches
    selections = chain['rng'].choice(
      onp.arange(0, self._observation_count),
      size=self._observation_count,
      replace=True)
    mask = onp.ones(chain['mb_size'], dtype=onp.bool_)
    return selections, mask

  def _shuffle_indices(self, chain):
    floor_draws = math.floor(self._observation_count / chain['mb_size'])
    ceil_draws = floor_draws + 1

    if chain['remaining_samples'] < chain['mb_size']:
      # The indices have to be refreshed. Shuffling is equivalent to drawing
      # without replacement.
      new_indices = chain['rng'].choice(
        onp.arange(0, self._observation_count),
        size=self._observation_count,
        replace=False)

      if chain['random_indices'] is None:
        # Special options for first run
        chain['draws'] = 0
        chain['random_indices'] = onp.zeros(
          ceil_draws * chain['mb_size'], dtype=onp.int_)

      # Update only invalid samples (do not overwrite still valid samples)
      update_idxs = onp.mod(
        onp.arange(self._observation_count)
        + chain['draws'] * chain['mb_size']
        + chain['remaining_samples'],
        ceil_draws * chain['mb_size'])
      chain['random_indices'][update_idxs] = new_indices
      chain['remaining_samples'] += self._observation_count

    # print(f"current indices: {chain['random_indices']}")
    # print(f"  ------------>: {onp.array(onp.arange(ceil_draws * chain['mb_size']) == chain['draws'] * chain['mb_size'], dtype=int)}")

    mask = onp.ones(chain['mb_size'], dtype=onp.bool_)

    # Take the new indices
    selections_idxs = onp.mod(
      onp.arange(chain['mb_size']) + chain['draws'] * chain['mb_size'],
      chain['mb_size'] * ceil_draws)
    selections = onp.copy(chain['random_indices'][selections_idxs])
    chain['draws'] = (chain['draws'] + 1) % ceil_draws
    chain['remaining_samples'] -= chain['mb_size']

    return selections, mask

  def _shuffle_in_epochs(self, chain):
    ceil_draws = math.ceil(self._observation_count / chain['mb_size'])

    if chain['draws'] == ceil_draws:
      # The indices have to be refreshed. Shuffling is equivalent to drawing
      # without replacement.
      new_indices = chain['rng'].choice(
        onp.arange(0, self._observation_count),
        size=self._observation_count,
        replace=False)

      if chain['random_indices'] is None:
        # Special options for first run
        chain['draws'] = 0
        chain['random_indices'] = onp.zeros(
          ceil_draws * chain['mb_size'], dtype=onp.int_)

      chain['random_indices'][0:self._observation_count] = new_indices
      chain['draws'] = 0

    start_idx = chain['mb_size'] * chain['draws']
    end_idx = chain['mb_size'] * (chain['draws'] + 1)

    mask = onp.arange(start_idx, end_idx) < self._observation_count

    # print(f"current indices: {chain['random_indices']}")
    # print(f"  ------------>: {onp.array(onp.arange(ceil_draws * chain['mb_size']) == chain['draws'] * chain['mb_size'], dtype=int)}")
    # print(f"  ------------>: {mask}")

    selections = onp.copy(chain['random_indices'][start_idx:end_idx])
    chain['draws'] += 1

    return selections, mask

from jax_sgmc.data.core import random_reference_data
x = onp.arange(10)
dl = NumpyDataLoader(x=x)
init_fn, batch_fn = random_reference_data(dl, 20, 3)

# chain_a = init_fn(shuffle=True)
chain_b = init_fn(shuffle=True, in_epochs=True)
