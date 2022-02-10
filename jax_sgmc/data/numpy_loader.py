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

"""Load numpy arrays in jit-compiled functions.


The numpy data loader is easy to use if the whole dataset fits into RAM and is
already present as numpy-arrays.

.. doctest::

  >>> import numpy as onp
  >>> from jax_sgmc import data
  >>> from jax_sgmc.data.numpy_loader import NumpyDataLoader

First we set up the dataset. This is very simply, as each array can be asigned
as a keyword argument to the dataloader. The keywords of the single arrays form
the keys of the pytree-dict, bundling all observations.

  >>> # The arrays must have the same length along the first dimension,
  >>> # corresponding to the total observation count
  >>> x = onp.arange(10)
  >>> y = onp.zeros((10, 2))
  >>>
  >>> data_loader = NumpyDataLoader(x_r=x, y_r=y)

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

The Numpy Data Loader accepts keyword arguments in
the init function to determnine the starting points of the chains.

  >>> rd_state = rd_init(seed=0)
  >>> new_state, (rd_batch, info) = rd_batch(rd_state, information=True)
  >>> print(rd_batch)
  {'x_r': DeviceArray([8, 9], dtype=int32), 'y_r': DeviceArray([[0., 0.],
               [0., 0.]], dtype=float32)}
  >>> # If necessary, information about the total sample count can be passed
  >>> print(info)
  MiniBatchInformation(observation_count=10, mask=DeviceArray([ True,  True], dtype=bool), batch_size=2)


Random Data Access
___________________

The NumpyDataLoader provides three different methods to randomly select
oservations:

- Independent draw (default): Draw from all samples with replacement.
- Shuffling: Draw from all samples without replacement and immediately reshuffle
  if all samples have been drawn.
- Shuffling in epochs: Draw from all samples without replacement and return mask
  to mark invalid samples at the end of the epoch.

This is illustrated at a small toy-dataset, which observation count is not a
multiplicity of the batch size:

.. doctest::

  >>> import numpy as onp
  >>> from jax_sgmc import data
  >>> from jax_sgmc.data.numpy_loader import NumpyDataLoader

  >>> x = onp.arange(10)
  >>> data_loader = NumpyDataLoader(x=x)
  >>> init_fn, batch_fn = data.random_reference_data(data_loader, 2, 3)

The prefered method has to be passed when initializing the different chains:

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
  ({'x': DeviceArray([4, 6, 6], dtype=int32)}, MiniBatchInformation(observation_count=10, mask=DeviceArray([ True,  True,  True], dtype=bool), batch_size=3))
  >>> eval_fn(shuffle_chain)
  ({'x': DeviceArray([0, 4, 7], dtype=int32)}, MiniBatchInformation(observation_count=10, mask=DeviceArray([ True,  True,  True], dtype=bool), batch_size=3))
  >>> eval_fn(epoch_chain)
  ({'x': DeviceArray([5, 0, 0], dtype=int32)}, MiniBatchInformation(observation_count=10, mask=DeviceArray([ True, False, False], dtype=bool), batch_size=3))

"""

# Todo: Rework
"""

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
  >>> from jax_sgmc.data.numpy import NumpyDataLoader

First, the data loader must be set up. The mini batch size is not required to
truly divide the total observation count. This is realized by filling up the
last batch with some values, which are sorted out either automatically or
directly by the user with a provided mask.

  >>> base = jnp.arange(10)
  >>>
  >>> data_loader = NumpyDataLoader(base=base)

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

import math
import itertools
from typing import Tuple, Any, Dict, List

import numpy as onp
import jax.numpy as jnp
import jax
from jax import random

from jax_sgmc.data.core import DeviceDataLoader, HostDataLoader, DataLoader
from jax_sgmc.data.core import mini_batch_information
from jax_sgmc.data.core import tree_index
from jax_sgmc.util import Array

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

  # Todo: Provide shuffling and in_epoch_shuffling too
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
      raise ValueError("in_epochs = True can only be used for shuffle = True.")

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
    # Data slicing is the same for all methods of random and ordered access,
    # only the indices for slicing differ. The method _get_indices find the
    # correct method for the chain.
    selections_idx, selections_mask = self._get_indices(chain_id)

    # Slice the data and transform into device array.
    selected_observations: Dict[str, Array] = dict()
    for key, data in self._reference_data.items():
      if data.ndim == 1:
        selection = jnp.array(data[selections_idx,])
      else:
        selection = jnp.array(data[selections_idx,::])
      selected_observations[key] = selection
    return selected_observations, jnp.array(selections_mask, dtype=jnp.bool_)

  def _get_indices(self, chain_id: int):
    chain = self._chains[chain_id]
    if chain['type'] == 'ordered':
      index_fn = self._ordered_indices
    elif chain['in_epochs']:
      index_fn = self._shuffle_in_epochs
    elif chain['shuffle']:
      index_fn = self._shuffle_indices
    else:
      index_fn = self._draw_indices
    indices, masks = list(zip(*map(
      lambda _: index_fn(chain),
      itertools.repeat(chain_id, self._chains[chain_id]['cache_size']))))
    return onp.array(indices), onp.array(masks, dtype=onp.bool_)

  def _ordered_indices(self, chain):
    idcs = onp.arange(chain['mb_size']) + chain['idx_offset']
    # Start again at the first sample if all samples have been returned
    if chain['idx_offset'] + chain['mb_size'] > self._observation_count:
      chain['idx_offset'] = 0
    else:
      chain['idx_offset'] += chain['mb_size']
    # Simply return the first samples again if less samples remain than
    # necessary to fill the cache. For consistency also return a mask to mark
    # the samples returned double.
    mask = onp.arange(chain['mb_size']) + chain['index_offset'] < self._observation_count

    yield onp.mod(idcs, self._observation_count), mask

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
      size=chain['mb_size'],
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

    # All samples are valid
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

    selections = onp.copy(chain['random_indices'][start_idx:end_idx])
    chain['draws'] += 1

    return selections, mask
