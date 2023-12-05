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

Limited device memory disallows to store all reference data on the device for
big datasets. With the following functions, mini-batches of data can be
requested in ``jit``-compiled functions without loading the entire dataset into
the device's memory.

In the background, a cache of mini-batches is sequentially requested by the
Host Callback Wrappers from the Data Loaders and loaded on the device via the
:mod:`jax_sgmc.util.host_callback` module.

"""

import abc
import threading

import warnings

from functools import partial
import itertools

from typing import Tuple, Any, Callable, Union, NamedTuple, Dict, Protocol, List

import jax
from jax import tree_util, lax
import jax.numpy as jnp
from jax.experimental import host_callback as hcb

import numpy as onp


from jax_sgmc.util import Array, stop_vmap
from jax_sgmc.util.uuid import JaxUUID

class MiniBatchInformation(NamedTuple):
  """Bundles all information about the reference data.

  Args:
    observation_count: Total number of observations
    effective_observation_count: The number of observations without the
      discarded samples remaining after e.g. shuffling.
    mini_batch: List of tuples, tuples consist of ``(observations, parameters)``

  """
  observation_count: Array
  mask: Array
  batch_size: Array


PyTree = Any
MiniBatch = Union[Tuple[PyTree],
                  Tuple[PyTree, MiniBatchInformation],
                  Tuple[PyTree, MiniBatchInformation, Array]]

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
    """dtype and shape of a single sample."""

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
                      ) ->Tuple[PyTree, Tuple[PyTree, MiniBatchInformation]]:
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
    """Register a new chain which assembles batches randomly.

    Args:
      cache_size: The number of drawn batches.
      mb_size: The number of observations per batch.
      seed: Set the random seed to start the chain at a well-defined state.

    Returns:
      Returns the id of the new chain.

    """

  @abc.abstractmethod
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


  @abc.abstractmethod
  def get_batches(self, chain_id: int) -> Tuple[PyTree, Union[Array, None]]:
    """Return batches from an ordered or random chain. """

  def batch_format(self,
                   cache_size: int,
                   mb_size: int,
                   ) -> Tuple[PyTree, MiniBatchInformation]:
    """Returns dtype and shape of cached mini-batches.

    Args:
      cache_size: number of cached mini-batches

    Returns:
      Returns a pytree with the same tree structure as the random data cache but
      with ``jax.ShapedDtypeStruct`` as leaves.
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
      batch_size=mb_size,
      mask=onp.ones(mb_size, dtype=onp.bool_))
    return format, mb_info

class CacheState(NamedTuple):
  """Caches several batches of randomly batched reference data.

  Args:
    cached_batches: An array of mini-batches
    cached_batches_count: Number of cached mini-batches. Equals the first
      dimension of the cached batches
    current_line: Marks the next batch to be returned.
    chain_id: Identifier of the chain
    state: Additional information
    valid: Array containing information about the validity of individual samples
  """
  callback_uuid: JaxUUID = None
  cached_batches: PyTree = None
  cached_batches_count: Array = None
  current_line: Array = None
  chain_id: Array = None
  state: PyTree = None
  valid: Array = None
  token: JaxUUID = None

random_data_state = CacheState

Batch = Union[Tuple[CacheState, PyTree],
              Tuple[CacheState, Tuple[PyTree, MiniBatchInformation]]]

class GetBatchFunction(Protocol):
  def __call__(self,
               data_state: CacheState,
               information: bool = False,
               device_count: int = 1) -> Batch:
    """Draws a batch of data.

    Args:
      data_state: State of the chain containing id and cached batches
      information: Include namedtuple containing information about the data
        and batch
      device_count: Number of the devices on which this function is going to be
        called with replicated data states.

    Returns:
      Returns the new state of the random chain and a batch. Optionally a
      namedtuple containing information about the batch and dataset can be
      returned.

    """

class MaskedMappedFunction(Protocol):
  def __call__(self,
               batch: PyTree,
               mask: Array,
               state: PyTree
               ) -> Tuple[PyTree, PyTree]:
    """Function which can be mapped over the whole dataset.

    A function of this form must be passed to the full data map function if it
    is called with ``masking = True``.

    Args:
      batch: Batch of data
      mask: Array marking invalid (double) samples
      state: Variables which results are used in the next computation

    Returns:
      Must return a tuple consisting of the computation results and the state
      which should be used in the computation of the next batch.

    """

class UnmaskedMappedFunction(Protocol):
  def __call__(self,
               batch: PyTree,
               state: PyTree
               ) -> Tuple[PyTree, PyTree]:
    """Function which can be mapped over the whole dataset.

    A function of this form must be passed to the full data map function if it
    is called with ``masking = False``.

    Args:
      batch: Batch of data
      state: Variables which results are used in the next computation

    Returns:
      Must return a tuple consisting of the computation results and the state
      which should be used in the computation of the next batch.

    """

MappedFunction = Union[MaskedMappedFunction, UnmaskedMappedFunction]

class FullDataMapFunction(Protocol):
  def __call__(self,
               fun: MappedFunction,
               data_state: CacheState,
               carry: PyTree,
               masking: bool = False,
               information: bool = False,
               device_count: int = 1
               ) -> Tuple[PyTree, PyTree]:
    """Maps a function over the complete dataset.

    Args:
      fun: Function to be mapped over the dataset
      data_state: Namedtuple containing the id of the chain and cached batches
      carry: Variables which are carried over to the next evaluation of ``fun``
      masking: If true, an array marking invalid samples is passed to the
        function such that a single result for a batch of data can be
        calculated. If false, then a result for each observation must be
        returned and the invalid results are discarded after the computation.
      information: Pass the batch information together with the batch
      device_count: Number of the devices on which this function is going to be
        called with replicated data states.

    Returns:
      Returns the new data state and the results of the computation including
      the carry of the last computation:

      ::

        (data_state, (results, carry)) = full_data_map(...)

    """

class FullDataMapperFunction(Protocol):
  def __call__(self,
               fun: MappedFunction,
               carry: PyTree,
               masking: bool = False,
               information: bool = False,
               batched: bool = True,
               device_count: int = 1
               ) -> PyTree:
    """Maps a function over the complete dataset.

    This function differs to :class:`FullDataMapFunction` that it acquires a
    :class:`CacheState` before each mapping over the full dataset.

    Args:
      fun: Function to be mapped over the dataset
      carry: Variables which are carried over to the next evaluation of ``fun``
      masking: If true, an array marking invalid samples is passed to the
        function such that a single result for a batch of data can be
        calculated. If false, then a result for each observation must be
        returned and the invalid results are discarded automatically after the
        computation.
      information: Pass the batch information together with the batch
      batched: Whether function to be mapped over full dataset is vectorized. If
        false, the function is vmapped such that it can process a batch of
        observations.
      device_count: Number of the devices on which this function is going to be
        called with replicated data states.

    Returns:
      Returns the results of the computation including the carry of the last
      computation:

      ::

        (results, carry) = full_data_mapper(...)

    """

RandomBatch = Tuple[Any, GetBatchFunction, Callable[[], None]]
OrderedBatch = Tuple[Any, FullDataMapFunction, Callable[[], None]]

class _Requests:

  def __init__(self):
    self._cached_requests: Dict[Any, List[int, CacheState, JaxUUID]] = {}
    self._requests: Dict[Any, Dict[int, CacheState, JaxUUID]] = {}
    self._host_data_loaders: Dict[Any, HostDataLoader] = {}

    self._lock = threading.Lock()

  def __call__(self,
               chain_id: int,
               token: JaxUUID,
               device: jax.Device,
               callback_uuid: JaxUUID,
               device_count: int = 1,
               strict: bool = False):
    with self._lock:
      if callback_uuid.as_uuid not in self._requests.keys():
        self._requests[callback_uuid.as_uuid] = {}

      # The token that is expected for a new callback
      current_token = self._requests[callback_uuid.as_uuid].get(int(chain_id))

      # Check if token is from previous request
      old_request = self._cached_requests.get(token.as_uuid)
      if old_request is not None:
        counter, callback_response, new_token = old_request
        if counter == 1:
          # The callback response is no longer necessary
          del self._cached_requests[token.as_uuid]
        else:
          self._cached_requests[token.as_uuid][0] -= 1

      # Check if token is invalid
      elif current_token != token.as_uuid and current_token is not None:
        if strict:
          raise RuntimeError(f"Device {device} made an invalid request for "
                             f"chain {chain_id}. This might be due to using a "
                             f"pmap in a jitted function. See "
                             f"usage/data.html#combining-pmap-and-jit in the "
                             f"docs.")
        warnings.warn(f"Device {device} made an invalid request for chain "
                      f"{chain_id}. This might be due to using a pmap in a "
                      f"jitted function. If the preservation ")

      # The request is valid and the first request, so the results have to be
      # stored in the cache
      else:
        # Issue a new token and request data
        new_token = JaxUUID()
        callback_response = self._host_data_loaders[callback_uuid.as_uuid].get_batches(chain_id)

        self._requests[callback_uuid.as_uuid][int(chain_id)] = new_token.as_uuid
        # Store data if other devices are going to request it
        if device_count != 1:
          self._cached_requests[token.as_uuid] = [
            device_count - 1, callback_response, new_token]

    return callback_response, new_token

  def __getstate__(self):
    state = self.__dict__.copy()
    # The lock should always be reinitialized
    del state['_lock']
    return state

  def __setstate__(self, state):
    self.__dict__.update(state)
    self._lock = threading.Lock()

_data_requests = _Requests()

def random_reference_data(data_loader: DataLoader,
                          cached_batches_count: int,
                          mb_size: int,
                          verify_calls: bool = False
                          ) -> RandomBatch:
  """Initializes reference data access in jit-compiled functions.

  Randomly draw batches from a given dataset on the host or the device.

  Args:
    data_loader: Reads data from storage.
    cached_batches_count: Number of batches in the cache. A larger number is
      faster, but requires more memory.
    mb_size: Size of the data batch.
    verify_calls: Verify calls to the host when using pmap. This might not be
      necessary if batches are assembles randomly without extra conditions.

  Returns:
    Returns a tuple of functions to initialize a new reference data state, get
    a minibatch from the reference data state and release the data loader after
    the last computation.

  """
  # Check batch size is not bigger than total observation count
  observation_count = data_loader.static_information["observation_count"]
  if observation_count < mb_size:
    raise ValueError(f"Batch size cannot be bigger than the number of total "
                     f"observations. Got {observation_count} and {mb_size}.")
  if cached_batches_count <= 0 or mb_size <= 0:
    raise ValueError(f"Cache size and batch size must be positive, got"
                     f"{cached_batches_count} and {mb_size}.")

  if isinstance(data_loader, HostDataLoader):
    return _random_reference_data_host(
      data_loader, cached_batches_count, mb_size,
      verify_calls=verify_calls)
  elif isinstance(data_loader, DeviceDataLoader):
    if not cached_batches_count == 1:
      raise ValueError("No caching on device.")
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
    Returns a tuple of functions to initialize a new reference data state, map a
    function over the complete dataset and release the data loader after the
    last computation.

  """
  # Check batch size is not bigger than total observation count
  observation_count = data_loader.static_information["observation_count"]
  if observation_count < mb_size:
    raise ValueError(f"Batch size cannot be bigger than the number of total "
                     f"observations. Got {observation_count} and {mb_size}.")

  if isinstance(data_loader, HostDataLoader):
    init_fn, (_batch_fn, mb_information), cleanup = _full_reference_data_host(
      data_loader, cached_batches_count, mb_size)
  elif isinstance(data_loader, DeviceDataLoader):
    init_fn, (_batch_fn, mb_information), cleanup = _full_reference_data_device(
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
                             masking=False,
                             device_count=1):
    # The mask has is 1 if the observation is valid and 0 otherwise. This is
    # necessary to ensure, that fun is always called with the same tree shape.
    observations = iteration * batch_size + jnp.arange(batch_size)
    mask = observations < mb_information.observation_count

    data_state, fun_state = state
    data_state, batch = _batch_fn(
      data_state,
      information=information, device_count=device_count)

    if masking:
      result, fun_state = fun(batch, mask, fun_state)
    else:
      result, fun_state = fun(batch, fun_state)

    return (data_state, fun_state), result

  def batch_scan(fun: MappedFunction,
                 data_state: CacheState,
                 carry: PyTree,
                 masking: bool = False,
                 information: bool = False,
                 device_count: int = 1
                 ) -> Tuple[PyTree, PyTree]:
    """Maps the function over all data.

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
                       masking=masking,
                       device_count=device_count)
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

  return init_fn, batch_scan, cleanup


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
    pytree: Tree to index with array-like leaves
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
                 mb_size: int,
                 verify_calls: bool = False
                 ) -> Tuple[GetBatchFunction, MiniBatchInformation]:
  # These are helper function which keep a reference to the stateful data object
  # and can be called via the host_callback.call function
  # The format of the mini batch is static.

  hcb_format, mb_information = data_loader.batch_format(
    cached_batches_count, mb_size=mb_size)
  mask_shape = (cached_batches_count, mb_size)

  # The definition requires passing an argument to the host function. The shape
  # of the returned data must be known before the first call. The chain id
  # determines whether the data is collected randomly or sequentially.

  def get_data(req, device):
    chain_id, callback_uuid, token, device_count = req
    # The data request class takes care of assigning the request to the right
    # data loader and verifies it.
    (new_data, mask), new_token = _data_requests(
      chain_id, token, device, callback_uuid, device_count,
      strict=verify_calls)
    if mask is None:
      # Assume all samples to be valid. It is important to perform the creation
      # of the array on the host, as otherwise a deadlock will occur.
      mask = onp.ones(mask_shape, dtype=jnp.bool_)
    return new_data, mask, new_token

  def _new_cache_fn(req: Tuple[CacheState, int]) -> CacheState:
    """This function is called if the cache must be refreshed."""
    state, device_count = req
    new_data, masks, token = hcb.call(
      get_data,
      (state.chain_id, state.callback_uuid, state.token, device_count),
      result_shape=(
        hcb_format,
        jax.ShapeDtypeStruct(shape=mask_shape, dtype=jnp.bool_),
        JaxUUID()),
      call_with_device=True)
    new_state = CacheState(
      cached_batches_count=state.cached_batches_count,
      cached_batches=new_data,
      current_line=jnp.array(0),
      chain_id=state.chain_id,
      valid=masks,
      callback_uuid=state.callback_uuid,
      token=token)
    return new_state

  def _old_cache_fn(req: Tuple[CacheState, int]) -> CacheState:
    """This function is called if the cache must not be refreshed."""
    return req[0]

  # Necessary, because cond is replaced by select under vmap, but the cond
  # branches have side effects.
  @stop_vmap.stop_vmap
  def _data_state_helper(data_state, device_count):
    return lax.cond(data_state.current_line == data_state.cached_batches_count,
                    _new_cache_fn,
                    _old_cache_fn,
                    (data_state, device_count))

  def batch_fn(data_state: CacheState,
               information: bool = False,
               device_count: int = 1
               ) -> Batch:
    """Draws a new random batch (hides data transfer between devices).

    Args:
      data_state: State with cached samples
      information: Whether to return batch information
      device_count: Number of parallel programs calling the batch function

    Returns:
      Returns the new data state and the next batch. Optionally an additional
      struct containing information about the batch can be returned.

    """
    if device_count > jax.device_count():
      raise ValueError(f"The value of device_count cannot exceed the true "
                       f"device count. Expecting device_count (given: "
                       f"{device_count}) <= {jax.device_count()}.")
    if device_count != 1:
      warnings.warn("Changing the device count can cause memory accumulation. "
                    "Continue only if you know what you are doing.")

    # Refresh the cache if necessary, after all cached batches have been used.
    data_state = _data_state_helper(data_state, device_count)
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
      valid=data_state.valid,
      callback_uuid=data_state.callback_uuid,
      token=data_state.token)

    info = MiniBatchInformation(
      observation_count = mb_information.observation_count,
      batch_size = mb_information.batch_size,
      mask = mask)

    if information:
      return new_state, (mini_batch, info)
    else:
      return new_state, mini_batch

  return batch_fn, mb_information


def _random_reference_data_host(data_loader: HostDataLoader,
                                cached_batches_count: int = 100,
                                mb_size: int = 1,
                                verify_calls: bool = False) -> RandomBatch:
  """Random reference data access via host-callback. """
  # Warn if cached_batches are bigger than total dataset
  observation_count = data_loader.static_information["observation_count"]
  if observation_count < cached_batches_count * mb_size:
    warnings.warn("Cached batches are bigger than the total dataset. Consider "
                  "using a DeviceDataLoader.")

  batch_fn, _ = _hcb_wrapper(
    data_loader, cached_batches_count, mb_size,
    verify_calls=verify_calls)

  callback_uuid = JaxUUID()
  _data_requests._host_data_loaders[callback_uuid.as_uuid] = data_loader

  def init_fn(**kwargs) -> CacheState:
    # Pass the data loader the information about the number of cached
    # mini-batches. The data loader returns an unique id for reproducibility
    chain_id = data_loader.register_random_pipeline(
      cached_batches_count,
      mb_size=mb_size,
      **kwargs)
    initial_state, initial_mask = _data_requests._host_data_loaders[callback_uuid.as_uuid].get_batches(chain_id)
    if initial_mask is None:
      initial_mask = jnp.ones((cached_batches_count, mb_size), dtype=jnp.bool_)
    inital_cache_state = CacheState(
      cached_batches=initial_state,
      cached_batches_count=jnp.array(cached_batches_count),
      current_line=jnp.array(0),
      chain_id=jnp.array(chain_id),
      valid=initial_mask,
      callback_uuid=callback_uuid,
      token=JaxUUID())
    return inital_cache_state

  def release():
    del _data_requests._host_data_loaders[callback_uuid.as_uuid]

  return init_fn, batch_fn, release


def _random_reference_data_device(data_loader: DeviceDataLoader,
                                  mb_size: int
                                  ) -> RandomBatch:
  """Random reference data on device. """

  def init_fn(**kwargs) -> CacheState:
    state = data_loader.init_random_data(**kwargs)
    return CacheState(state=state)

  def batch_fn(state: CacheState,
               information: bool = False
               ) -> Batch:

    state, (batch, mb_info) = data_loader.get_random_data(
      state.state, batch_size=mb_size)
    new_state = CacheState(state=state)

    if information:
      return new_state, (batch, mb_info)
    else:
      return new_state, batch

  def release():
    pass

  return init_fn, batch_fn, release

def _full_reference_data_host(data_loader: HostDataLoader,
                              cached_batches_count: int = 100,
                              mb_size: int = None
                              ) -> Tuple[Callable, Tuple[Callable, MiniBatchInformation], Callable]:
  """Sequentially load batches of reference data via host-callback. """
  # Warn if cached_batches are bigger than total dataset
  observation_count = data_loader.static_information["observation_count"]
  if observation_count < cached_batches_count * mb_size:
    warnings.warn("Cached batches are bigger than the total dataset. Consider "
                  "using a DeviceDataLoader.")

  batch_fn = _hcb_wrapper(
    data_loader,
    cached_batches_count,
    mb_size,
    verify_calls=True)

  # Register the data loader
  callback_uuid = JaxUUID()
  _data_requests._host_data_loaders[callback_uuid.as_uuid] = data_loader

  def init_fn(**kwargs) -> CacheState:
    # Pass the data loader the information about the number of cached
    # mini-batches. The data loader returns an unique id for reproducibility
    chain_id = _data_requests._host_data_loaders[callback_uuid.as_uuid].register_ordered_pipeline(
      cached_batches_count,
      mb_size=mb_size,
      **kwargs)
    initial_state, initial_mask = _data_requests._host_data_loaders[callback_uuid.as_uuid].get_batches(chain_id)
    if initial_mask is None:
      initial_mask = jnp.ones((cached_batches_count, mb_size), dtype=jnp.bool_)
    inital_cache_state=CacheState(
      cached_batches=initial_state,
      cached_batches_count=jnp.array(cached_batches_count),
      current_line=jnp.array(0),
      chain_id=jnp.array(chain_id),
      valid=initial_mask,
      callback_uuid=callback_uuid,
      token=JaxUUID())
    return inital_cache_state

  def release():
    del _data_requests._host_data_loaders[callback_uuid.as_uuid]

  return init_fn, batch_fn, release


def _full_reference_data_device(data_loader: DeviceDataLoader,
                                mb_size: int = None
                                ) -> Tuple[Callable,
                                           Tuple[Callable,
                                                 MiniBatchInformation],
                                           Callable]:
  """Batches the dataset on the device. """

  reference_data = data_loader.get_full_data()
  total_observations = data_loader.static_information["observation_count"]


  # The information about the batches need to be static.
  mb_info = MiniBatchInformation(
    observation_count=total_observations,
    batch_size=mb_size,
    mask=onp.ones(mb_size),
  )

  def init_fn(offset: jnp.ndarray = 0):
    if offset >= total_observations:
      raise ValueError(f"The offset cannot be greater than the total "
                       f"observation count. Given {offset} and "
                       f"{total_observations}.")

    init_state = CacheState(current_line=offset)
    return init_state

  def batch_fn(data_state: CacheState,
               information: bool = False,
               device_count: int = 1):
    del device_count
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

  def release():
    pass

  return init_fn, (batch_fn, mb_info), release


class _FullDataHelper:
  """Class to keep track of unused CacheStates. """

  def __init__(self, data_loader, cache_size, batch_size):
    self._init_fn, self._map_fn, self._cleanup_fn = full_reference_data(
      data_loader, cache_size, batch_size)

    # Initialize the first cache state to compute the shape of the CacheState
    self._unused_states = [self._new_cache_state()]
    self._cache_state_format = tree_util.tree_map(
      lambda leaf: jax.ShapeDtypeStruct(dtype=leaf.dtype, shape=leaf.shape),
      self._unused_states[0])

    self._lock = threading.Lock()

  @property
  def get_map_fn(self) -> FullDataMapFunction:
    return self._map_fn

  @property
  def get_cache_state_format(self) -> PyTree:
    return self._cache_state_format

  def get_cache_state(self) -> CacheState:
    """Return unused cache state or create new one if none available."""
    with self._lock:
      if len(self._unused_states) == 0:
        state = self._new_cache_state()
      else:
        state = self._unused_states.pop()
    return state

  def free_cache_state(self, cache_state: CacheState) -> Array:
    """Adds cache state back to the unused cache states."""
    with self._lock:
      self._unused_states.append(cache_state)
    return jnp.array(1.0)

  def _new_cache_state(self) -> CacheState:
    """Creates a new cache state. """
    new_cache_state = self._init_fn()
    return new_cache_state

  def cleanup(self):
    self._cleanup_fn()
    self._unused_states = None

  def __getstate__(self):
    state = self.__dict__.copy()
    del state["_lock"]
    return state

  def __setstate__(self, state):
    self.__dict__.update(state)
    self._lock = threading.Lock()


def full_data_mapper(data_loader: DataLoader = None,
                     cached_batches_count: int = 1,
                     mb_size: int = 1
                     ) -> Tuple[FullDataMapperFunction, Callable]:
  """Initializes a functional to map a function over a complete dataset.

  This function extends the functionality of
  :func:`full_reference_data` by loading the data states
  form the host before each mapping.

  Args:
    data_loader: Reads data from storage
    cached_batches_count: Number of batches in the cache. A larger number is
      faster, but requires more memory
    mb_size: Size of the data batch

  Returns:
    Returns a tuple of functions to map another function over a complete dataset
    of an appropriate :class:`DataLoader` and another function to release
    the data loader after the last computation.

  """

  _helper = _FullDataHelper(data_loader, cached_batches_count, mb_size)

  # Helper functions to load/save the CacheStates on the host via host_callback
  def _get_state() -> CacheState:
    cache_state = hcb.call(
      lambda _: _helper.get_cache_state(),
      jnp.array(1.0),
      result_shape=_helper.get_cache_state_format)
    return cache_state

  def _free_cache_state(cache_state: CacheState, results: PyTree) -> Array:
    # Loop-through the results to hinder XLA to remove the tap call
    results = hcb.id_tap(
      lambda cs, _: _helper.free_cache_state(cs),
      cache_state,
      result=results)
    return results

  def mapper_fn(fun: Union[MaskedMappedFunction, UnmaskedMappedFunction],
                carry: PyTree,
                masking: bool = False,
                information: bool = False,
                batched: bool = True,
                device_count: int = 1) -> PyTree:
    data_state = _get_state()

    # Batch the function if it is not batched.
    if batched:
      batched_fun = fun
    else:
      if masking:
        raise ValueError("The function must be vectorized manually to allow "
                         "masking.")

      if mb_size == 1:
        # No vmapping required but first axis of all observations must be
        # removed.
        def batched_fun(batch, state):
          squeezed_batch = tree_util.tree_map(
            partial(jnp.squeeze, axis=0),
            batch)
          result, state = fun(squeezed_batch, state)
          expanded_result = tree_util.tree_map(
            partial(jnp.expand_dims, axis=0),
                    result)
          return expanded_result, state
      else:
        # Only the first resulting state is passed to the next iteration
        vmapped_fun = jax.vmap(fun, in_axes=(0, None))
        def batched_fun(batch, state):
          results, states = vmapped_fun(batch, state)
          if states is None:
            state = None
          else:
            state = tree_util.tree_map(
              partial(jnp.take_along_axis, indices=0, axis=0),
              states)
          return results, state

    (new_data_state, results) = _helper.get_map_fn(
      batched_fun, data_state, carry,
      masking=masking, information=information, device_count=device_count)
    results = _free_cache_state(new_data_state, results)
    return results

  def release():
    _helper.cleanup()

  return mapper_fn, release
