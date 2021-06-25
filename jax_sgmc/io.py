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

"""Save and load results. """

# Todo: If dataloader supports checkpointing, it must be able to checkpoint
#       itself.

import abc

from pathlib import Path
from functools import partial
import threading

from typing import Any, NoReturn, Union, Tuple, Callable

from collections import namedtuple

import numpy as onp

import jax.numpy as jnp
from jax import tree_util, lax

try:
  import ujson
except ModuleNotFoundError:
  ujson = None

try:
  import h5py
except ModuleNotFoundError:
  h5py = None

from jax_sgmc import data
from jax_sgmc import scheduler
from jax_sgmc.util import host_callback

PyTree = Any

saving_state = namedtuple("saving_state",
                          ["chain_id",
                           "saved_samples",
                           "data"])

Saving = Tuple[Callable[[], saving_state],
               Callable[[saving_state, scheduler.schedule, Any, Any, Any], Union[Any, NoReturn]],
               Callable[[Any], Union[Any, NoReturn]]]


class DataCollector(metaclass=abc.ABCMeta):
  """Collects sampled data and data loader states. """

  @abc.abstractmethod
  def register_data_loader(self, data_loader: data.DataLoader):
    """Register data loader to save the state. """

  @abc.abstractmethod
  def register_chain(self) -> int:
    """Register a chain to save samples from. """

  @abc.abstractmethod
  def save(self, chain_id: int, values):
    """Called with collected samples. """

  @abc.abstractmethod
  def finalize(self, chain_id: int):
    """Called after solver finished. """

  @abc.abstractmethod
  def finished(self, chain_id: int):
    """Called in main thread after jax threads have been released."""

  @abc.abstractmethod
  def checkpoint(self, chain_id: int, state):
    """Called every n'th step. """

  @abc.abstractmethod
  def resume(self):
    """Called to restore data loader state and return sample states. """


class PickleCollector(DataCollector):
  """Stores samples in pickle format. """

  def __init__(self):
    raise NotImplementedError("Not implemented yet")

  def register_data_loader(self, data_loader: data.DataLoader):
    """Register data loader to save the state. """

  def register_chain(self) -> int:
    """Register a chain to save samples from. """

  def save(self, chain_id: int, values):
    """Called with collected samples. """

  def finalize(self, chain_id: int):
    """Called after solver finished. """

  def checkpoint(self, chain_id: int, state):
    """Called every n'th step. """

  def resume(self):
    """Called to restore data loader state and return sample states. """


class JSONCollector(DataCollector):
  """Save samples in json format.

  Args:
    dir: Directory to save the collected samples.
    write_frequency: Number of samples to be collected before a file is written

  """

  def __init__(self, dir: str, write_frequency=0):
    assert ujson is not None, "ujson is required to save samples to json files"
    self._dir = Path(dir)
    self._dir.mkdir(exist_ok=True)
    self._collected_samples = []
    self._sample_count = []
    self._write_frequency = write_frequency

  def register_data_loader(self, data_loader: data.DataLoader):
    """Not supported for JSON collector. """
    raise NotImplementedError("Checkpointing is not supported by JSON loader.")

  def register_chain(self) -> int:
    """Register a chain to save samples from. """
    new_chain = len(self._collected_samples)
    self._collected_samples.append([])
    self._sample_count.append(0)
    return new_chain

  def save(self, chain_id: int, values):
    """Called with collected samples. """
    # Store new values
    self._collected_samples[chain_id].append(
      tree_util.tree_map(onp.array, values)
    )
    self._sample_count[chain_id] += 1
    # Write to file but keep collected samples in memory
    if self._write_frequency > 0:
      if (self._sample_count[chain_id] % self._write_frequency) == 0:
        self._write_file(chain_id, self._sample_count[chain_id])

  def finalize(self, chain_id: int):
    """Called after solver finished. """
    if self._sample_count[chain_id] > 0:
      self._write_file(chain_id, self._sample_count[chain_id])

  def checkpoint(self, chain_id: int, state):
    """Called every n'th step. """
    raise NotImplementedError("Checkpointing is not supported by JSON loader.")

  def resume(self):
    """Called to restore data loader state and return sample states. """
    raise NotImplementedError("Checkpointing is not supported by JSON loader.")

  def _write_file(self, chain_id, iteration):
    filename = self._dir / f"chain_{chain_id}_iteration_{iteration}.json"
    filename.touch()
    # Transform collected samples to list after chaining together
    stacked_samples = tree_util.tree_map(
      lambda *samples: onp.stack(samples, axis=0),
      *self._collected_samples[chain_id]
    )
    samples_as_list = tree_util.tree_map(
      lambda leaf: leaf.tolist(),
      stacked_samples
    )
    with open(filename, "w") as file:
      ujson.dump(samples_as_list, file)

  def finished(self, chain_id: int):
    """Simply return, nothing to wait for. """
    return None


class HDF5Collector(DataCollector):

  def __init__(self):
    assert h5py is not None, "h5py must be installed to use this DataCollector."
    self._finished = []
    self._samples = []

  def register_data_loader(self, data_loader: data.DataLoader):
    """Register data loader to save the state. """

  def register_chain(self) -> int:
    """Register a chain to save samples from. """
    chain_id = len(self._finished)
    self._finished.append(threading.Barrier(2))
    self._samples.append([])
    return chain_id

  def save(self, chain_id: int, values):
    """Called with collected samples. """
    self._samples[chain_id].append(
      tree_util.tree_map(onp.array, values))

  def finalize(self, chain_id: int):
    """Called after solver finished. """
    # Is called after all host callback calls have been processed
    self._finished[chain_id].wait()

  def finished(self, chain_id):
    self._finished[chain_id].wait()
    # Stack the samples
    stacked_samples = tree_util.tree_map(
      lambda *leaves: onp.stack(leaves, axis=0),
      *self._samples[chain_id])
    return stacked_samples

  def checkpoint(self, chain_id: int, state):
    """Called every n'th step. """

  def resume(self):
    """Called to restore data loader state and return sample states. """


  @staticmethod
  def _groupnames(tree):
    def one_flatten_step(tree):
      """Helper to flatten tree for one step if it is not a leaf."""
      return lambda t: tree != t

    def recurse(treedef, tree):
      """Recursively build the node names. """
      if tree_util.treedef_is_leaf(treedef):
        for idx in range(treedef.num_leaves):
          yield f"/{type(tree).__name__}~{idx}"
      else:
        # Get the child treedefs and the node treedef
        children = [treedef.children(),
                    tree_util.tree_flatten(
                      tree, is_leaf=one_flatten_step(tree))[0]]
        for idx, (child_treedef, child_tree) in enumerate(zip(*children)):
          for child_name in recurse(child_treedef, child_tree):
            yield f"/{type(tree).__name__}~{idx}" + child_name

    treedef = tree_util.tree_structure(tree)
    return list(recurse(treedef, tree))

  def finished(self, chain_id: int):
    """Simply return, nothing to wait for. """
    return None

class MemoryCollector(DataCollector):
  """Stores samples entirely in RAM (numpy arrays).

  The RAM is usually larger than the device array and thus allows to store a
  greater number of samples.

  """

  def __init__(self):
    self._finished = []
    self._samples = []

  def register_data_loader(self, data_loader: data.DataLoader):
    """Register data loader to save the state. """

  def register_chain(self) -> int:
    """Register a chain to save samples from. """
    chain_id = len(self._finished)
    self._finished.append(threading.Barrier(2))
    self._samples.append([])
    return chain_id

  def save(self, chain_id: int, values):
    """Called with collected samples. """
    self._samples[chain_id].append(
      tree_util.tree_map(onp.array, values))

  def finalize(self, chain_id: int):
    """Called after solver finished. """
    # Is called after all host callback calls have been processed
    self._finished[chain_id].wait()

  def finished(self, chain_id):
    self._finished[chain_id].wait()
    # Stack the samples
    stacked_samples = tree_util.tree_map(
      lambda *leaves: onp.stack(leaves, axis=0),
      *self._samples[chain_id])
    return stacked_samples

  def checkpoint(self, chain_id: int, state):
    """Called every n'th step. """

  def resume(self):
    """Called to restore data loader state and return sample states. """


def load(init_state, checkpoint):
  """Reconstructs an earlier checkpoint."""


def save(data_collector: DataCollector = None,
         checkpoint_every: int = 0
         ) -> Saving:
  """Initializes asynchronous saving of samples and checkpoints.

  Accepted samples are sent to the host and processed there. This optimizes the
  memory usage drastically and also allows to gain insight in the data while the
  simulation is running.

  Args:
    data_collector: Stateful object for data storage and serialization
    checkpoint_every: Create a checkpoint for late resuming every n iterations

  Notes:
    Checkpointing is currently not supported.

  """

  # Todo: Implement checkpointing
  if checkpoint_every != 0:
    raise NotImplementedError("Checkpointing is not supported yet.")

  # Helper functions for host_callback

  def _save(data, *unused_args):
    chain_id, data = data
    data_collector.save(chain_id, data)

  def _save_wrapper(args) -> int:
    # Use the result to count the number of saved samples. The result must be
    # used to avoid loosing the call to jax's optimizations.
    counter = host_callback.id_tap(_save, args, result=1)
    return counter

  def init(init_sample, init_checkpoint, static_information) -> saving_state:
    """Initializes the saving state.

    Args:
      init_sample: Determining shape and dtype of collected samples
      init_checkpoint: Determining shape and dtype of checkpointed states
      static_information: Information about e. g. the total count of samples
        collected.

    Returns:
      Returns initial state.
    """
    chain_id = data_collector.register_chain()
    # The count of saved samples is important to ensure that the callback
    # function is not removed by jax's optimization procedures.
    initial_state = saving_state(
      chain_id=chain_id,
      saved_samples=0,
      data=None
    )
    return initial_state

  def save(state: saving_state,
           schedule: scheduler.schedule,
           data: Any,
           scheduler_state: scheduler.scheduler_state = None,
           solver_state: Any = None):
    """Calls the data collector on the host via host callback module."""

    # Kepp the sample if it is not subject to burn in or thinning.
    keep = jnp.logical_and(schedule.burn_in, schedule.accept)

    # Save sample if samples is not subject to burn in or discarded by thinning
    saved = lax.cond(keep,
                     _save_wrapper,
                     lambda *args: 0,
                     (state.chain_id, data))

    # Todo: Implement checkpointing
    # last_checkpoint = lax.cond(time_for_checkpoint,
    #                            _checkpoint_wrapper,
    #                            lambda *args: last_checkpoint,
    #                            (scheduler_state, soler_state)

    new_state = saving_state(
      chain_id=state.chain_id,
      saved_samples=state.saved_samples + saved)
    return new_state, None

  def postprocess(state: saving_state, unused_saved):
    # Call with host callback to ensure that finalized is called after all other
    # id_tap processes were finished.
    host_callback.id_tap(
      lambda id, *unused: data_collector.finalize(id),
      state.chain_id)
    collected_samples = data_collector.finished(state.chain_id)
    return {"sample_count": state.saved_samples,
            "samples": collected_samples}

  return init, save, postprocess

# Todo: Removed the sample collection, consider removing last argument in
#       postprocess function
def no_save() -> Saving:
  """Do not save the data but return it instead.

  If the samples are small, collection on the device is possible. Samples must
  copied repeatedly.

  Returns:
    Returns a saving strategy, which keeps the samples entirely in the
    devices memory.

  """
  def init(init_sample: PyTree,
           unused_solver_state: PyTree,
           static_information: PyTree
           ) -> saving_state:
    """Initializes the saving state.

    Args:
      init_sample: Determining shape and dtype of collected samples
      init_checkpoint: Determining shape and dtype of checkpointed states
      static_information: Information about e. g. the total count of samples
        collected.

    Returns:
      Returns initial state.
    """
    def init_zeros(leaf):
      shape = leaf.shape
      new_shape = tuple(onp.append(static_information.samples_collected, shape))
      new_shape = tree_util.tree_map(int, new_shape)
      return jnp.zeros(new_shape, dtype=leaf.dtype)

    init_data = tree_util.tree_map(init_zeros, init_sample)

    # The chain id is unnecessary
    return saving_state(chain_id=0, saved_samples=0, data=init_data)

  def _update_data_leaf(idx, data_leaf, new_slice):
    return data_leaf.at[idx].set(new_slice)

  def _save_sample(args):
    state, sample = args
    new_data = tree_util.tree_map(
      partial(_update_data_leaf, state.saved_samples),
      state.data,
      sample
    )
    new_state = saving_state(
      chain_id=state.chain_id,
      saved_samples=state.saved_samples + 1,
      data=new_data)
    return new_state

  def save(state: saving_state,
           schedule: scheduler.schedule,
           sample: Any,
           **unused_kwargs: Any
           ) -> Any:
    """Determine whether a sample should be saved.

    A sample will be saved it is not subject to burn in and not discarded due
    to thinning.

    """
    keep_sample = jnp.logical_and(schedule.burn_in, schedule.accept)
    new_state = lax.cond(keep_sample,
                         _save_sample,
                        lambda args: args[0],
                         (state, sample))

    return new_state, None

  def postprocess(state: saving_state, unused_saved):
    """Return the accepted samples. """
    return {"sample_count": state.saved_samples,
            "samples": state.data}

  return init, save, postprocess
