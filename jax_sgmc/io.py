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

import abc

from pathlib import Path

from typing import Any, NoReturn, Union, Tuple, Callable

from collections import namedtuple

import jax.numpy as jnp
from jax import tree_util, lax

try:
  import ujson
except ModuleNotFoundError:
  ujson = None

from jax_sgmc import data
from jax_sgmc import scheduler
from jax_sgmc.util import host_callback

saving_state = namedtuple("saving_state",
                          ["chain_id",
                           "saved_samples"])

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
    self._collected_samples.append(None)
    self._sample_count.append(0)
    return new_chain

  def save(self, chain_id: int, values):
    """Called with collected samples. """
    # Store new values
    if self._collected_samples[chain_id] is None:
      self._collected_samples[chain_id] = tree_util.tree_map(
        lambda leaf: [leaf.tolist()], values)
    else:
      tree_util.tree_map(
        lambda leaf, leaflist: leaflist.append(leaf.tolist()),
        values,
        self._collected_samples[chain_id]
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
    with open(filename, "w") as file:
      ujson.dump(self._collected_samples[chain_id], file)


class MemoryCollector(DataCollector):
  """Stores samples entirely in RAM (numpy arrays).

  The RAM is usually larger than the device array and thus allows to store a
  greater number of samples.

  """

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

  def init() -> saving_state:
    """Initializes the saving state. """
    chain_id = data_collector.register_chain()
    # The count of saved samples is important to ensure that the callback
    # function is not removed by jax's optimization procedures.
    initial_state = saving_state(
      chain_id=chain_id,
      saved_samples=0
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
    return {"sample_count": state.saved_samples,
            "samples": None}

  return init, save, postprocess


def no_save() -> Saving:
  """Do not save the data but return it instead.

  If the samples are small, collection on the device is possible. And not
  accepted samples, e. g. due to burn in are discarded after all samples were
  collected.

  Returns:
    Returns a saving strategy. The save function simply returns the passed data
    and the information, whether the data should be kept. The postprocessing
    function discards all data which should not be kept.

  """
  def init() -> saving_state:
    # The chain id is unnecessary
    return saving_state(chain_id=0, saved_samples=0)

  def save(unused_state: saving_state,
           schedule: scheduler.schedule,
           sample: Any,
           **unused_kwargs: Any
           ) -> Any:
    """Determine whether a sample should be saved.

    A sample will be saved it is not subject to burn in and not discareded due
    to thinning.

    """
    keep_sample = jnp.logical_and(schedule.burn_in, schedule.accept)
    return unused_state, (sample, keep_sample)

  def postprocess(unused_state: saving_state, saved):
    """Return the accepted samples. """
    samples, keep = saved
    return {"sample_count": jnp.sum(keep),
            "samples": tree_util.tree_map(lambda l: l[keep], samples)}

  return init, save, postprocess
