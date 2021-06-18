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

from typing import Any, NoReturn, Union, Tuple, Callable

import jax.numpy as jnp
from jax import tree_util

from jax_sgmc import data
from jax_sgmc import scheduler

Saving = Tuple[Callable[[Any, Any], NoReturn],
               Callable[[scheduler.schedule, Any], Union[Any, NoReturn]],
               Callable[[Any], Union[Any, NoReturn]]]

class DataCollector():
  """Collects sampled data and data loader states. """

  def __init__(self):

    self._data_loaders = []

  def register_data_loader(self, data_loader: data.DataLoader):
    """Register data loader to save the state.

    The registered data loader must itself define a strategy to store and
    restore it's state form a python object.

    Args:
      data_loader: DataLoader to checkpoint or restore.
    """
    self._data_loaders.append(data_loader)

  def _save_data_loaders(self):
    pass

  def _restore_data_loaders(self):
    pass

def load(init_state, checkpoint):
  """Reconstructs an earlier checkpoint."""

# Todo: Maybe integrate into saving
def checkpoint():
  """Initializes checkpointing. """

def save(data_collector: DataCollector = None) -> Saving:
  """Initializes saving."""


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
  def init(*unused_args, **unused_kwargs) -> NoReturn:
    return None

  def save(schedule: scheduler.schedule,
           value: Any
           ) -> Any:
    return value, jnp.logical_and(schedule.burn_in, schedule.accept)

  def postprocess(saved):
    """Only return accepted samples. """
    data, keep = saved
    return tree_util.tree_map(lambda l: l[keep], data)

  return init, save, postprocess
