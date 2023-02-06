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

"""Save and checkpoint chains.

  **JaxSGMC** supports saving and checkpointing inside jit-compiled
  functions. Saving works by combining a Data Collector with the host callback
  wrappers.

"""

# Todo: If dataloader supports checkpointing, it must be able to checkpoint
#       itself.

import abc
import itertools

from pathlib import Path
from functools import partial
import threading

from typing import Any, Union, Tuple, Callable, Type, Dict, NoReturn

from collections import namedtuple

import numpy as onp

import jax.numpy as jnp
from jax import tree_util, lax
from jax.experimental import host_callback

try:
  import ujson
except ModuleNotFoundError:
  ujson = None

try:
  import h5py
  HDF5File = h5py.File
except ModuleNotFoundError:
  HDF5File = None
  h5py = None

# Import haiku to register a rule for transforming FlatMapping into a dict
try:
  import haiku._src.data_structures as haiku_ds
except ModuleNotFoundError:
  haiku_ds = None

from jax_sgmc import data
from jax_sgmc import scheduler
from jax_sgmc.util import stop_vmap

PyTree = Any

# Global rules for translating a tree-node into a dict
_dictionize_rules: Dict[Type, Callable] = {}

def register_dictionize_rule(type: Type) -> Callable[[Callable], None]:
  """Decorator to define new rules transforming a pytree node to a dict.

  By default, transformations are defined for some default types:

    - list
    - dict
    - (named)tuple

  Additionally, transformation for the following optional libraries are
  implemented:

  - haiku._src.data_structures.FlatMapping

  Args:
    type: Type (or class) of the currently undefined node

  Returns:
    The decorated function is not intended to be used directly.

  """
  def register_function(rule: Callable):
    global _dictionize_rules
    assert type not in _dictionize_rules.keys(), f"Rule for {type.__name__} is " \
                                                 f"already defined."
    _dictionize_rules[type] = rule
  return register_function

def _default_dictionize(node):
  leaves = tree_util.tree_leaves(node)
  leaf_names = [f"unknown~pytree~leaf~{idx}" for idx in range(len(leaves))]
  return zip(leaf_names, leaves)

def _dictionize(node):
  """Apply the registered rules to a pytree node."""
  global _dictionize_rules
  # If there are no leaves, then a dictionize rule is not required
  if len(tree_util.tree_leaves(node)) == 0:
    return []
  for node_type, node_rule in _dictionize_rules.items():
    if isinstance(node, node_type):
      return node_rule(node)

  return _default_dictionize(node)

def pytree_to_dict(tree: PyTree):
  """Constructs a dictionary from a pytree.

  Transforms each node of a pytree to a dict by appling defined dictionize. New
  rules can be specified by the :func:`jax_sgmc.io.register_dictionize_rule`
  -decorator.

  .. doctest::

    >>> from jax_sgmc import io
    >>> some_tree = {'a': 0.0, 'b': {'b1': [0.0, 0.1], 'b2': 0.0}}
    >>> as_dict = io.pytree_to_dict(some_tree)
    >>> print(as_dict)
    {'a': 0.0, 'b': {'b1': {'list_element_0': 0.0, 'list_element_1': 0.1}, 'b2': 0.0}}

  Args:
    tree: All nodes of the tree must either have no children or a registered
      transformation to dict.

  Returns:
    Returns the tree as a dict with similar structure.

  """
  global _dictionize_rules
  pytree_def = tree_util.tree_structure(tree)
  if tree_util.treedef_is_leaf(pytree_def):
    return tree
  else:
    return {key: pytree_to_dict(childs)
            for key, childs
            in _dictionize(tree)}

def pytree_dict_keys(tree: PyTree):
  """Returns a list of keys to acces the leaves of the tree.

  Args:
    tree: Pytree as a dict

  Returns:
    Returns a list of tuples, where each tuple contains the keys to access the
    leaf of the flattened pytree in the unflattened dict.

    For example:

    .. doctest::

      >>> from jax import tree_leaves
      >>> from jax_sgmc import io
      >>> pytree = {"a": [0.0, 1.0], "b": 2.0}
      >>> pytree_as_dict = io.pytree_to_dict(pytree)
      >>> pytree_leaves = tree_leaves(pytree_as_dict)
      >>> pytree_keys = io.pytree_dict_keys(pytree_as_dict)
      >>> print(pytree_leaves)
      [0.0, 1.0, 2.0]
      >>> print(pytree_keys)
      [('a', 'list_element_0'), ('a', 'list_element_1'), ('b',)]

  """
  node = pytree_to_dict(tree)
  leaves, treedef = tree_util.tree_flatten(node)
  idx_tree = tree_util.tree_unflatten(treedef, list(range(len(leaves))))
  key_list = [None] * len(leaves)
  def _recurse(node, path):
    if node is None:
      return
    elif isinstance(node, int):
      key_list[node] = path
    else:
      for key, value in node.items():
        _recurse(value, path + [key])
  _recurse(idx_tree, [])
  return [tuple(key) for key in key_list]

def dict_to_pytree(pytree_as_dict: dict, target: PyTree):
  """Restores the original tree structure given by the target from a dict.

  Restores the pytree as a dict to its original tree structure. This function
  can also operate on subtrees, as long as the subtree (a dict) of the pytree
  as dict matches the subtree of the target dict.

  .. doctest::

    >>> from jax_sgmc import io
    >>> some_tree = {'a': 0.0, 'b': {'b1': [0.0, 0.1], 'b2': 0.0}}
    >>> as_dict = io.pytree_to_dict(some_tree)
    >>> sub_pytree = io.dict_to_pytree(as_dict['b'], some_tree['b'])
    >>> print(sub_pytree)
    {'b1': [0.0, 0.1], 'b2': 0.0}

  Args:
    pytree_as_dict: A pytree which has been transformed to a dict of dicts.
    target: A pytree defining the original tree structure.

  """
  target_dict_keys = pytree_dict_keys(target)
  tree_structure = tree_util.tree_structure(target)
  def _recurse_get(key_list):
    key_list = list(key_list)
    element = pytree_as_dict
    while len(key_list) > 0:
      element = element.get(key_list.pop(0))
    return element
  new_leaves = map(_recurse_get, target_dict_keys)
  return tree_util.tree_unflatten(tree_structure, new_leaves)

@register_dictionize_rule(dict)
def _dict_to_dict(some_dict: dict):
  return some_dict.items()

@register_dictionize_rule(list)
def _list_to_dict(some_list: list):
  return ((f"list_element_{idx}", value) for idx, value in enumerate(some_list))

@register_dictionize_rule(tuple)
def _namedtuple_to_dict(some_tuple: Union[tuple, namedtuple]):
  if hasattr(some_tuple, '_fields'):
    return some_tuple._asdict().items()
  else:
    return ((f"list_element_{idx}", value) for idx, value in enumerate(some_tuple))

if haiku_ds is not None:
  @register_dictionize_rule(haiku_ds.FlatMapping)
  def _flat_mapping_to_dict(flat_mapping: haiku_ds.FlatMapping):
    return haiku_ds.to_immutable_dict(flat_mapping).items()

saving_state = namedtuple("saving_state",
                          ["chain_id",
                           "saved_samples",
                           "data"])

Saving = Tuple[Callable[[PyTree, PyTree, PyTree], saving_state],
               Callable[[saving_state,
                         jnp.bool_,
                         PyTree,
                         PyTree,
                         PyTree],
                        Union[Any, NoReturn]],
               Callable[[Any], Union[Any, NoReturn]]]

class DataCollector(metaclass=abc.ABCMeta):
  """Collects sampled data and data loader states. """

  @abc.abstractmethod
  def register_data_loader(self, data_loader: data.DataLoader):
    """Register data loader to save the state. """

  @abc.abstractmethod
  def register_chain(self,
                     init_sample: PyTree = None,
                     init_checkpoint: PyTree = None,
                     static_information: PyTree = None
                     ) -> int:
    """Registers a chain to save samples from.

    Args:
      init_sample: Determining shape, dtype and tree structure of sample.
      init_checkpoint: Determining shape, dtype and tree structure of solver
        state to enable checkpointing.
      static_information: Information about the total number of collected
        samples

    Returns:
      Returns id of the new chain.
    """

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
    """Called every nth step. """

  @abc.abstractmethod
  def resume(self):
    """Called to restore data loader state and return sample states. """

# Todo: Maybe remove? Not usable for big data -> better hdf5
class JSONCollector(DataCollector):
  """Saves samples in json format.

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

  def register_chain(self,
                     init_sample: PyTree = None,
                     init_checkpoint: PyTree = None,
                     static_information: PyTree = None
                     ) -> int:
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
    """Called every nth step. """
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
      ujson.dump(samples_as_list, file) # pylint: disable=c-extension-no-member

  def finished(self, chain_id: int):
    """Simply return, nothing to wait for. """
    return None


class HDF5Collector(DataCollector):
  """Save to hdf5 format.

  This data collector supports serializing collected samples and checkpoints
  into the hdf5 file format. The samples are saved in a structure similar to the
  original pytree and can thus be viewed easily via the hdf5-viewer.

  Note:
    This class requires that ``h5py`` is installed. Additional information
    can be found in the :ref:`installation instructions<additional_requirements>`.

  .. doctest::

    >>> import tempfile
    >>> tf = tempfile.TemporaryFile()
    >>>
    >>> import h5py
    >>> from jax_sgmc import io
    >>>
    >>> file = h5py.File(tf, "w") # Use a real file for real saving
    >>> data_collector = io.HDF5Collector(file)
    >>> saving = io.save(data_collector)
    >>>
    >>> # ... use the solver ...
    >>>
    >>> # Close the file
    >>> file.close()

  Args:
    file: hdf5 file object

  """

  def __init__(self, file: HDF5File):
    assert h5py is not None, "h5py must be installed to use this DataCollector."
    # Barrier to wait until all data has been processed
    self._finished = []
    self._sample_count = []
    self._leaf_names = []
    self._file = file

  def register_data_loader(self, data_loader: data.DataLoader):
    """Registers data loader to save the state. """

  def register_chain(self,
                     init_sample: PyTree = None,
                     init_checkpoint: PyTree = None,
                     static_information: PyTree = None
                     ) -> int:
    """Registers a chain to save samples from.

    Args:
      init_sample: Pytree determining tree structure and leafs of samples to be
        saved.
      init_checkpoint: Pytree determining tree structure and leafs of the states
        to be checkpointed
      static_information: Information about the total numbers of samples
        collected.
    """
    assert init_sample is not None, "Need a sample-pytree to allocate memory."
    assert init_checkpoint is not None, "Need a checkpoint-pytree to allocate " \
                                        "memory."

    chain_id = len(self._finished)
    # Save the pytree as it would be transformed to a dict by pytree_to_dict
    leaf_names = ["/".join(itertools.chain([f"/chain~{chain_id}"], key_tuple))
                  for key_tuple in pytree_dict_keys(init_sample)]
    leaves = tree_util.tree_leaves(init_sample)
    # Allocate memory upfront, this is more effective than the chunk-wise
    # allocation
    for leaf_name, leaf in zip(leaf_names, leaves):
      new_shape = tuple(int(s) for s in
                        itertools.chain([static_information.samples_collected],
                                        leaf.shape))
      self._file.create_dataset(
        leaf_name,
        shape=new_shape,
        dtype=leaf.dtype)
    self._sample_count.append(0)
    self._leaf_names.append(leaf_names)
    self._finished.append(threading.Barrier(2))
    return chain_id

  def save(self, chain_id: int, values):
    """Saves new leaves to dataset.

    Args:
      chain_id: ID from register_chain
      values: Tree leaves of sample

    """
    for leaf_name, value in zip(self._leaf_names[chain_id], values):
      self._file[leaf_name][self._sample_count[chain_id]] = value
    self._sample_count[chain_id] += 1

  def finalize(self, chain_id: int):
    """Waits for all writing to be finished (scheduled via host_callback).

    Args:
      chain_id: Id of the chain which is finished

    """
    # Is called after all host callback calls have been processed
    # self._finished[chain_id].wait()

  def checkpoint(self, chain_id: int, state):
    """Called every nth step. """

  def resume(self):
    """Called to restore data loader state and return sample states. """

  def finished(self, chain_id: int):
    """Returns after everything has been written to the file.

    Finalize is scheduled via host_callback and finished is called in the normal
    python flow. Via a barrier it is possible to pause the program flow until
    all asynchronously saved data has been processed.

    Args:
      chain_id: ID of chain requesting continuation of normal program flow.

    """
    # self._finished[chain_id].wait()

class MemoryCollector(DataCollector):
  """Stores samples entirely in RAM (numpy arrays).

  The RAM is usually larger than the device array and thus allows to store a
  greater number of samples.

  Args:
    save_dir: Directory to output results as numpy-npz with one file per chain.
      If none, the results will only be returned.

  """

  def __init__(self, save_dir=None):
    self._finished = []
    self._samples = []
    self._samples_count = []
    self._treedefs = []
    self._leafnames = []
    self._dir = save_dir

  def register_data_loader(self, data_loader: data.DataLoader):
    """Registers data loader to save the state. """

  def register_chain(self,
                     init_sample: PyTree = None,
                     static_information = None,
                     **unused_kwargs) -> int:
    """Registers a chain to save samples from.

    Args:
      init_sample: Pytree determining tree structure and leafs of samples to be
        saved.
      static_information: Information about the total numbers of samples
        collected.
    """
    assert init_sample is not None, "Need a sample-pytree to allocate memory."

    chain_id = len(self._finished)
    leaves, treedef = tree_util.tree_flatten(init_sample)
    def leaf_shape(leaf):
      new_shape = onp.append(
        static_information.samples_collected,
        leaf.shape)
      new_shape = tuple(int(s) for s in new_shape)
      return new_shape
    sample_cache = [onp.zeros(leaf_shape(leaf), dtype=leaf.dtype) for leaf in leaves]

    # Only generate the keys for each leaf if necessary
    if self._dir:
      pytree_keys = pytree_dict_keys(init_sample)
    else:
      pytree_keys = [f"leaf~{idx}" for idx in range(len(leaves))]

    self._finished.append(threading.Lock())
    self._finished[chain_id].acquire()
    self._samples.append(sample_cache)
    self._treedefs.append(treedef)
    self._samples_count.append(0)
    self._leafnames.append(["/".join(key_tuple) for key_tuple in pytree_keys])
    return chain_id

  def save(self, chain_id: int, values):
    """Saves new leaves to dataset.

    Args:
      chain_id: ID from register_chain
      values: Tree leaves of sample

    """
    sample_cache = self._samples[chain_id]
    for leaf, value in zip(sample_cache, values):
      leaf[self._samples_count[chain_id]] = value
    self._samples_count[chain_id] += 1

  def finalize(self, chain_id: int):
    """Waits for all writing to be finished (scheduled via host_callback).

    Args:
      chain_id: ID of the chain which is finished

    """
    # Is called after all host callback calls have been processed
    self._finished[chain_id].release()
    if self._dir:
      output_dir = Path(self._dir)
      output_file = output_dir / f"chain_{chain_id}.npz"
      output_dir.mkdir(exist_ok=True)
      onp.savez(
        output_file,
        **dict(zip(self._leafnames[chain_id], self._samples[chain_id])))

  def finished(self, chain_id):
    """Returns samples after all data has been processed.

    Finalize is scheduled via host_callback and finished is called in the normal
    python flow. Via a barrier it is possible to pause the program flow until
    all asynchronously saved data has been processed.

    Args:
      chain_id: ID of chain requesting continuation of normal program flow.

    Returns:
      Returns the collected samples in the original tree format but with numpy-
      arrays as leaves.

    """
    self._finished[chain_id].acquire()
    self._finished[chain_id].release()
    # Restore original tree shape
    return tree_util.tree_unflatten(
      self._treedefs[chain_id],
      self._samples[chain_id])

  def checkpoint(self, chain_id: int, state):
    """Called every nth step. """

  def resume(self):
    """Called to restore data loader state and return sample states. """


def load(init_state, checkpoint):
  """Reconstructs an earlier checkpoint."""
  raise NotImplementedError("Checkpointing is currently not supported.")

def save(data_collector: DataCollector = None,
         checkpoint_every: int = 0
         ) -> Saving:
  """Initializes asynchronous saving of samples and checkpoints.

  Accepted samples are sent to the host and processed there. This optimizes the
  memory usage drastically and also allows gaining insight in the data while the
  simulation is running.

  Returns statistics and samples depending on the Data Collector. For
  example hdf5 can be used for samples larger than the (device-)memory.
  Therefore, no samples are returned. Instead, the memory collector returns
  the samples collected as numpy arrays.

  Example usage:

    .. doctest::

      >>> import jax.numpy as jnp
      >>> from jax.lax import scan
      >>> from jax_sgmc import io, scheduler
      >>>
      >>> dc = io.MemoryCollector()
      >>> init_save, save, postprocess_save = io.save(dc)
      >>>
      >>> def update(saving_state, it):
      ...   saving_state = save(saving_state, jnp.mod(it, 2) == 0, {'iteration': it})
      ...   return saving_state
      >>>
      >>> # The information about the number of collected samples must be defined
      >>> # before the run
      >>> static_information = scheduler.static_information(samples_collected=3)
      >>>
      >>> # The saving function must now the form of the sample which should be saved
      >>> saving_state = init_save({'iteration': jnp.array(0)}, {}, static_information)
      >>> final_state, _ = scan(update, saving_state, jnp.arange(5))
      >>>
      >>> saved_samples = postprocess_save(final_state, None)
      >>> print(saved_samples)
      {'sample_count': Array(3, dtype=int32, weak_type=True), 'samples': {'iteration': array([0, 2, 4], dtype=int32)}}


  Args:
    data_collector: Stateful object for data storage and serialization
    checkpoint_every: Create a checkpoint for late resuming every n iterations

  Warning:
    Checkpointing is currently not supported.

  Returns:
    Returns a saving strategy.

  """

  # Todo: Implement checkpointing
  if checkpoint_every != 0:
    raise NotImplementedError("Checkpointing is not supported yet.")

  # Helper functions for host_callback

  def _save(data, *unused_args):
    chain_ids, data = data

    # id_tap sends batched arguments to the host
    if chain_ids.ndim == 0:
      data_collector.save(chain_ids, data)
    else:
      for idx, chain_id in enumerate(chain_ids):
        data_collector.save(chain_id, [leaf[idx] for leaf in data])

  def _save_wrapper(args) -> int:
    # Use the result to count the number of saved samples. The result must be
    # used to avoid losing the call to Jax's optimizations.
    # Only return the leaves, as tree structure is redundant and requires
    # flattening on the host.
    chain_id, data = args
    flat_args = (chain_id, tree_util.tree_leaves(data))
    counter = host_callback.id_tap(_save, flat_args, result=1)
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
    chain_id = data_collector.register_chain(
      init_sample=init_sample,
      init_checkpoint=init_checkpoint,
      static_information=static_information)
    # The count of saved samples is important to ensure that the callback
    # function is not removed by jax's optimization procedures.
    initial_state = saving_state(
      chain_id=chain_id,
      saved_samples=0,
      data=None)
    return initial_state

  @stop_vmap.stop_vmap
  def _save_helper(keep, state, sample):
    return lax.cond(keep,
                    _save_wrapper,
                    lambda *args: 0,
                    (state.chain_id, sample))

  # Todo: Generalize the saving by contracting the scheduler state and the
  #       solver state to a single checkpointing state.
  def save(state: saving_state,
           keep: jnp.bool_,
           sample: Any,
           scheduler_state: scheduler.scheduler_state = None,
           solver_state: Any = None):
    """Calls the data collector on the host via host callback module."""

    # Save sample if samples is not subject to burn in or discarded by thinning
    saved = _save_helper(keep, state, sample)

    # Todo: Implement checkpointing
    # last_checkpoint = lax.cond(time_for_checkpoint,
    #                            _checkpoint_wrapper,
    #                            lambda *args: last_checkpoint,
    #                            (scheduler_state, soler_state)

    new_state = saving_state(
      chain_id=state.chain_id,
      saved_samples=state.saved_samples + saved,
      data=None)
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
  """Does not save the data on the host but return it instead.

  If the samples are small, collection on the device is possible. Samples must
  be copied repeatedly.

  Save keep every second element of a 5 element scan

  .. doctest::

    >>> import jax.numpy as jnp
    >>> from jax.lax import scan
    >>> from jax_sgmc import io, scheduler
    >>>
    >>> init_save, save, postprocess_save = io.no_save()
    >>>
    >>> def update(saving_state, it):
    ...   saving_state = save(saving_state, jnp.mod(it, 2) == 0, {'iteration': it})
    ...   return saving_state
    >>>
    >>> # The information about the number of collected samples must be defined
    >>> # before the run
    >>> static_information = scheduler.static_information(samples_collected=3)
    >>>
    >>> # The saving function must now the form of the sample which should be saved
    >>> saving_state = init_save({'iteration': jnp.array(0)}, None, static_information)
    >>> final_state, _ = scan(update, saving_state, jnp.arange(5))
    >>>
    >>> saved_samples = postprocess_save(final_state, None)
    >>> print(saved_samples)
    {'sample_count': Array(3, dtype=int32, weak_type=True), 'samples': {'iteration': Array([0, 2, 4], dtype=int32)}}

  Returns:
    Returns a saving strategy, which keeps the samples entirely in the
    device's memory.

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
      sample)
    new_state = saving_state(
      chain_id=state.chain_id,
      saved_samples=state.saved_samples + 1,
      data=new_data)
    return new_state

  @stop_vmap.stop_vmap
  def _save_helper(keep, state, sample):
    return lax.cond(keep,
                    _save_sample,
                    lambda args: args[0],
                    (state, sample))

  def save(state: saving_state,
           keep: jnp.bool_,
           sample: Any,
           **unused_kwargs: Any
           ) -> Any:
    """Determines whether a sample should be saved.

    A sample will be saved it is not subject to burn in and not discarded due
    to thinning.

    """
    new_state = _save_helper(keep, state, sample)
    return new_state, None

  def postprocess(state: saving_state, unused_saved):
    """Return the accepted samples. """
    return {"sample_count": state.saved_samples,
            "samples": state.data}

  return init, save, postprocess
