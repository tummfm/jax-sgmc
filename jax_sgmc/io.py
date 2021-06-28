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
import itertools

from pathlib import Path
from functools import partial
import threading

from typing import Any, NoReturn, Union, Tuple, Callable, Type, Dict

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

# Import haiku to register a rule for transforming FlatMapping into a dict
try:
  import haiku._src.data_structures as haiku_ds
except ModuleNotFoundError:
  haiku_ds = None

from jax_sgmc import data
from jax_sgmc import scheduler
from jax_sgmc.util import host_callback

PyTree = Any

# Global rules for translating a tree-node into a dict
_dictionize_rules: Dict[Type, Callable] = {}

def register_dictionize_rule(type: Type) -> NoReturn:
  """Decorator to define new rules transforming a pytree node to a dict.

  By default, transformations are defined for some default types:
    - list
    - dict
    - (named)tuple

  Additionaly, transformation for the following optional libraries are
  implemented:
    - haiku._src.data_structures.FlatMapping

  New a new transformation rule is a function, which accepts a pytree node of
  a specific type an returns a iterable, which itself returns `(key, value)`-
  pairs.

  .. doctest::

    >>> from jax_sgmc import io
    >>> from jax.tree_util import register_pytree_node
    >>>
    >>> class SomeClass:
    ...   def __init__(self, value):
    ...     self._value = value
    >>>
    >>> # Do not forget to register the class as jax pytree node
    >>> register_pytree_node(SomeClass,
    ...                      lambda sc: (sc._value, None),
    ...                      lambda _, data: SomeClass(value=data))
    >>>
    >>> # Now define a rule to transform the class into a dict
    >>> @io.register_dictionize_rule(SomeClass)
    ... def some_class_to_dict(instance_of_some_class):
    ...   return [("this_is_the_key", instance_of_some_class._value)]
    >>>
    >>> some_class = SomeClass({'a': 0.0, 'b': 0.5})
    >>> some_class_as_dict = io.pytree_to_dict(some_class)
    >>>
    >>> print(some_class_as_dict)
    {'this_is_the_key': {'a': 0.0, 'b': 0.5}}

  Args:
    type: Type (or class) of the currently undefined node

  Returns:
    The decoreated function is not intended to be used directly.

  """
  def register_function(rule: Callable):
    global _dictionize_rules
    assert type not in _dictionize_rules.keys(), f"Rule for {type.__name__} is " \
                                                f"already defined."
    _dictionize_rules[type] = rule
  return register_function

def _dictionize(node):
  """Apply the registered rules to a pytree node."""
  global _dictionize_rules
  # If there are no leaves, then a dictionize rule is not required
  if len(tree_util.tree_leaves(node)) == 0:
    return []
  for node_type, node_rule in _dictionize_rules.items():
    if isinstance(node, node_type):
      return node_rule(node)
  raise NotImplementedError(f"Node type {type(node).__name__} cannot be "
                            f"transformed to dict, please register a rule.")

def pytree_to_dict(tree: PyTree):
  """Construct a dictionary from a pytree.

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
    if isinstance(node, int):
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

Saving = Tuple[Callable[[], saving_state],
               Callable[[saving_state, scheduler.schedule, Any, Any, Any], Union[Any, NoReturn]],
               Callable[[Any], Union[Any, NoReturn]]]


class DataCollector(metaclass=abc.ABCMeta):
  """Collects sampled data and data loader states. """

  @abc.abstractmethod
  def register_data_loader(self, data_loader: data.DataLoader):
    """Register data loader to save the state. """

  @abc.abstractmethod
  def register_chain(self,
                     init_sample: PyTree,
                     init_checkpoint: PyTree,
                     static_information: PyTree
                     ) -> int:
    """Register a chain to save samples from.

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
    """Called every n'th step. """

  @abc.abstractmethod
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

  def register_chain(self,
                     init_sample: PyTree,
                     init_checkpoint: PyTree,
                     static_information: PyTree
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
      ujson.dump(samples_as_list, file) # pylint: disable=c-extension-no-member

  def finished(self, chain_id: int):
    """Simply return, nothing to wait for. """
    return None


class HDF5Collector(DataCollector):
  """Save to hdf5 format.

  Args:
    file: hdf5 file object

  """

  def __init__(self, file=h5py.File):
    assert h5py is not None, "h5py must be installed to use this DataCollector."
    self._finished = []
    self._sample_count = []
    self._leaf_names = []
    self._file = file

  def register_data_loader(self, data_loader: data.DataLoader):
    """Register data loader to save the state. """

  def register_chain(self,
                     init_sample: PyTree,
                     init_checkpoint: PyTree,
                     static_information: PyTree
                     ) -> int:
    """Register a chain to save samples from. """
    chain_id = len(self._finished)
    leaf_names = ["/".join(itertools.chain([f"/chain~{chain_id}"], key_tuple))
                  for key_tuple in pytree_dict_keys(init_sample)]
    print(leaf_names)
    leaves = tree_util.tree_leaves(init_sample)
    # Build the datasets
    # Todo: Maybe flatten completely
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
    """Save new leaves to dataset. """
    for leaf_name, value in zip(self._leaf_names[chain_id], values):
      self._file[leaf_name][self._sample_count[chain_id]] = value
    self._sample_count[chain_id] += 1

  def finalize(self, chain_id: int):
    """Wait for all writing to be finished. """
    # Is called after all host callback calls have been processed
    self._finished[chain_id].wait()

  def checkpoint(self, chain_id: int, state):
    """Called every n'th step. """

  def resume(self):
    """Called to restore data loader state and return sample states. """

  def finished(self, chain_id: int):
    """Return after everything has been written to the file. """
    self._finished[chain_id].wait()

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
    """Register data loader to save the state. """

  def register_chain(self,
                     init_sample,
                     init_checkpoint,
                     static_information) -> int:
    """Register a chain to save samples from. """
    chain_id = len(self._finished)
    leaves, treedef = tree_util.tree_flatten(init_sample)
    def leaf_shape(leaf):
      new_shape = onp.append(
        static_information.samples_collected,
        leaf.shape)
      new_shape = tuple(int(s) for s in new_shape)
      return new_shape
    sample_cache = [onp.zeros(leaf_shape(leaf), dtype=leaf.dtype) for leaf in leaves]

    self._finished.append(threading.Barrier(2))
    self._samples.append(sample_cache)
    self._treedefs.append(treedef)
    self._samples_count.append(0)
    self._leafnames.append(_groupnames(init_sample))
    return chain_id

  def save(self, chain_id: int, values):
    """Called with collected samples. """
    sample_cache = self._samples[chain_id]
    for leaf, value in zip(sample_cache, values):
      leaf[self._samples_count[chain_id]] = value
    self._samples_count[chain_id] += 1

  def finalize(self, chain_id: int):
    """Called after solver finished. """
    # Is called after all host callback calls have been processed
    self._finished[chain_id].wait()
    output_dir = Path(self._dir)
    output_file = output_dir / f"chain_{chain_id}.npz"
    output_dir.mkdir(exist_ok=True)
    onp.savez(
      output_file,
      **dict(zip(self._leafnames[chain_id], self._samples[chain_id])))

  def finished(self, chain_id):
    self._finished[chain_id].wait()
    # Restore original tree shape
    return tree_util.tree_unflatten(
      self._treedefs[chain_id],
      self._samples[chain_id])

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
    # Only return the leaves, as tree structure is redundant an requires
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
      data=None
    )
    return initial_state

  def save(state: saving_state,
           schedule: scheduler.schedule,
           sample: Any,
           scheduler_state: scheduler.scheduler_state = None,
           solver_state: Any = None):
    """Calls the data collector on the host via host callback module."""

    # Kepp the sample if it is not subject to burn in or thinning.
    keep = jnp.logical_and(schedule.burn_in, schedule.accept)

    # Save sample if samples is not subject to burn in or discarded by thinning
    saved = lax.cond(keep,
                     _save_wrapper,
                     lambda *args: 0,
                     (state.chain_id, sample))

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
    sample = pytree_to_dict(sample)
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

# Todo: Define such specific group-names or reduce to flat list?
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