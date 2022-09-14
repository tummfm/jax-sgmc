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

"""Use samples saved with :class:`jax_sgmc.io.HDF5Collector` as reference data. """

import itertools
from typing import Any

import h5py
import numpy as onp
import jax.numpy as jnp
import jax

from jax_sgmc.data.numpy_loader import NumpyDataLoader
from jax_sgmc.io import pytree_dict_keys

PyTree = Any

# Inherit from NumpyDataLoader because slicing of arrays is similar
class HDF5Loader(NumpyDataLoader):
  """Load reference data from HDF5-files.

  This data loader can load reference data stored in HDF5 files. This makes it
  possible to use the :mod:`jax_sgmc.data` module to evaluate samples saved via
  the :class:`jax_sgmc.io.HDF5Collector`.

  Args:
    file: Path to the HDF5 file containing the reference data
    subdir: Path to the subset of the data set which should be loaded
    sample: PyTree to specify the original shape of the sub-pytree before it
      has been saved by the :class:`jax_sgmc.io.HDF5Collector`

   """

  def __init__(self, file, subdir="/chain~0/variables/", sample=None):
    # The sample is necessary to return the observations in the correct format.
    super().__init__()

    if isinstance(file, h5py.File):
      self._dataset = file
    else:
      self._dataset = h5py.File(name=file, mode="r")
    self._reference_data = ["/".join(itertools.chain([subdir], key_tuple))
                            for key_tuple in pytree_dict_keys(sample)]
    self._pytree_structure = jax.tree_structure(sample)
    self._sample_format = jax.tree_map(
      lambda leaf: jax.ShapeDtypeStruct(shape=leaf.shape, dtype=leaf.dtype),
      sample)

    observations_counts = [len(self._dataset[leaf_name])
                           for leaf_name in self._reference_data]
    self._observation_count = observations_counts[0]

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
    select_unique_idx = [onp.unique(batch_idx, return_inverse=True)
                     for batch_idx in selections_idx]

    # Slice the data and transform into pytree
    selected_observations = []
    for leaf_name in self._reference_data:
      unique_selections = [jnp.array(self._dataset[leaf_name][batch_idx])
                           for batch_idx, select_unique in select_unique_idx]
      selected_observations.append([unique[restore_idx]
                                    for unique, (_, restore_idx)
                                    in zip(unique_selections, select_unique_idx)])

    selected_observations = [jnp.array(leaf) for leaf in selected_observations]
    selected_observations = jax.tree_unflatten(self._pytree_structure,
                                               selected_observations)

    return selected_observations, jnp.array(selections_mask, dtype=jnp.bool_)

  def save_state(self, chain_id: int) -> PyTree:
    raise NotImplementedError("Saving of the DataLoader state is not supported.")

  def load_state(self, chain_id: int, data) -> None:
    raise NotImplementedError("Loading of the DataLoader state is not supported.")

  @property
  def _format(self):
    """Returns shape and dtype of a single observation. """
    return self._sample_format

  @property
  def static_information(self):
    """Returns information about total samples count and batch size. """
    information = {
      "observation_count": self._observation_count
    }
    return information

  def close(self):
    self._dataset.close()
