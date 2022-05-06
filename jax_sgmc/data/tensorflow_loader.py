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

"""Load Tensorflow-Datasets in jit-compiled functions.

The tensorflow dataloader supports tensorflow Datasets, e.g. from the
`tensorflow_datasets` package.

Note:
  This submodule requires that ``tensorflow`` and ``tensorflow_datasets`` are
  installed. Additional information can be found in the
  :ref:`installation instructions<additional_requirements>`.

"""

from typing import List, Any

import jax
import jax.numpy as jnp
from jax import tree_util

from tensorflow import data as tfd
import tensorflow_datasets as tfds

from jax_sgmc.data.core import HostDataLoader

PyTree = Any
TFDataSet = tfd.Dataset

class TensorflowDataLoader(HostDataLoader):
  """Load data from a tensorflow dataset object.

  The tensorflow datasets package provides a high number of ready to go
  datasets, which can be provided directly to the Tensorflow Data Loader.

  ::

    import tensorflow_datasets as tdf
    import tensorflow_datasets as tfds
    from jax_sgmc import data
    from jax_sgmc.data.tensorflow_loader import TensorflowDataLoader

    pipeline = tfds.load("cifar10", split="train")
    data_loader = TensorflowDataLoader(pipeline, shuffle_cache=100, exclude_keys=['id'])

  Args:
    pipeline: A tensorflow data pipeline, which can be obtained from the
      tensorflow dataset package

  """

  def __init__(self,
               pipeline: TFDataSet,
               mini_batch_size: int = None,
               shuffle_cache: int = 100,
               exclude_keys: List = None):
    super().__init__()
    # Tensorflow is in general not required to use the library
    assert mini_batch_size is None, "Depreceated"
    assert TFDataSet is not None, "Tensorflow must be installed to use this " \
                                  "feature."
    assert tfds is not None, "Tensorflow datasets must be installed to use " \
                             "this feature."

    self._observation_count = jnp.int32(pipeline.cardinality().numpy())
    # Basic pipeline, from which all other pipelines are constructed
    self._pipeline = pipeline
    self._exclude_keys = [] if exclude_keys is None else exclude_keys
    self._shuffle_cache = shuffle_cache

    self._pipelines: List[TFDataSet] = []

  def register_random_pipeline(self,
                               cache_size: int = 1,
                               mb_size: int = None,
                               **kwargs) -> int:
    """Register a new chain which draws samples randomly.

    Args:
      cache_size: The number of drawn batches.
      mb_size: The number of observations per batch.

    Returns:
      Returns the id of the new chain.

    """

    # Assert that not kwargs are passed with the intention to control the
    # initial state of the tensorflow data loader
    assert kwargs == {}, "Tensorflow data loader does not accept additional "\
                         "kwargs"
    new_chain_id = len(self._pipelines)

    # Randomly draw a number of cache_size mini_batches, where each mini_batch
    # contains self.mini_batch_size elements.
    random_data = self._pipeline.repeat()
    random_data = random_data.shuffle(self._shuffle_cache)
    random_data = random_data.batch(mb_size)
    random_data = random_data.batch(cache_size)

    # The data must be transformed to numpy arrays, as most numpy arrays can
    # be transformed to the duck-typed jax array form
    # random_data = tfds.as_numpy(random_data)
    random_data = iter(random_data)

    self._pipelines.append(random_data)

    return new_chain_id

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
    raise NotImplementedError

  def get_batches(self, chain_id: int) -> PyTree:
    """Draws a batch from a chain.

    Args:
      chain_id: ID of the chain, which holds the information about the form of
        the batch and the process of assembling.

    Returns:
      Returns a batch of batches as registered by :func:`register_random_pipeline` or
      :func:`register_ordered_pipeline` with `cache_size` batches holding
      `mb_size` observations.

    """

    # Not supported data types, such as strings, must be excluded before
    # transformation to jax types.
    numpy_batch = next(self._pipelines[chain_id])
    if self._exclude_keys is not None:
      for key in self._exclude_keys:
        del numpy_batch[key]

    return tree_util.tree_map(jnp.array, numpy_batch), None

  @property
  def _format(self):
    data_spec = self._pipeline.element_spec
    if self._exclude_keys is not None:
      not_excluded_elements = {id: elem for id, elem in data_spec.items()
                               if id not in self._exclude_keys}
    else:
      not_excluded_elements = data_spec

    def leaf_dtype_struct(leaf):
      shape = tuple(int(s) for s in leaf.shape if s is not None)
      dtype = leaf.dtype.as_numpy_dtype
      return jax.ShapeDtypeStruct(
        dtype=dtype,
        shape=shape)

    return tree_util.tree_map(leaf_dtype_struct, not_excluded_elements)

  @property
  def static_information(self):
    """Returns information about total samples count and batch size. """
    information = {
      "observation_count" : self._observation_count
    }
    return information
