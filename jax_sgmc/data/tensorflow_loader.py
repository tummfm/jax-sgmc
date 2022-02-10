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

Random Access
_______________________

The tensorflow data loader is a great choice for many standard datasets
available on tensorflow_datasets.

.. doctest::

  >>> import tensorflow_datasets as tfds
  >>> from jax import tree_util
  >>> from jax_sgmc import data
  >>>
  >>> # Helper function to look at the data provided
  >>> def show_data(data):
  ...   for key, item in data.items():
  ...     print(f"{key} with shape {item.shape} and dtype {item.dtype}")

The pipeline returned by tfds load can be directly passet to the data loader.
However, not all tensorflow data types can be transformed to jax data types, for
eample the feature 'id', which is a string. Those keys can be simply excluded
by passing the keyword argument `exclude_keys`.

  >>> # The data pipeline can be used directly
  >>> pipeline, info = tfds.load("cifar10", split="train", with_info=True)
  >>> print(info.features)
  FeaturesDict({
      'id': Text(shape=(), dtype=tf.string),
      'image': Image(shape=(32, 32, 3), dtype=tf.uint8),
      'label': ClassLabel(shape=(), dtype=tf.int64, num_classes=10),
  })
  >>>
  >>> data_loader = data.TensorflowDataLoader(pipeline, shuffle_cache=10, exclude_keys=['id'])
  >>>
  >>> # If the model needs data for initialization, an all zero batch can be
  >>> # drawn with the correct shapes and dtypes
  >>> show_data(data_loader.initializer_batch(mb_size=1000))
  image with shape (1000, 32, 32, 3) and dtype uint8
  label with shape (1000,) and dtype int32

The host callback wrappers cache some data in the device memory to reduce the
number of calls to the host. The cache size equals the number of batches stored
on the device. A bigger cache size is more effective in computation time, but
has an increased device memory consumption.

  >>> data_init, data_batch = data.random_reference_data(data_loader, 100, 1000)
  >>>
  >>> init_state = data_init()
  >>> new_state, batch = data_batch(init_state)
  >>> show_data(batch)
  image with shape (1000, 32, 32, 3) and dtype uint8
  label with shape (1000,) and dtype int32



"""

from typing import List, Any

import jax.numpy as jnp
from jax import tree_util

import tensorflow_datasets as tfds

from jax_sgmc.data.core import HostDataLoader

PyTree = Any

class TensorflowDataLoader(HostDataLoader):
  """Load data from a tensorflow dataset object.

  The tensorflow datasets package provides a high number of ready to go
  datasets, which can be provided directly to the Tensorflow Data Loader.

  .. doctest::

    >>> import tensorflow_datasets as tdf
    >>> from jax_sgmc import data
    >>>
    >>> pipeline = tfds.load("cifar10", split="train")
    >>> data_loader = data.TensorflowDataLoader(pipeline, shuffle_cache=100, exclude_keys=['id'])

  Args:
    pipeline: A tensorflow data pipeline, which can be obtained from the
      tensorflow dataset package

  """

  def __init__(self,
               pipeline: tfds.TFDataSet,
               mini_batch_size: int = None,
               shuffle_cache: int = 100,
               exclude_keys: List = None):
    super().__init__()
    # Tensorflow is in general not required to use the library
    assert mini_batch_size is None, "Depreceated"
    assert tfds.TFDataSet is not None, "Tensorflow must be installed to use this " \
                                  "feature."
    assert tfds is not None, "Tensorflow datasets must be installed to use " \
                             "this feature."

    self._observation_count = jnp.int32(pipeline.cardinality().numpy())
    # Basic pipeline, from which all other pipelines are constructed
    self._pipeline = pipeline
    self._exclude_keys = [] if exclude_keys is None else exclude_keys
    self._shuffle_cache = shuffle_cache

    self._pipelines: List[tfds.TFDataSet] = []

  def register_random_pipeline(self,
                               cache_size: int = 1,
                               mb_size: int = None,
                               **kwargs) -> int:
    """Register a new chain which draw samples randomly.

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
    random_data = tfds.as_numpy(random_data)
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
      Returns a superbatch as registered by :func:`register_random_pipeline` or
      :func:`register_ordered_pipeline` with `cache_size` batches holding
      `mb_size` observations.

    """

    # Not supported data types, such as strings, must be excluded before
    # transformation to jax types.
    numpy_batch = next(self._pipelines[chain_id])
    if self._exclude_keys is not None:
      for key in self._exclude_keys:
        del numpy_batch[key]

    return tree_util.tree_map(jnp.array, numpy_batch)

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
