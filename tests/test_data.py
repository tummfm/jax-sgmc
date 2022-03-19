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

import numpy as onp

import tensorflow as tf
import tensorflow_datasets as tfds

import jax.numpy as jnp
from jax import ShapeDtypeStruct
from jax import test_util

from jax_sgmc.data.core import random_reference_data, full_reference_data
from jax_sgmc.data.numpy_loader import NumpyDataLoader, DeviceNumpyDataLoader
from jax_sgmc.data.tensorflow_loader import TensorflowDataLoader

import pytest

# Todo: Maybe parametrize size of dataset or shape
@pytest.fixture
def dataset():
  obs_count = 13

  data = {
    "ordered_indices": jnp.arange(obs_count, dtype=jnp.int_),
    "all_ones": jnp.ones(obs_count, dtype=jnp.bool_),
    "shape_5": jnp.ones((obs_count, 5), dtype=jnp.float_),
    "shape_3_5": jnp.ones((obs_count, 3, 5), dtype=jnp.int_)
  }

  data_format = {key: ShapeDtypeStruct(dtype=value.dtype, shape=value.shape)
                 for key, value in data.items()}

  return data, data_format, obs_count


class TestNumpyDeviceDataLoader:

  @pytest.fixture
  def data_loader(self, dataset):
    data, _, _ = dataset
    return NumpyDataLoader(**data)

  def test_initializer_batch_format(self, data_loader, dataset):
    _, data_format, _ = dataset
    batch_size = 7

    init_batch = data_loader.initializer_batch(batch_size)

    for key in data_format.keys():
      required_shape = (batch_size,) + data_format[key].shape[1:]
      assert required_shape == init_batch[key].shape
      assert data_format[key].dtype == init_batch[key].dtype

  @pytest.mark.parametrize("shuffle, in_epochs",
                           [(False, False),
                            (True, False),
                            (True, True)])
  def test_get_batches_random_shape(self, data_loader, dataset, shuffle, in_epochs):
    _, data_format, _ = dataset
    batch_size = 7
    cache_size = 5

    chain_id = data_loader.register_random_pipeline(cache_size=cache_size,
                                                    mb_size=batch_size,
                                                    in_epochs=in_epochs,
                                                    shuffle=shuffle)

    batch, _ = data_loader.get_batches(chain_id)

    for key in data_format.keys():
      required_shape = (cache_size, batch_size) + data_format[key].shape[1:]
      assert required_shape == batch[key].shape
      assert data_format[key].dtype == batch[key].dtype


  def test_get_batches_ordered_shape(self, data_loader, dataset):
    _, data_format, _ = dataset
    batch_size = 7
    cache_size = 5

    chain_id = data_loader.register_ordered_pipeline(cache_size=cache_size,
                                                     mb_size=batch_size)

    batch, _ = data_loader.get_batches(chain_id)

    for key in data_format.keys():
      required_shape = (cache_size, batch_size) + data_format[key].shape[1:]
      assert required_shape == batch[key].shape
      assert data_format[key].dtype == batch[key].dtype

  def test_all_keys_passed(self, dataset):
    data, _, _ = dataset

    cache_size = 3
    batch_size = 2

    # Test that all keys are returned in a batch
    data_loader = NumpyDataLoader(**data)

    chain_a = data_loader.register_random_pipeline(cache_size=cache_size,
                                                   mb_size=batch_size)
    chain_b = data_loader.register_ordered_pipeline(cache_size=cache_size,
                                                    mb_size=batch_size)

    batch_a, _ = data_loader.get_batches(chain_a)
    batch_b, _ = data_loader.get_batches(chain_b)

    assert set(batch_a.keys()) == set(data.keys())
    assert set(batch_b.keys()) == set(data.keys())

  def test_shuffeling(self, data_loader, dataset):
    _, data_format, obs_count = dataset
    batch_size = 3

    ceil_samples = int(onp.ceil(obs_count / batch_size))
    n_duplicated = ceil_samples * batch_size - obs_count

    # Get all samples in one cache and all samples in single caches
    chain_a = data_loader.register_random_pipeline(cache_size=ceil_samples,
                                                   mb_size=batch_size,
                                                   shuffle=True)
    chain_b = data_loader.register_random_pipeline(cache_size=1,
                                                   mb_size=batch_size,
                                                   shuffle=True)

    samples_a = data_loader.get_batches(chain_a)[0]["ordered_indices"]
    samples_b = onp.array([data_loader.get_batches(chain_b)[0]["ordered_indices"] for _ in range(ceil_samples)])

    # There must be exactly two duplicates
    _, count_a = onp.unique(samples_a, return_counts=True)
    _, count_b = onp.unique(samples_b, return_counts=True)

    assert onp.sum(count_a == 2) == n_duplicated
    assert onp.sum(count_b == 2) == n_duplicated
    assert onp.sum(count_a == 1) == obs_count - n_duplicated
    assert onp.sum(count_b == 1) == obs_count - n_duplicated

  def test_shuffeling_in_epochs(self, data_loader, dataset):
    _, data_format, obs_count = dataset
    batch_size = 3

    ceil_samples = int(onp.ceil(obs_count / batch_size))

    # Get all samples in one cache and all samples in single caches
    chain_a = data_loader.register_random_pipeline(cache_size=ceil_samples,
                                                   mb_size=batch_size,
                                                   shuffle=True,
                                                   in_epochs=True)
    chain_b = data_loader.register_random_pipeline(cache_size=1,
                                                   mb_size=batch_size,
                                                   shuffle=True,
                                                   in_epochs=True)

    samples_a, mask_a = data_loader.get_batches(chain_a)
    samples_a = samples_a["ordered_indices"]
    returned_b = [data_loader.get_batches(chain_b)
                 for _ in range(ceil_samples)]
    samples_b = onp.array([batch["ordered_indices"] for batch, _ in returned_b])
    mask_b = onp.array([mask for _, mask in returned_b])

    # Every element must appear exactly once
    _, count_a = onp.unique(samples_a[mask_a], return_counts=True)
    _, count_b = onp.unique(samples_b[mask_b], return_counts=True)

    assert onp.sum(count_a == 1) == obs_count
    assert onp.sum(count_b == 1) == obs_count

  def test_full_data(self, data_loader, dataset):
    _, data_format, obs_count = dataset
    batch_size = 3

    ceil_samples = int(onp.ceil(obs_count / batch_size))

    # Get all samples in one cache and all samples in single caches
    chain_a = data_loader.register_ordered_pipeline(cache_size=ceil_samples,
                                                    mb_size=batch_size)
    chain_b = data_loader.register_ordered_pipeline(cache_size=1,
                                                    mb_size=batch_size)

    samples_a, mask_a = data_loader.get_batches(chain_a)
    samples_a = samples_a["ordered_indices"]
    returned_b = [data_loader.get_batches(chain_b)
                 for _ in range(ceil_samples)]
    samples_b = onp.array([batch["ordered_indices"] for batch, _ in returned_b])
    mask_b = onp.array([mask for _, mask in returned_b])

    # Every element must appear exactly once
    _, count_a = onp.unique(samples_a[mask_a], return_counts=True)
    _, count_b = onp.unique(samples_b[mask_b], return_counts=True)

    print(mask_a)
    print(samples_a)

    assert onp.sum(count_a == 1) == obs_count
    assert onp.sum(count_b == 1) == obs_count

  def test_seeding(self, data_loader, dataset):
    batch_size = 3

    # Check that chains with the same seed return the same batches
    chain_a1 = data_loader.register_random_pipeline(cache_size=1,
                                                   mb_size=1,
                                                   seed=1)
    chain_a2 = data_loader.register_random_pipeline(cache_size=1,
                                                    mb_size=1,
                                                    seed=1)
    chain_b = data_loader.register_random_pipeline(cache_size=1,
                                                   mb_size=1,
                                                   seed=2)

    batch_a1, _ = data_loader.get_batches(chain_a1)
    batch_a2, _ = data_loader.get_batches(chain_a2)
    batch_b, _ = data_loader.get_batches(chain_b)

    assert batch_a1["ordered_indices"] == batch_a2["ordered_indices"]
    assert batch_b["ordered_indices"] != batch_a2["ordered_indices"]

  @pytest.mark.parametrize("shuffle, in_epochs", [(False, False),
                                                  (True, False),
                                                  (True, True)])
  def test_checkpoint_random_data(self, data_loader, dataset, shuffle, in_epochs):
    _, _, obs_count = dataset
    # Check that the NumpyDataLoader can be restarted from a saved state

    cache_size = 3
    batch_size = 2

    chain_a = data_loader.register_random_pipeline(shuffle=shuffle,
                                                   in_epochs=in_epochs,
                                                   cache_size=cache_size,
                                                   mb_size=batch_size)
    chain_b = data_loader.register_random_pipeline(shuffle=shuffle,
                                                   in_epochs=in_epochs,
                                                   cache_size=cache_size,
                                                   mb_size=batch_size)

    # Run to save the state before a masked state is returned
    n_iter = int(obs_count / (cache_size * batch_size))
    for i in range(n_iter):
      data_loader.get_batches(chain_a)
      data_loader.get_batches(chain_b)

    # Checkpoint
    checkpoint_a = data_loader.save_state(chain_a)
    checkpoint_b = data_loader.save_state(chain_b)

    # Draw masked sample
    batches_a = data_loader.get_batches(chain_a)
    batches_b = data_loader.get_batches(chain_b)

    # Initialize new chains and update them to the same state
    new_chain_a = data_loader.register_random_pipeline(cache_size=cache_size,
                                                       mb_size=batch_size,
                                                       shuffle=shuffle,
                                                       in_epochs=in_epochs)
    new_chain_b = data_loader.register_random_pipeline(cache_size=cache_size,
                                                       mb_size=batch_size,
                                                       shuffle=shuffle,
                                                       in_epochs=in_epochs)

    data_loader.load_state(new_chain_a, checkpoint_a)
    data_loader.load_state(new_chain_b, checkpoint_b)

    new_batches_a = data_loader.get_batches(new_chain_a)
    new_batches_b = data_loader.get_batches(new_chain_b)

    # The old and the new chains should have returned the same samples
    test_util.check_eq(batches_a, new_batches_a)
    test_util.check_eq(batches_b, new_batches_b)

  def test_checkpoint_full_data(self, data_loader, dataset):
    _, _, obs_count = dataset
    # Check that the NumpyDataLoader can be restarted from a saved state

    cache_size = 3
    batch_size = 2

    chain_a = data_loader.register_ordered_pipeline(cache_size=cache_size,
                                                    mb_size=batch_size)
    chain_b = data_loader.register_ordered_pipeline(cache_size=cache_size,
                                                    mb_size=batch_size)

    # Run to save the state before a masked state is returned
    n_iter = int(obs_count / (cache_size * batch_size))
    for i in range(n_iter):
      data_loader.get_batches(chain_a)
      data_loader.get_batches(chain_b)

    # Checkpoint
    checkpoint_a = data_loader.save_state(chain_a)
    checkpoint_b = data_loader.save_state(chain_b)

    # Draw masked sample
    batches_a = data_loader.get_batches(chain_a)
    batches_b = data_loader.get_batches(chain_b)

    # Initialize new chains and update them to the same state
    new_chain_a = data_loader.register_ordered_pipeline(cache_size=cache_size,
                                                        mb_size=batch_size)
    new_chain_b = data_loader.register_ordered_pipeline(cache_size=cache_size,
                                                        mb_size=batch_size)

    data_loader.load_state(new_chain_a, checkpoint_a)
    data_loader.load_state(new_chain_b, checkpoint_b)

    new_batches_a = data_loader.get_batches(new_chain_a)
    new_batches_b = data_loader.get_batches(new_chain_b)

    # The old and the new chains should have returned the same samples
    test_util.check_eq(batches_a, new_batches_a)
    test_util.check_eq(batches_b, new_batches_b)

  @pytest.mark.parametrize("data",
                           [{"x": [0, 1, 2], "y": [[0, 1], [0, 2], [0, 3]]},
                            {"x": onp.arange(3), "y": onp.zeros((3, 3))},
                            {"x": jnp.arange(3), "y": jnp.zeros((3, 3))}],
                           ids=["python list", "numpy array", "device array"])
  def test_input_data_type(self, data):
    # The data returned must always be a jax device array
    data_loader = NumpyDataLoader(**data)

    chain_a = data_loader.register_random_pipeline(cache_size=1, mb_size=1)
    batch, _ = data_loader.get_batches(chain_a)

    assert type(batch["x"]) == type(jnp.array(1))
    assert type(batch["y"]) == type(jnp.array(1))

  def test_input_data_wrong_shape(self):
    # The first axis determines the total number of observations
    x = jnp.arange(5)
    y = jnp.zeros((3, 7))

    with pytest.raises(ValueError):
      NumpyDataLoader(x=x, y=y)


class TestNumpyHostDataLoader:

  @pytest.fixture
  def data_loader(self, dataset):
    data, _, _ = dataset
    return DeviceNumpyDataLoader(**data)

  def test_initializer_batch_format(self, data_loader, dataset):
    _, data_format, _ = dataset

    batch_size = 7

    init_batch = data_loader.initializer_batch(batch_size)

    for key in data_format.keys():
      required_shape = (batch_size,) + data_format[key].shape[1:]
      assert required_shape == init_batch[key].shape
      assert data_format[key].dtype == init_batch[key].dtype

@pytest.mark.tensorflow
class TestTensorflowDataLoader:

  @pytest.fixture
  def tf_dataset(self, dataset):
    data, data_format, obs_count = dataset

    def gen():
      for idx in range(obs_count):
        yield {key: onp.array(value[idx]) for key, value in data.items()}

    output_signature = {
      key: tf.TensorSpec(shape=format.shape[1:], dtype=format.dtype)
      for key, format in data_format.items()
    }

    # Export tensorflow Dataset here to be able to test exclude keys later on
    tfds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    return tfds

  def test_initializer_batch_format(self, tf_dataset, dataset):
    _, data_format, _ = dataset

    batch_size = 7

    data_loader = TensorflowDataLoader(tf_dataset)
    init_batch = data_loader.initializer_batch(mb_size=batch_size)

    for key in data_format.keys():
      required_shape = (batch_size,) + data_format[key].shape[1:]
      assert required_shape == init_batch[key].shape
      assert data_format[key].dtype == init_batch[key].dtype

  def test_get_batches_random_shape(self, tf_dataset, dataset):
    _, data_format, _ = dataset
    batch_size = 7
    cache_size = 5

    data_loader = TensorflowDataLoader(tf_dataset,
                                       shuffle_cache=10)

    chain_id = data_loader.register_random_pipeline(cache_size=cache_size,
                                                    mb_size=batch_size)

    batch, _ = data_loader.get_batches(chain_id)
    print(batch)

    for key in data_format.keys():
      required_shape = (cache_size, batch_size) + data_format[key].shape[1:]
      assert required_shape == batch[key].shape
      assert data_format[key].dtype == batch[key].dtype


  @pytest.mark.skip("Not Implemented")
  def test_get_batches_ordered_shape(self):
    assert False

  @pytest.mark.parametrize("num_exclude", [1, 2, 3])
  def test_exclude_keys(self, tf_dataset, dataset, num_exclude):
    _, data_format, _ = dataset

    all_keys = list(data_format.keys())
    excluded_keys = all_keys[0:num_exclude]
    contained_keys = all_keys[num_exclude:]

    data_loader = TensorflowDataLoader(tf_dataset,
                                       exclude_keys=excluded_keys)

    chain_id = data_loader.register_random_pipeline(cache_size=2,
                                                    mb_size=3)
    batch, _ = data_loader.get_batches(chain_id)

    for key in excluded_keys:
      assert key not in batch.keys()

    for key in contained_keys:
      assert key in batch.keys()

  @pytest.mark.parametrize("name, exclude_keys",
                           [("MNIST", []),
                            ("Cifar10", ["id"])])
  def test_tensorflow_datasets(self, name, exclude_keys):
    cache_size = 2
    batch_size = 3

    tf_dataset, info = tfds.load(name, split="train", with_info=True)
    data_loader = TensorflowDataLoader(tf_dataset,
                                       exclude_keys=exclude_keys)

    chain_id = data_loader.register_random_pipeline(cache_size=cache_size,
                                                    mb_size=batch_size)
    batch, _ = data_loader.get_batches(chain_id)
    init_batch = data_loader.initializer_batch()

    # Check that format and shape is correct
    for key in info.features:
      if key in exclude_keys:
        continue
      required_shape = (cache_size, batch_size) + info.features[key].shape
      required_dtype = init_batch[key].dtype
      assert required_shape == batch[key].shape
      assert required_dtype == batch[key].dtype

class TestHostCallback:
  pass


class TestRandomDataAccess:

  @pytest.fixture
  def data_loader(self, dataset):
    data, _, _ = dataset
    return NumpyDataLoader(**data)

  def test_cache_size_invalid_arguments(self, data_loader, dataset):
    _, _, obs_count = dataset
    # The cache size must be positive and should warn if the cache is bigger
    # than the total dataset.

    with pytest.raises(ValueError):
      random_reference_data(data_loader, cached_batches_count=0, mb_size=1)

    # This warning only occurs if a HostDataLoader is used
    with pytest.warns(Warning):
      random_reference_data(data_loader,
                            cached_batches_count=obs_count + 1,
                            mb_size=1)

  def test_mb_size_invalid_arguments(self, data_loader, dataset):
    _, _, obs_count = dataset
    # The batch size must be positive and smaller or equal than the total
    # observation size.

    with pytest.raises(ValueError):
      random_reference_data(data_loader, cached_batches_count=1, mb_size=0)

    with pytest.raises(ValueError):
      random_reference_data(data_loader,
                            cached_batches_count=1,
                            mb_size=obs_count + 1)


class TestFullDataAccess:
  pass
