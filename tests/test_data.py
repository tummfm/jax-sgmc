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
import functools

import numpy as onp

try:
  import tensorflow as tf
  import tensorflow_datasets as tfds
except ModuleNotFoundError:
  tf = None
  tfds = None

import jax
import jax.numpy as jnp
from jax import ShapeDtypeStruct

from jax import random
from jax import lax

from jax_sgmc.data.core import random_reference_data, full_reference_data, full_data_mapper
from jax_sgmc.data.numpy_loader import NumpyDataLoader, DeviceNumpyDataLoader
from jax_sgmc.util import testing

try:
  from jax_sgmc.data.tensorflow_loader import TensorflowDataLoader
except ModuleNotFoundError:
  TensorflowDataLoader = None

from jax_sgmc.util import list_vmap, list_pmap

import pytest

# Todo: Maybe parametrize size of dataset or shape
@pytest.fixture(params=[13, 20])
def dataset(request):
  obs_count = request.param

  data = {
    "ordered_indices": jnp.arange(obs_count, dtype=jnp.int_),
    "all_ones": jnp.ones(obs_count, dtype=jnp.bool_),
    "shape_5": jnp.ones((obs_count, 5), dtype=jnp.float_),
    "shape_3_5": jnp.ones((obs_count, 3, 5), dtype=jnp.int_)
  }

  data_format = {key: ShapeDtypeStruct(dtype=value.dtype, shape=value.shape)
                 for key, value in data.items()}

  return data, data_format, obs_count


class TestNumpyDataLoader:

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
    samples_b = onp.array([data_loader.get_batches(chain_b)[0]["ordered_indices"]
                           for _ in range(ceil_samples)])

    # There must be exactly two duplicates
    _, count_a = onp.unique(samples_a, return_counts=True)
    _, count_b = onp.unique(samples_b, return_counts=True)

    assert count_a.size == obs_count
    assert count_b.size == obs_count
    assert onp.sum(count_a == 1) == obs_count - n_duplicated
    assert onp.sum(count_b == 1) == obs_count - n_duplicated
    assert onp.sum(count_a == 2) == n_duplicated
    assert onp.sum(count_b == 2) == n_duplicated

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

    assert count_a.size == obs_count
    assert count_b.size == obs_count
    assert onp.sum(count_a == 1) == obs_count
    assert onp.sum(count_b == 1) == obs_count

  def test_full_data_order(self, data_loader, dataset):
    # The order of the data should not change
    _, data_format, obs_count = dataset
    batch_size = 3

    ceil_samples = int(onp.ceil(obs_count / batch_size))

    # Get all samples in one cache and all samples in single caches
    chain_a = data_loader.register_ordered_pipeline(cache_size=ceil_samples,
                                                    mb_size=batch_size)

    # Run for at least two complete iterations over the dataset
    for _ in range(3):
      cache, masks = data_loader.get_batches(chain_a)
      for it in range(batch_size):
        start_idx = it * batch_size
        expected_idx = start_idx + onp.arange(batch_size)

        indices = cache["ordered_indices"][it]
        mask = masks[it]

        # Check that expected indices correspond to actual indices for all valid
        # samples
        testing.assert_equal(indices[mask],
                           onp.squeeze(expected_idx[mask]))

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


  @pytest.mark.parametrize("shuffle, in_epochs",
                           [(False, False),
                            (True, False),
                            (True, True)])
  def test_cache_size_order(self, data_loader, shuffle, in_epochs):
    """Check that changing the cache size does not influence the order. """

    cs_small = 3
    cs_big = 5
    mb_size = 2

    small_chain = data_loader.register_random_pipeline(
      cs_small, mb_size, seed=0, shuffle=shuffle, in_epochs=in_epochs)
    big_chain = data_loader.register_random_pipeline(
      cs_big, mb_size, seed=0, shuffle=shuffle, in_epochs=in_epochs)
    small_batch, _ = data_loader.get_batches(small_chain)
    big_batch, _ = data_loader.get_batches(big_chain)

    for key in small_batch.keys():
      for idx in range(cs_small):
        testing.assert_equal(small_batch[key][idx], big_batch[key][idx])

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
    testing.assert_equal(batches_a, new_batches_a)
    testing.assert_equal(batches_b, new_batches_b)

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
    testing.assert_equal(batches_a, new_batches_a)
    testing.assert_equal(batches_b, new_batches_b)

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


class TestNumpyDeviceDataLoader:

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

  @pytest.mark.parametrize("data",
                           [{"x": [0, 1, 2], "y": [[0, 1], [0, 2], [0, 3]]},
                            {"x": onp.arange(3), "y": onp.zeros((3, 3))},
                            {"x": jnp.arange(3), "y": jnp.zeros((3, 3))}],
                           ids=["python list", "numpy array", "device array"])
  def test_input_data_type(self, data):
    # The data returned must always be a jax device array
    data_loader = DeviceNumpyDataLoader(**data)

    chain_a = data_loader.init_random_data()
    _, (batch, _) = data_loader.get_random_data(chain_a, batch_size=3)

    assert type(batch["x"]) == type(jnp.array(1))
    assert type(batch["y"]) == type(jnp.array(1))

  def test_input_data_wrong_shape(self):
    # The first axis determines the total number of observations
    x = jnp.arange(5)
    y = jnp.zeros((3, 7))

    with pytest.raises(ValueError):
      NumpyDataLoader(x=x, y=y)

  def test_seeding(self, data_loader):
    batch_size = 3

    # Check that chains with the same seed return the same batches
    chain_a1 = data_loader.init_random_data(key=random.PRNGKey(1))
    chain_a2 = data_loader.init_random_data(key=random.PRNGKey(1))
    chain_b = data_loader.init_random_data(key=random.PRNGKey(2))

    _, (batch_a1, _) = data_loader.get_random_data(chain_a1, batch_size)
    _, (batch_a2, _) = data_loader.get_random_data(chain_a2, batch_size)
    _, (batch_b, _) = data_loader.get_random_data(chain_b, batch_size)

    assert jnp.all(batch_a1["ordered_indices"] == batch_a2["ordered_indices"])
    assert jnp.any(batch_b["ordered_indices"] != batch_a2["ordered_indices"])

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


class TestRandomDataAccess:

  @pytest.fixture
  def data_loader(self, dataset):
    data, _, _ = dataset
    return NumpyDataLoader(**data)

  @pytest.fixture
  def example_problem(self):
    # This problem should check that each sample

    def init_test_fn(obs_count, batch_size, data_fn):
      init_data_fn, batch_fn, _ = data_fn
      num_its = int(onp.ceil(obs_count / batch_size))

      def init_fn():
        # An array to count the number of observations returned
        sample_counts = jnp.zeros(obs_count)
        data_state = init_data_fn(shuffle=True, in_epochs=True)
        return data_state, sample_counts, 0.0

      def _test_update_fn(state, _):
        data_state, sample_counts, valid_obs_count = state
        data_state, (batch, mb_info) = batch_fn(
          data_state,
          information=True
        )
        sample_counts = sample_counts.at[batch["ordered_indices"]].add(mb_info.mask)
        valid_obs_count += jnp.sum(mb_info.mask)
        return (data_state, sample_counts, valid_obs_count), None

      def test_fn(init_state):
        return lax.scan(_test_update_fn, init_state, onp.arange(num_its))

      return init_fn, test_fn

    return init_test_fn

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


  def test_random_data_acces_with_in_epoch_shuffeling(self, data_loader, example_problem, dataset):
    _, _, obs_count = dataset

    cache_size = 2
    batch_size = 3

    data_fn = random_reference_data(data_loader, cache_size, batch_size)

    init_fn, test_fn = example_problem(obs_count, batch_size, data_fn)

    (_, test_sample_count, test_obs_size), _ = test_fn(init_fn())

    assert onp.all(test_sample_count == 1)
    assert test_obs_size == obs_count

  def test_jit_random_data(self, data_loader, example_problem, dataset):
    _, _, obs_count = dataset

    cache_size = 2
    batch_size = 3

    data_fn = random_reference_data(data_loader, cache_size, batch_size)

    init_fn, test_fn = example_problem(obs_count, batch_size, data_fn)

    test_fn_jit = jax.jit(test_fn)
    test_fn_jit(init_fn())

    (_, test_sample_count, test_obs_size), _ = test_fn_jit(init_fn())

    assert onp.all(test_sample_count == 1)
    assert test_obs_size == obs_count

  @pytest.mark.skip("Custom Batching not implemented")
  def test_vmap_random_data(self, data_loader, example_problem, dataset):
    _, _, obs_count = dataset

    vmap_size = 2

    cache_size = 2
    batch_size = 3

    data_fn = random_reference_data(data_loader, cache_size, batch_size)

    init_fn, test_fn = example_problem(obs_count, batch_size, data_fn)

    test_fn_vmap = list_vmap(test_fn)
    init_states = [init_fn() for _ in range(vmap_size)]

    results = test_fn_vmap(*init_states)

    for res in results:
      (_, test_sample_count, test_obs_size), _ = res
      assert onp.all(test_sample_count == 1)
      assert test_obs_size == obs_count

  @pytest.mark.pmap
  def test_pmap_random_data(self, data_loader, example_problem, dataset):
    _, _, obs_count = dataset

    pmap_size = jax.device_count()

    cache_size = 2
    batch_size = 3

    data_fn = random_reference_data(data_loader, cache_size, batch_size)

    init_fn, test_fn = example_problem(obs_count, batch_size, data_fn)

    test_fn_pmap = list_pmap(test_fn)
    init_states = [init_fn() for _ in range(pmap_size)]

    results = test_fn_pmap(*init_states)

    for res in results:
      (_, test_sample_count, test_obs_size), _ = res
      assert onp.all(test_sample_count == 1)
      assert test_obs_size == obs_count


class TestFullDataMapper:

  @pytest.fixture
  def data_loader(self, dataset):
    data, _, _ = dataset
    return NumpyDataLoader(**data)

  @pytest.fixture
  def example_problem_mask(self):
    def test_update_fn(batch, mask, _):
        return (jnp.sum(mask), batch["ordered_indices"], mask), None
    return test_update_fn

  @pytest.fixture
  def example_problem_no_mask(self):
    def test_update_fn(batch, _):
      return batch["ordered_indices"], None
    return test_update_fn

  @pytest.fixture
  def example_problem_pmap(self):
    @jax.pmap
    def identity_fn(x):
      return x

    def test_update_fn(batch, _):
      original_shape = jax.tree_map(jnp.shape, batch)
      pmap_batch = jax.tree_map(
        functools.partial(jnp.reshape, newshape=(jax.device_count(), -1)),
        batch)
      pmap_identity = identity_fn(pmap_batch)
      identity = jax.tree_map(jnp.reshape, pmap_identity, original_shape)
      return identity, None

    return test_update_fn

  def test_full_data_map_mask(self, dataset, data_loader, example_problem_mask):
    _, _, obs_count = dataset

    mapper, release = full_data_mapper(cached_batches_count=3,
                                       mb_size=2,
                                       data_loader=data_loader)

    results, _ = mapper(example_problem_mask, None, masking=True)
    sum_mask, samples, masks = results

    samples = onp.ravel(samples)
    masks = onp.ravel(masks)
    sum_mask = onp.sum(sum_mask)

    # Every element must appear exactly once
    _, count = onp.unique(samples[masks], return_counts=True)

    assert onp.sum(count == 1) == obs_count
    assert sum_mask == obs_count

    release()

  def test_jit_full_data_map_mask(self, dataset, data_loader,
                              example_problem_mask):
    _, _, obs_count = dataset

    mapper, release = full_data_mapper(cached_batches_count=3,
                                       mb_size=2,
                                       data_loader=data_loader)

    jit_mapped = jax.jit(lambda: mapper(example_problem_mask, None, masking=True))
    results, _ = jit_mapped()
    sum_mask, samples, masks = results

    samples = onp.ravel(samples)
    masks = onp.ravel(masks)
    sum_mask = onp.sum(sum_mask)

    # Every element must appear exactly once
    _, count = onp.unique(samples[masks], return_counts=True)

    assert onp.sum(count == 1) == obs_count
    assert sum_mask == obs_count

    release()

  def test_full_data_map_no_mask(self, dataset, data_loader, example_problem_no_mask):
    _, _, obs_count = dataset

    mapper, release = full_data_mapper(cached_batches_count=3,
                                       mb_size=2,
                                       data_loader=data_loader)

    samples, _ = mapper(example_problem_no_mask, None, masking=False)

    # Every element must appear exactly once
    _, count = onp.unique(samples, return_counts=True)

    assert onp.sum(count == 1) == obs_count

    release()

  def test_jit_full_data_map_no_mask(self, dataset, data_loader, example_problem_no_mask):
    _, _, obs_count = dataset

    mapper, release = full_data_mapper(cached_batches_count=3,
                                       mb_size=2,
                                       data_loader=data_loader)

    jit_mapped = jax.jit(lambda: mapper(example_problem_no_mask, None, masking=False))
    samples, _ = jit_mapped()

    # Every element must appear exactly once
    _, count = onp.unique(samples, return_counts=True)

    assert onp.sum(count == 1) == obs_count

    release()

  @pytest.mark.pmap
  def test_pmap_full_data_map_no_mask(self, dataset, data_loader,
                                     example_problem_pmap):
    _, _, obs_count = dataset

    mapper, release = full_data_mapper(cached_batches_count=2,
                                       mb_size=jax.device_count(),
                                       data_loader=data_loader)

    map_fn = functools.partial(mapper, example_problem_pmap)

    assert jax.device_count() != 1

    # Mapping should work when jitted
    results, _ = jax.jit(map_fn)(None)

    # Every element must appear exactly once
    _, count = onp.unique(results["ordered_indices"], return_counts=True)

    assert onp.sum(count == 1) == obs_count

    release()

  @pytest.mark.parametrize("mb_size", [1, 2])
  def test_full_data_unbatched(self, dataset, data_loader, mb_size, example_problem_no_mask):
    _, _, obs_count = dataset

    mapper, release = full_data_mapper(cached_batches_count=3,
                                       mb_size=mb_size,
                                       data_loader=data_loader)

    samples, _ = mapper(example_problem_no_mask, None, masking=False, batched=False)

    # Every element must appear exactly once
    _, count = onp.unique(samples, return_counts=True)

    assert onp.sum(count == 1) == obs_count

    release()


class TestFullDataAccess:

  @pytest.fixture
  def data_loader(self, dataset):
    data, _, _ = dataset
    return NumpyDataLoader(**data)

  @pytest.fixture
  def example_problem_mask(self):

    def init_test_fn(data_fn):
      init_data_fn, map_fn, _ = data_fn

      def init_fn():
        # An array to count the number of observations returned
        return init_data_fn()

      def _test_update_fn(batch, mask, _):
        return (jnp.sum(mask), batch["ordered_indices"], mask), None


      def test_fn(data_state):
        return map_fn(_test_update_fn, data_state, None, masking=True)

      return init_fn, test_fn
    return init_test_fn

  @pytest.fixture
  def example_problem_no_mask(self):
    def init_test_fn(data_fn):
      init_data_fn, map_fn, _ = data_fn

      def init_fn():
        # An array to count the number of observations returned
        return init_data_fn()

      def _test_update_fn(batch, _):
        return batch["ordered_indices"], None


      def test_fn(data_state):
        return map_fn(_test_update_fn, data_state, None, masking=False)

      return init_fn, test_fn
    return init_test_fn

  def test_jit_full_data_map_mask(self, dataset, data_loader,
                              example_problem_mask):
    _, _, obs_count = dataset
    data_map = full_reference_data(data_loader,
                                   cached_batches_count=3,
                                   mb_size=2)

    init_fn, test_fn = example_problem_mask(data_map)
    test_fn_jit = jax.jit(test_fn)
    test_fn_jit(init_fn())

    _, (results, _) = test_fn_jit(init_fn())
    sum_mask, samples, masks = results

    samples = onp.ravel(samples)
    masks = onp.ravel(masks)
    sum_mask = onp.sum(sum_mask)

    # Every element must appear exactly once
    _, count = onp.unique(samples[masks], return_counts=True)

    assert onp.sum(count == 1) == obs_count
    assert sum_mask == obs_count

  @pytest.mark.parametrize("cs, mb",
                           [(2, 3),
                            (3, 2),
                            (13, 2),
                            (2, 13),
                            (13, 13)])
  def test_full_data_map_no_mask(self, dataset, data_loader, example_problem_no_mask, cs, mb):
    num_maps = 3 # Check multiple iterations over the full dataset

    _, _, obs_count = dataset
    data_map = full_reference_data(data_loader,
                                   cached_batches_count=cs,
                                   mb_size=mb)

    init_fn, test_fn = example_problem_no_mask(data_map)

    data_state = init_fn()

    for it in range(num_maps):
      data_state, (samples, _) = test_fn(data_state)

      # Every element must appear exactly once and the order of the samples
      # remains unchanged
      _, count = onp.unique(samples, return_counts=True)

      assert onp.sum(count == 1) == obs_count
      assert count.size == obs_count
      testing.assert_equal(samples, onp.arange(obs_count))

  def test_jit_full_data_map_no_mask(self, dataset, data_loader, example_problem_no_mask):
    _, _, obs_count = dataset
    data_map = full_reference_data(data_loader,
                                  cached_batches_count=3,
                                   mb_size=2)

    init_fn, test_fn = example_problem_no_mask(data_map)
    test_fn_jit = jax.jit(test_fn)
    test_fn_jit(init_fn())

    _, (samples, _) = test_fn_jit(init_fn())

    # Every element must appear exactly once
    _, count = onp.unique(samples, return_counts=True)

    assert onp.sum(count == 1) == obs_count

  @pytest.mark.skip("Custom Batching not implemented")
  def test_vmap_full_data_map_no_mask(self, dataset, data_loader, example_problem_no_mask):
    pass

  @pytest.mark.pmap
  def test_pmap_full_data_map_no_mask(self, dataset, data_loader, example_problem_no_mask):
    pass

  @pytest.mark.skip("Custom Batching not implemented")
  def test_vmap_full_data_map_no_mask(self, dataset, data_loader, example_problem_no_mask):
    pass

  @pytest.mark.pmap
  def test_pmap_full_data_map_no_mask(self, dataset, data_loader, example_problem_no_mask):
    pass