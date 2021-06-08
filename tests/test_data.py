"""Test the data loaders. """
# Pylint complains about the fixtures, because they are static
# pylint: disable=R

from collections import namedtuple
from functools import partial

try:
  import tensorflow
  import tensorflow_datasets
except ImportError:
  print("Tensorflow not found")
  tensorflow = None
  tensorflow_datasets = None

import jax
import jax.numpy as jnp

import numpy as onp

import pytest

from jax_sgmc import data, util

@pytest.mark.tensorflow
class TestTFLoader:

  @pytest.fixture(scope='class')
  def dataset(self):
    """Construct tensorflow dataset"""
    def generate_dataset():
      n = 10
      for i in range(n):
        yield {"a": i, "b": [i + 0.1 * 1, i + 0.11]}
    ds = tensorflow.data.Dataset.from_generator(
      generate_dataset,
      output_types={'a': tensorflow.float32,
      'b': tensorflow.float32},
      output_shapes={'a': tuple(), 'b': (2,)})
    return ds

  @pytest.fixture(params=[1, 10, 100])
  def dataloader(self, request, dataset):
    pipeline = data.TensorflowDataLoader(dataset, request.param, 100)
    return pipeline, request.param

  def test_batch_size(self, dataloader):
    pipeline, mb_size = dataloader

    cs = 3

    batch_info, dtype = pipeline.batch_format(cs)
    new_pipe, batch = pipeline.register_random_pipeline(cs)

    print(dtype)
    print(batch)

    def check_fn(leaf):
      assert leaf.shape[0] == cs
      assert leaf.shape[1] == batch_info.batch_size

    jax.tree_map(check_fn, batch)

    cs = 30

    batch_info, dtype = pipeline.batch_format(cs)
    new_pipe, batch = pipeline.register_random_pipeline(cs)

    print(dtype)
    print(batch)

    def check_fn(leaf):
      assert leaf.shape[0] == cs
      assert leaf.shape[1] == batch_info.batch_size

    jax.tree_map(check_fn, batch)

  def test_batch_information_caching(self, dataloader):
    cs = 3

    pipeline, mb_size = dataloader

    batch_info, dtype = pipeline.batch_format(cs)
    new_pipe, _ = pipeline.register_random_pipeline(cs)

    # Check that format is correct
    assert batch_info.batch_size == mb_size

    # Check that no new pipe exists
    print(f"Pipe id {new_pipe}")
    assert new_pipe == 0

    second_pipe, _ = pipeline.register_random_pipeline(3)
    print(f"Second pipe id {second_pipe}")
    assert second_pipe != 0

    # Check that two pipelines exist
    assert len(pipeline._random_pipelines) == 2

    # Check that dtype is correct
    random_batch = pipeline.random_batches(new_pipe)
    def assert_fn(x, y):
      assert x.dtype == y.dtype
      assert x.shape == y.shape
    jax.tree_map(assert_fn, random_batch, dtype)

  @pytest.mark.parametrize("excluded", [[], ["a"], ["a", "b"]])
  def test_exclude_keys(self, dataset, excluded):
    mb_size = 2
    cs = 3

    pipeline= data.TensorflowDataLoader(dataset, mb_size, 100,
                                        exclude_keys=excluded)
    _, first_batch = pipeline.register_random_pipeline(cs)

    for key in first_batch.keys():
      assert key not in excluded

  @pytest.mark.parametrize("cs", [1, 5, 10, 15])
  def test_sizes(self, dataloader, cs):

    pipeline, mb_size = dataloader
    _, first_batch = pipeline.register_random_pipeline(cache_size=cs)

    def test_fn(elem):
      assert elem.shape[0] == cs
      assert elem.shape[1] == mb_size

    jax.tree_map(test_fn, first_batch)

class TestNumpyLoader:

  @pytest.fixture(scope='class')
  def dataset(self):
    """Construct a dataset for the numpy data loader."""

    n = 4
    def generate_a(n):
      for i in range(n):
        yield i
    def generate_b(n):
      for i in range(n):
        yield [i + 0.1 * 1, i + 0.11]
    dataset = {'a': onp.array(list(generate_a(n))),
               'b': onp.array(list(generate_b(n)))}
    print(dataset)
    return dataset

  @pytest.fixture(scope='function', params=[1, 3, 5])
  def dataloader(self, request, dataset):
    """Construct a numpy data loader."""
    pipeline = data.NumpyDataLoader(request.param, **dataset)
    return pipeline, request.param

  def test_batch_size(self, dataloader):
    pipeline, mb_size = dataloader

    cs = 3

    batch_info, dtype = pipeline.batch_format(cs)
    new_pipe, batch = pipeline.register_random_pipeline(cs)

    print(dtype)
    print(batch)

    def check_fn(leaf):
      assert leaf.shape[0] == cs
      assert leaf.shape[1] == batch_info.batch_size

    jax.tree_map(check_fn, batch)

    cs = 30

    batch_info, dtype = pipeline.batch_format(cs)
    new_pipe, batch = pipeline.register_random_pipeline(cs)

    print(dtype)
    print(batch)

    def check_fn(leaf):
      assert leaf.shape[0] == cs
      assert leaf.shape[1] == batch_info.batch_size

    jax.tree_map(check_fn, batch)

  def test_batch_information_caching(self, dataloader):
    cs = 3

    pipeline, mb_size = dataloader

    batch_info, dtype = pipeline.batch_format(cs)
    new_pipe, _ = pipeline.register_random_pipeline(cs)

    # Check that format is correct
    assert batch_info.batch_size == mb_size

    # Check that no new pipe exists
    print(f"Pipe id {new_pipe}")
    assert new_pipe == 0

    second_pipe, _ = pipeline.register_random_pipeline(3)
    print(f"Second pipe id {second_pipe}")
    assert second_pipe != 0

    # Check that two pipelines exist
    assert len(pipeline._PRNGKeys) == 2

    # Check that dtype is correct
    random_batch = pipeline.random_batches(new_pipe)
    def assert_fn(x, y):
      assert x.dtype == y.dtype
      assert x.shape == y.shape
    jax.tree_map(assert_fn, random_batch, dtype)

  @pytest.mark.parametrize("cs_small, cs_big", [(5,7), (7, 11)])
  def test_cache_size_order(self, dataloader, cs_small, cs_big):
    pipeline, _ = dataloader

    _, small_batch = pipeline.register_random_pipeline(
      cs_small, key=jax.random.PRNGKey(0))
    _, big_batch = pipeline.register_random_pipeline(
      cs_big, key=jax.random.PRNGKey(0))

    def check_fn(a, b):
      for idx in range(cs_small):
        assert jnp.all(a[idx, ::] == b[idx, ::])

    jax.tree_map(check_fn, small_batch, big_batch)

  @pytest.mark.parametrize("cs", [1, 2, 4])
  def test_relation(self, dataloader, cs):
    """Test if sample and observations are corresponding"""

    pipeline, _ = dataloader
    new_pipe, _ = pipeline.register_random_pipeline(cs)

    def check_fn(a, b):
      for i, mb in enumerate(a):
        for j, sample in enumerate(mb):
          a = sample
          b_target = onp.array([a + 0.1, a + 0.11])
          b_is = b[i, j, ::]

          print(f"a: {a}")
          print(f"b should be: {b_target}")
          print(f"b is: {b_is}")

          assert jnp.all(b_is == b_target)

    for _ in range(3):
      batch = pipeline.random_batches(new_pipe)
      jax.tree_map(check_fn, batch['a'], batch['b'])

@pytest.mark.tensorflow
class TestRandomAccess:

  data_format = namedtuple("data_format",
                           ["mb_size",
                            "cs"])

  @pytest.fixture(scope='function', params=[data_format(1, 1),
                                         data_format(1, 7),
                                         data_format(19, 1),
                                         data_format(19, 7)])
  def data_loader_mock(self, request, mocker):
    """Construct tensorflow dataset without shuffling."""
    def generate_dataset():
      n = 5
      for i in range(n):
        yield {"a": i, "b": [i + 0.1 * 1, i + 0.11], "c": [[1*i, 2*i], [3*i, 4*i]]}
    ds = tensorflow.data.Dataset.from_generator(
      generate_dataset,
      output_types={'a': tensorflow.float32,
                    'b': tensorflow.float32,
                    'c': tensorflow.float32},
      output_shapes={'a': tuple(), 'b': (2,), 'c': (2, 2)},)
    ds = ds.repeat().batch(request.param.mb_size)
    ds_cache = ds.batch(request.param.cs)

    init_value = jax.tree_map(jnp.array, next(iter(tensorflow_datasets.as_numpy(ds_cache))))
    ds_cache = iter(tensorflow_datasets.as_numpy(ds_cache))

    ml = mocker.Mock(data.DataLoader)

    ml.batch_format.return_value = None, data.tree_dtype_struct(init_value)
    ml.register_random_pipeline.return_value = (0, jax.tree_map(jnp.array,
                                                            next(ds_cache)))
    def get_batch(chain_id):
      del chain_id
      return jax.tree_map(jnp.array, next(ds_cache))

    ml.register_ordered_pipeline.return_value = 0
    ml.random_batches.side_effect = get_batch

    return ml, request.param, ds

  def test_cache_order(self, data_loader_mock):
    """Test that the cache does not change the order """

    DL, format, dataset = data_loader_mock
    iterations = int(format.cs * 4.1)

    init_fn, batch_fn = data.random_reference_data(DL, format.cs)
    batch_fn = jax.jit(batch_fn)
    ds = iter(tensorflow_datasets.as_numpy(dataset))

    # Check that the values are returned in the correct order
    state = init_fn()
    for _ in range(iterations):
      compare_batch = jax.tree_map(jnp.array, next(ds))
      state, (batch, _) = batch_fn(state)
      for key in compare_batch.keys():
        print(batch)
        print(compare_batch)
        assert jnp.all(batch[key] == compare_batch[key])

    # Check that random batches are not drawn more often than necessary

    assert DL.random_batches.call_count == int(4.1 - 1.0)

  # Todo: Improve this test
  def test_vmap(self, pmap_setup, data_loader_mock):
    DL, format, _ = data_loader_mock

    init_fn, batch_fn = data.random_reference_data(DL, format.cs)

    def test_function_data_loader(state):
      state, (batch, _) = batch_fn(state)
      return jax.tree_map(jnp.sum, batch)

    init_states = [
      init_fn(key=jax.random.PRNGKey(idx))
      for idx in range(pmap_setup.host_count)
    ]
    transform_fn, _ = util.pytree_list_transform(
      init_states
    )

    # Init states vorspulen
    for idx in range(pmap_setup.host_count):
      for i in range(idx):
        init_states[idx], _ = batch_fn(init_states[idx])
    init_states = transform_fn(init_states)

    helper_fn = jax.jit(data.vmap_helper(batch_fn))
    _, (batches, _) = helper_fn(init_states)

    vmap_helper_result = jax.vmap(partial(jax.tree_map, jnp.sum))(batches)
    vmap_result = jax.vmap(test_function_data_loader)(init_states)

    def check_fn(a, b):
      assert jnp.all(a == b)

    jax.tree_map(check_fn, vmap_helper_result, vmap_result)
