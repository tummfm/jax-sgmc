"""Test the data loaders. """
# Pylint complains about the fixtures, because they are static
# pylint: disable=R

from collections import namedtuple

from unittest import mock

try:
  import tensorflow
  import tensorflow_datasets
except ImportError:
  print("Tensorflow not found")

import jax
import jax.numpy as jnp

import numpy as onp

import pytest

from jax_sgmc import data



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
    return dataset

  @pytest.fixture(params=[1, 3, 5])
  def dataloader(self, request, dataset):
    """Construct a numpy data loader."""
    pipeline = data.NumpyDataLoader(request.param, **dataset)
    return pipeline, request.param

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

  @pytest.mark.parametrize("cs", [1, 2, 4])
  def test_relation(self, dataloader, cs):
    """Test if sample and observations are corresponding"""

    pipeline, mb_size = dataloader
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

  @pytest.fixture(scope='class', params=[data_format(1, 1),
                                         data_format(1, 7),
                                         data_format(19, 1),
                                         data_format(19, 7)])
  def data_loader_mock(self, request):
    """Construct tensorflow dataset without shuffling."""
    def generate_dataset():
      n = 5
      for i in range(n):
        yield {"a": i, "b": [i + 0.1 * 1, i + 0.11]}
    ds = tensorflow.data.Dataset.from_generator(
      generate_dataset,
      output_types={'a': tensorflow.float32,
      'b': tensorflow.float32},
      output_shapes={'a': tuple(), 'b': (2,)})
    ds = ds.repeat().batch(request.param.mb_size)
    ds_cache = ds.batch(request.param.cs)

    init_value = jax.tree_map(jnp.array, next(iter(tensorflow_datasets.as_numpy(ds_cache))))
    ds_cache = iter(tensorflow_datasets.as_numpy(ds_cache))

    ml = mock.Mock(data.DataLoader)

    ml.batch_format.return_value = None, data.tree_dtype_struct(init_value)
    ml.register_random_pipeline.return_value = (0, jax.tree_map(jnp.array,
                                                            next(ds_cache)))
    def get_batch(*args):
      return jax.tree_map(jnp.array, next(ds_cache))

    ml.register_ordered_pipeline.return_value = 0
    ml.random_batches.side_effect = get_batch

    return ml, request.param, ds

  def test_cache_order(self, data_loader_mock):
    """Test that the cache does not change the order """
    DL, format, dataset = data_loader_mock
    iterations = int(format.cs * 4.1)

    init_fn, batch_fn = data.random_reference_data(DL, format.cs)
    ds = iter(tensorflow_datasets.as_numpy(dataset))

    # Check that the values are returned in the correct order
    state = init_fn()
    for idx in range(iterations):
      compare_batch = jax.tree_map(jnp.array, next(ds))
      state, (batch, _) = batch_fn(state)
      for key in compare_batch.keys():
        print(batch)
        print(compare_batch)
        assert jnp.all(batch[key] == compare_batch[key])

    # Check that random batches are not drawn more often than necessary

    assert DL.random_batches.call_count == int(4.1 - 1.0)