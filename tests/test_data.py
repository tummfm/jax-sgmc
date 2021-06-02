"""Test the data loaders. """
# Pylint complains about the fixtures, because they are static
# pylint: disable=R

try:
  import tensorflow
except ImportError:
  print("Tensorflow not found")

import jax
import jax.numpy as jnp

import numpy as onp

import pytest

from jax_sgmc import data



@pytest.mark.tensorflow
class TestTFLoader:

  @pytest.fixture
  def dataset(self):
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

  def test_batch_information_caching(self, dataset):
    mb_size = 2
    cs = 3

    pipeline = data.TensorflowDataLoader(dataset, mb_size, 100)

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

  @pytest.mark.parametrize("cs, mb_size", [(1, 1), (10, 1), (1, 10), (10, 10)])
  def test_sizes(self, dataset, cs, mb_size):

    pipeline = data.TensorflowDataLoader(dataset, mb_size, 100)
    _, first_batch = pipeline.register_random_pipeline(cache_size=cs)

    def test_fn(elem):
      assert elem.shape[0] == cs
      assert elem.shape[1] == mb_size

    jax.tree_map(test_fn, first_batch)


class TestNumpyLoader:

  @pytest.fixture
  def dataset(self):
    n = 10
    def generate_a(n):
      for i in range(n):
        yield i
    def generate_b(n):
      for i in range(n):
        yield [i + 0.1 * 1, i + 0.11]
    dataset = {'a': onp.array(list(generate_a(n))),
               'b': onp.array(list(generate_b(n)))}
    return dataset

  def test_batch_information_caching(self, dataset):
    mb_size = 2
    cs = 3

    pipeline = data.NumpyDataLoader(mb_size, **dataset)

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

  @pytest.mark.parametrize("mb_size, cs", [(1, 3), (3, 1), (3, 2), (1, 1)])
  def test_relation(self, dataset, mb_size, cs):
    """Test if sample and observations are corresponding"""

    pipeline = data.NumpyDataLoader(mb_size, **dataset)
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

    for _ in range(100):
      batch = pipeline.random_batches(new_pipe)
      jax.tree_map(check_fn, batch['a'], batch['b'])
