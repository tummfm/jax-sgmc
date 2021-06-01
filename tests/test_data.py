"""Test the data loaders. """

import sys

sys.path.append('..')

from jax_sgmc import data
import tensorflow

import jax

import numpy as onp

import pytest


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
    new_pipe, first_batch = pipeline.register_random_pipeline(cs)

    # Check that format is correct
    assert batch_info.batch_size == mb_size

    # Check that no new pipe exists
    print(f"Pipe id {new_pipe}")
    assert new_pipe == 0

    second_pipe, second_batch = pipeline.register_random_pipeline(3)
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

  @pytest.mark.parametrize("exclude", [([],), (["a"],), (["a", "b"],)])
  def test_exclude_keys(self, dataset, exclude):
    mb_size = 2
    cs = 3
    excluded = ["a"]

    pipeline= data.TensorflowDataLoader(dataset, mb_size, 100,
                                        exclude_keys=excluded)
    new_pipe, first_batch = pipeline.register_random_pipeline(cs)

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

  def setup_numpy_loader(self, batch_size, **kwargs):
    data_loader = data.NumpyDataLoader(batch_size, **kwargs)
    chain_id = data_loader.register_random_pipeline()
    return lambda :data_loader.random_batches(chain_id)

  def test_batch_size(self):
    """Generate an array and matrix with different dimensions and test shape of
    the result."""

    batch_size = 10

    obs = [onp.ones((4,)),
           onp.ones((40,)),
           onp.ones((50, 60)),
           onp.ones((60, 70, 80))]

    par = [onp.ones((4,)),
           onp.ones((40,)),
           onp.ones((50,50)),
           onp.ones((60, 50, 50))]

    for ob, pa in zip(obs, par):

      random_batch_fn = self.setup_numpy_loader(batch_size, obs=ob, par=pa)
      batch = random_batch_fn()

      print(batch)

      test_obs_batch = batch["obs"][0]
      test_par_batch = batch["par"][0]

      # Count of samples is correct
      assert test_obs_batch.shape[0] == batch_size
      assert test_par_batch.shape[0] == batch_size

      # Shape of samples is correct
      assert test_obs_batch.shape[1:] == ob.shape[1:]
      assert test_par_batch.shape[1:] == pa.shape[1:]


  def test_relation(self):
    """Test if sample and observations are corresponding"""

    obs = onp.arange(10)
    par = onp.arange(10)

    random_batch_fn = self.setup_numpy_loader(batch_size=2, obs=obs, par=par)

    for i in range(100):
      batch = random_batch_fn()
      obs_batch, par_batch = batch["obs"], batch["par"]

      assert onp.sum(obs_batch) == onp.sum(par_batch)
