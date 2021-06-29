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
from jax import test_util

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

  @pytest.mark.parametrize("cs", (3, 13, 29))
  def test_batch_size(self, dataloader, cs):
    pipeline, _ = dataloader

    def check_fn(leaf):
      assert leaf.shape[0] == cs
      assert leaf.shape[1] == batch_info.batch_size

    _, batch_info = pipeline.batch_format(cs)
    chain_id = pipeline.register_random_pipeline(cs)
    batch = pipeline.random_batches(chain_id)

    print(batch)

    jax.tree_map(check_fn, batch)

  @pytest.mark.parametrize("excluded", [[], ["a"], ["a", "b"]])
  def test_exclude_keys(self, dataset, excluded):
    mb_size = 2
    cs = 3

    pipeline = data.TensorflowDataLoader(dataset, mb_size, 100,
                                        exclude_keys=excluded)
    batch_format, _ = pipeline.batch_format(10)
    chain_id = pipeline.register_random_pipeline(cs)
    first_batch = pipeline.random_batches(chain_id)

    for key in first_batch.keys():
      assert key not in excluded
    for key in batch_format.keys():
      assert key not in excluded

  @pytest.mark.parametrize("cs", [1, 5, 10, 15])
  def test_sizes(self, dataloader, cs):

    pipeline, mb_size = dataloader
    chain_id = pipeline.register_random_pipeline(cache_size=cs)
    first_batch = pipeline.random_batches(chain_id)

    def test_fn(elem):
      assert elem.shape[0] == cs
      assert elem.shape[1] == mb_size

    jax.tree_map(test_fn, first_batch)

class TestNumpyLoader:

  @pytest.fixture(scope='class')
  def dataset(self):
    """Construct a dataset for the numpy data loader."""

    n = 17
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

  def test_kwargs(self, dataloader):
    """Test that the initial state can be set with the seed as kwarg. """

    seed = 10
    pipeline, _ = dataloader

    _, _ = pipeline.batch_format(10)
    chain_a = pipeline.register_random_pipeline(10, seed=seed)
    chain_b = pipeline.register_random_pipeline(10, seed=seed)

    batch_a = pipeline.random_batches(chain_a)
    batch_b = pipeline.random_batches(chain_b)

    def check_fn(a, b):
      assert jnp.all(a == b)

    jax.tree_map(check_fn, batch_a, batch_b)

  @pytest.mark.parametrize("cs", (3, 13, 30))
  def test_batch_size(self, dataloader, cs):
    """Check that the returned batches have the right format and dtype. """
    pipeline, _ = dataloader

    def check_fn(leaf):
      assert leaf.shape[0] == cs
      assert leaf.shape[1] == batch_info.batch_size

    _, batch_info = pipeline.batch_format(cs)
    chain_id = pipeline.register_random_pipeline(cs)
    batch = pipeline.random_batches(chain_id)

    print(batch)

    jax.tree_map(check_fn, batch)

  @pytest.mark.parametrize("cs_small, cs_big", [(5,7), (7, 11)])
  def test_cache_size_order(self, dataloader, cs_small, cs_big):
    """Check that changing the cache size does not influence the order. """
    pipeline, _ = dataloader

    small_chain = pipeline.register_random_pipeline(
      cs_small, seed=0)
    big_chain = pipeline.register_random_pipeline(
      cs_big, seed=0)
    small_batch = pipeline.random_batches(small_chain)
    big_batch = pipeline.random_batches(big_chain)

    def check_fn(a, b):
      for idx in range(cs_small):
        assert jnp.all(a[idx, ::] == b[idx, ::])

    jax.tree_map(check_fn, small_batch, big_batch)

  @pytest.mark.parametrize("cs", [1, 2, 4])
  def test_relation(self, dataloader, cs):
    """Test if sample and observations are corresponding"""

    pipeline, _ = dataloader
    new_pipe = pipeline.register_random_pipeline(cs)

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

  def test_checkpointing(self, dataloader):
    """Test that the state can be saved and restored. """

    pipeline, _ = dataloader

    chain_1 = pipeline.register_random_pipeline(3)
    chain_2 = pipeline.register_random_pipeline(5)

    # Get one batch information
    _ = pipeline.batch_format(3)

    # Draw some samples
    for i in range(11):
      pipeline.random_batches(chain_1)
      pipeline.random_batches(chain_2)

    # Checkpoint
    checkpoint = pipeline.save_state()

    # Draw some more date for checkpoint
    no_break_data = [pipeline.random_batches(chain_1),
                     pipeline.random_batches(chain_1),
                     pipeline.random_batches(chain_2),
                     pipeline.random_batches(chain_2)]

    _ = pipeline.batch_format(chain_1)
    _ = pipeline.batch_format(chain_2)

    pipeline.load_state(checkpoint)

    after_resume_data = [pipeline.random_batches(chain_1),
                         pipeline.random_batches(chain_1),
                         pipeline.random_batches(chain_2),
                         pipeline.random_batches(chain_2)]

    test_util.check_eq(no_break_data, after_resume_data)

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

    ml.batch_format.return_value = data.tree_dtype_struct(init_value), None
    ml.register_random_pipeline.return_value = 0
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
      state, batch = batch_fn(state)
      for key in compare_batch.keys():
        assert jnp.all(batch[key] == compare_batch[key])

    # Check that random batches are not drawn more often than necessary

    assert DL.random_batches.call_count == int(4.1)

  # Todo: Improve this test
  def test_vmap(self, data_loader_mock):
    DL, format, _ = data_loader_mock

    states_count = 5

    init_fn, batch_fn = data.random_reference_data(DL, format.cs)

    def test_function_data_loader(state):
      state, (batch, _) = batch_fn(state, information=True)
      return jax.tree_map(jnp.sum, batch)

    init_states = [
      init_fn(key=jax.random.PRNGKey(idx))
      for idx in range(states_count)
    ]

    # Init states vorspulen
    for idx in range(states_count):
      for _ in range(idx):
        init_states[idx], _ = batch_fn(init_states[idx], information=True)
    init_states = util.pytree_list_to_leaves(init_states)

    helper_fn = jax.jit(data.vmap_helper(partial(batch_fn, information=True)))
    _, (batches, _) = helper_fn(init_states)

    vmap_helper_result = jax.vmap(partial(jax.tree_map, jnp.sum))(batches)
    vmap_result = jax.vmap(test_function_data_loader)(init_states)

    def check_fn(a, b):
      assert jnp.all(a == b)

    jax.tree_map(check_fn, vmap_helper_result, vmap_result)
