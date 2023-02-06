"""Test io module. """
from functools import partial

from jax import random, jit, pmap, vmap, lax, tree_map
import jax.numpy as jnp

import pytest

from jax_sgmc import io, scheduler, util
from jax_sgmc.util import testing

class TestDictTransformation:
  pass







@pytest.mark.hdf5
class TestHDF5Collector:
  pass






class TestMemoryCollector:
  pass








class TestSaving:
  """Test saving."""

  @pytest.fixture
  def test_function(self):

    shape = (4, 3)

    def _update(state, _):
      key, split1, split2 = random.split(state["key"], 3)
      noise = random.normal(split1, shape=shape)
      new_state = {"key": key,
                   "results": {"noise": noise,
                               "sum": state["results"]["sum"] + noise}}
      return new_state, new_state

    def _init(seed):
      init_state = {"key": random.PRNGKey(seed),
                    "results": {"noise": jnp.zeros(shape),
                                "sum": jnp.zeros(shape)}}
      return init_state

    def _run(init_state, keep):
      _, all_results = lax.scan(_update, init_state, keep)
      accepted_results = tree_map(lambda leaf: leaf[keep], all_results)
      return accepted_results

    return _init, _update, _run

  def test_no_save(self, test_function):
    """Test saving by running save against no_save and direct output on random
    operations.
    """

    # Calculate the true results
    _init, _update, _run = test_function
    accepted_list = random.bernoulli(random.PRNGKey(11), shape=(15,))
    count = jnp.sum(accepted_list)

    init_state = _init(0)
    reference_solution = _run(init_state, accepted_list)

    # Run the no_save solution

    init_save, save, postprocess_save = io.no_save()

    def no_save_run(init_sample):
      def update(state, keep):
        saving_state, simulation_state = state
        simulation_state, sample = _update(simulation_state, None)
        saving_state, _ = save(saving_state, keep, simulation_state)
        return (saving_state, simulation_state), None
      saving_state = init_save(init_sample, {}, io.scheduler.static_information(samples_collected=count))
      (saving_state, _), _ = lax.scan(update, (saving_state, init_sample), accepted_list)
      return postprocess_save(saving_state, None)["samples"]

    no_save_results = no_save_run(init_state)

    # Check close

    testing.assert_equal(no_save_results, reference_solution)

  def test_save(self, test_function):
    """Test saving by running save against no_save and direct output on random
    operations.
    """

    # Calculate the true results
    _init, _update, _run = test_function
    accepted_list = random.bernoulli(random.PRNGKey(11), shape=(15,))
    count = jnp.sum(accepted_list)

    init_state = _init(0)
    reference_solution = _run(init_state, accepted_list)

    # Run the no_save solution

    data_collector = io.MemoryCollector()
    init_save, save, postprocess_save = io.save(data_collector)

    def save_run(init_sample):
      def update(state, keep):
        saving_state, simulation_state = state
        simulation_state, sample = _update(simulation_state, None)
        saving_state, _ = save(saving_state, keep, simulation_state)
        return (saving_state, simulation_state), None
      saving_state = init_save(init_sample, {}, io.scheduler.static_information(samples_collected=count))
      (saving_state, _), _ = lax.scan(update, (saving_state, init_sample), accepted_list)
      return postprocess_save(saving_state, None)["samples"]

    save_results = save_run(init_state)

    # Check close

    testing.assert_equal(save_results, reference_solution)

  @pytest.mark.skip
  def test_save_vmap(self, test_function):
    """Test saving by running save against no_save and direct output on random
    operations.
    """

    # Calculate the true results
    _init, _update, _run = test_function
    accepted_list = random.bernoulli(random.PRNGKey(11), shape=(15,))
    count = jnp.sum(accepted_list)
    seeds = jnp.arange(2)


    init_states = list(map(_init, seeds))
    reference_solution = list(map(lambda state: _run(state, accepted_list), init_states))

    # Run the no_save solution

    data_collector = io.MemoryCollector()
    init_save, save, postprocess_save = io.save(data_collector)

    vmap_init_states = [((init_save(init_sample, {}, io.scheduler.static_information(samples_collected=count)),
                         init_sample), accepted_list) for init_sample in init_states]

    # We use list_vmap, because we tun the chain over a list of pytrees. Also,
    # we currently have to vectorize over the acceptance list because of the
    # implementation of stop_vmap
    @util.list_vmap
    def save_run(init_state):
      init_args, accept = init_state
      saving_state, init_sample = init_args
      def update(state, keep):
        saving_state, simulation_state = state
        simulation_state, sample = _update(simulation_state, None)
        saving_state, _ = save(saving_state, keep, simulation_state)
        return (saving_state, simulation_state), None

      (saving_state, _), _ = lax.scan(update, (saving_state, init_sample), accept)
      return saving_state

    save_results = [postprocess_save(res, None)["samples"] for res in save_run(*vmap_init_states)]

    # Check close

    testing.assert_equal(save_results, reference_solution)
