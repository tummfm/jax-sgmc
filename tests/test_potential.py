"""Test the evaluation of the potential."""

import jax
import jax.numpy as jnp

from jax import random, jit

# from jax_sgmc.data import mini_batch
from jax_sgmc.potential import minibatch_potential
from jax_sgmc.potential import stochastic_potential_gradient
from jax_sgmc.data import mini_batch_information

# Todo: Test the potential evaluation function on arbitrary pytrees.

class TestPotential():
  # Helper functions

  def linear_potential(self):
    def likelihood(sample, ref_data):
      parameters = ref_data["parameters"]
      observations = ref_data["observations"]
      return jnp.dot(sample, parameters) - jnp.sum(observations)

    def prior(sample):
      return jnp.float32(0.0)

    # def result(sample, parameters, observations):

    self.likelihood = likelihood
    self.prior = prior

  @staticmethod
  def sample_batch(sample_size, batch_size, obs_size):
    key = random.PRNGKey(0)

    split, key = random.split(key)
    sample = random.normal(key, shape=(sample_size,))

    split, key = random.split(key)
    parameters = random.normal(key, shape=(batch_size, sample_size))

    split, key = random.split(key)
    observations = random.normal(key, shape=(batch_size, obs_size))

    information = mini_batch_information(observation_count=batch_size,
                                         batch_size=batch_size)
    data = {"parameters": parameters,
            "observations": observations}

    return (data, information), sample

  # The real tests

  def test_map(self):

    self.linear_potential()

    potential_fun_map = minibatch_potential(self.prior,
                                            self.likelihood,
                                            strategy='map')
    potential_fun_vmap = minibatch_potential(self.prior,
                                             self.likelihood,
                                             strategy='vmap')
    stochastic_gradient_map = stochastic_potential_gradient(self.prior,
                                                            self.likelihood,
                                                            strategy='map')
    stochastic_gradient_vmap = stochastic_potential_gradient(self.prior,
                                                             self.likelihood,
                                                             strategy='vmap')

    potential_fun_map = jit(potential_fun_map)
    potential_fun_vmap = jit(potential_fun_vmap)
    stochastic_gradient_map = jit(stochastic_gradient_map)
    stochastic_gradient_vmap = jit(stochastic_gradient_vmap)

    example_batch, example_sample = self.sample_batch(10, 5, 12)

    result_map = potential_fun_map(example_sample, example_batch)
    result_vmap = potential_fun_vmap(example_sample, example_batch)
    result_map_gradient = stochastic_gradient_map(example_sample,
                                                  example_batch)
    result_vmap_gradient = stochastic_gradient_vmap(example_sample,
                                                    example_batch)

    # Check that the ouput is scalar
    assert result_map.shape == tuple()
    assert result_vmap.shape == tuple()

    print(result_map_gradient)
    print(result_vmap_gradient)

    # Check that the ouput is the same
    assert result_map == result_vmap
    assert jnp.all(jnp.abs(result_map_gradient - result_vmap_gradient) <
                   0.05 * 0.001 * (jnp.abs(result_map_gradient) +
                                   jnp.abs(result_vmap_gradient)))





