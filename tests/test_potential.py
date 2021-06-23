"""Test the evaluation of the potential."""

import itertools

import jax
import jax.numpy as jnp

from jax import random, jit, test_util

import pytest

# from jax_sgmc.data import mini_batch
from jax_sgmc.potential import minibatch_potential
from jax_sgmc.data import mini_batch_information

# Todo: Test the potential evaluation function on arbitrary pytrees.

class TestPotential():
  # Helper functions

  @pytest.fixture
  def potential(self):
    """Define likelihood with pytree sample and pytree reference data."""

    def likelihood(sample, reference_data):
      scale = sample["scale"]
      bases = sample["base"]
      powers = reference_data["power"]
      ref_scale = reference_data["scale"]
      return scale * ref_scale * jnp.sum(jnp.power(bases, powers))

    def prior(sample):
      return jnp.exp(-sample["scale"])

    return prior, likelihood

  @pytest.mark.parametrize("obs, dim", itertools.product([7, 11], [3, 5]))
  def test_stochastic_potential_zero(self, potential, obs, dim):
    _, likelihood = potential
    prior = lambda _: 0.0
    # Setup potential

    scan_pot = minibatch_potential(prior, likelihood, strategy="map")
    vmap_pot = minibatch_potential(prior, likelihood, strategy="vmap")
    pmap_pot = minibatch_potential(prior, likelihood, strategy="pmap")

    # Setup reference data
    key = random.PRNGKey(0)

    split1, split2 = random.split(key, 2)
    observations = {"scale": random.exponential(split1, shape=(obs,)),
                    "power": random.exponential(split2, shape=(obs, dim))}
    reference_data = observations, mini_batch_information(observation_count=obs,
                                                          batch_size=obs)
    sample = {"scale": 0.5, "base": jnp.zeros(dim)}

    zero_array = jnp.array(-0.0)
    scan_result, _ = scan_pot(sample, reference_data)
    vmap_result, _ = vmap_pot(sample, reference_data)
    pmap_result, _ = pmap_pot(sample, reference_data)

    test_util.check_close(scan_result, zero_array)
    test_util.check_close(vmap_result, zero_array)
    test_util.check_close(pmap_result, zero_array)

  @pytest.mark.parametrize("obs, dim", itertools.product([7, 11], [3, 5]))
  def test_stochastic_potential_jit(self, potential, obs, dim):
    _, likelihood = potential
    prior = lambda _: 0.0
    # Setup potential

    scan_pot = jit(minibatch_potential(prior, likelihood, strategy="map"))
    vmap_pot = jit(minibatch_potential(prior, likelihood, strategy="vmap"))
    pmap_pot = jit(minibatch_potential(prior, likelihood, strategy="pmap"))

    # Setup reference data
    key = random.PRNGKey(0)

    split1, split2 = random.split(key, 2)
    observations = {"scale": random.exponential(split1, shape=(obs,)),
                    "power": random.exponential(split2, shape=(obs, dim))}
    reference_data = observations, mini_batch_information(
      observation_count=obs,
      batch_size=obs)
    sample = {"scale": 0.5, "base": jnp.zeros(dim)}

    zero_array = jnp.array(-0.0)
    scan_result, _ = scan_pot(sample, reference_data)
    vmap_result, _ = vmap_pot(sample, reference_data)
    pmap_result, _ = pmap_pot(sample, reference_data)

    test_util.check_close(scan_result, zero_array)
    test_util.check_close(vmap_result, zero_array)
    test_util.check_close(pmap_result, zero_array)

  @pytest.mark.parametrize("obs, dim", itertools.product([7, 11], [3, 5]))
  def test_stochastic_potential_equal(self, potential, obs, dim):
    prior, likelihood = potential
    # Setup potential

    scan_pot = jit(minibatch_potential(prior, likelihood, strategy="map"))
    vmap_pot = jit(minibatch_potential(prior, likelihood, strategy="vmap"))
    pmap_pot = jit(minibatch_potential(prior, likelihood, strategy="pmap"))

    # Setup reference data
    key = random.PRNGKey(0)

    split1, split2, split3 = random.split(key, 3)
    observations = {"scale": random.exponential(split1, shape=(obs,)),
                    "power": random.exponential(split2, shape=(obs, dim))}
    reference_data = observations, mini_batch_information(
      observation_count=obs,
      batch_size=obs)
    sample = {"scale": 0.5, "base": random.uniform(split3, (dim, ))}

    scan_result, _ = scan_pot(sample, reference_data)
    vmap_result, _ = vmap_pot(sample, reference_data)
    pmap_result, _ = pmap_pot(sample, reference_data)

    test_util.check_close(scan_result, vmap_result)
    test_util.check_close(scan_result, pmap_result)

  @pytest.mark.parametrize("obs, dim", itertools.product([7, 11], [3, 5]))
  def test_stochastic_potential_gradient_equal(self, potential, obs, dim):
    prior, likelihood = potential
    # Setup potential

    scan_grad = jit(
      jax.grad(minibatch_potential(prior, likelihood, strategy="map"),
               has_aux=True,
               argnums=0))
    vmap_grad = jit(
      jax.grad(minibatch_potential(prior, likelihood, strategy="vmap"),
               has_aux=True,
               argnums=0))
    pmap_grad = jit(
      jax.grad(minibatch_potential(prior, likelihood, strategy="pmap"),
               has_aux=True,
               argnums=0))

    # Setup reference data
    key = random.PRNGKey(0)

    split1, split2, split3 = random.split(key, 3)
    observations = {"scale": random.exponential(split1, shape=(obs,)),
                    "power": random.exponential(split2, shape=(obs, dim))}
    reference_data = observations, mini_batch_information(
      observation_count=obs,
      batch_size=obs)
    sample = {"scale": 0.5, "base": random.uniform(split3, (dim,))}

    scan_result, _ = scan_grad(sample, reference_data)
    vmap_result, _ = vmap_grad(sample, reference_data)
    pmap_result, _ = pmap_grad(sample, reference_data)

    test_util.check_close(scan_result, vmap_result)
    test_util.check_close(scan_result, pmap_result)

  @pytest.mark.parametrize("obs, dim", itertools.product([7, 11], [3, 5]))
  def test_stochastic_potential_gradient_shape(self, potential, obs, dim):
    _, likelihood = potential
    prior = lambda _: 0.0
    # Setup potential

    scan_grad = jit(
      jax.grad(minibatch_potential(prior, likelihood, strategy="map"),
               has_aux=True,
               argnums=0))
    vmap_grad = jit(
      jax.grad(minibatch_potential(prior, likelihood, strategy="vmap"),
               has_aux=True,
               argnums=0))
    pmap_grad = jit(
      jax.grad(minibatch_potential(prior, likelihood, strategy="pmap"),
               has_aux=True,
               argnums=0))

    # Setup reference data
    key = random.PRNGKey(0)

    # Set scale to zero to get zero gradient
    split1, split2 = random.split(key, 2)
    observations = {"scale": jnp.zeros(obs),
                    "power": random.exponential(split1, shape=(obs, dim))}
    reference_data = observations, mini_batch_information(
      observation_count=obs,
      batch_size=obs)
    sample = {"scale": 0.5, "base": random.uniform(split2, (dim,))}

    zero_gradient = jax.tree_map(jnp.zeros_like, sample)
    scan_result, _ = scan_grad(sample, reference_data)
    vmap_result, _ = vmap_grad(sample, reference_data)
    pmap_result, _ = pmap_grad(sample, reference_data)

    print(scan_result)
    print(vmap_result)
    print(pmap_result)

    test_util.check_close(scan_result, zero_gradient)
    test_util.check_close(vmap_result, zero_gradient)
    test_util.check_close(pmap_result, zero_gradient)
