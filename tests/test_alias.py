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

"""Test the convergence of the solver on small toy problem. """

import jax
from jax import random
import jax.numpy as jnp
from jax.scipy.stats import norm

from scipy import stats as scpstats

import pytest

from jax_sgmc import data
from jax_sgmc import potential
from jax_sgmc import alias
from jax_sgmc.data.numpy_loader import NumpyDataLoader


@pytest.fixture
def kolmogorov_smirnov_setup():

  # Sample from a gaussian distribution
  samples = 100
  sigma = 0.5

  key = random.PRNGKey(11)
  x = sigma * random.normal(key, (samples, ))

  # No need for reference data
  data_loader = data.numpy_loader.DeviceNumpyDataLoader(x=jnp.zeros((2, 1)))
  batch_fn = data.random_reference_data(data_loader, 1, 1, verify_calls=True)

  def likelihood_fn(sample, _):
    return -0.5 * (sample / sigma) ** 2

  def prior_fn(sample):
    return jnp.asarray(0.0)

  potential_fn = potential.minibatch_potential(
    prior_fn, likelihood_fn, strategy="vmap")
  full_potential_fn = potential.full_potential(
    prior_fn, likelihood_fn, strategy="vmap"
  )

  def check_fn(sampled):
    statistic = scpstats.kstest(sampled, x)
    assert statistic.pvalue > 0.05, (
      f"Solver generated non-normal distributed samples with 95% confidence "
      f"(p_value is {statistic.pvalue})."
    )

  return data_loader, batch_fn, potential_fn, full_potential_fn, x[0], check_fn


@pytest.fixture
def problem():

  # Reference Data

  N = 4
  samples = 1000  # Total samples
  sigma = 0.5

  key = random.PRNGKey(0)
  split1, split2, split3 = random.split(key, 3)

  w = random.uniform(split3, minval=-1, maxval=1, shape=(N, 1))
  noise = sigma * random.normal(split2, shape=(samples, 1))
  x = random.uniform(split1, minval=-10, maxval=10, shape=(samples, N))
  x = jnp.stack([x[:, 0] + x[:, 1], x[:, 1], 0.1 * x[:, 2] - 0.5 * x[:, 3],
                 x[:, 3]]).transpose()
  y = jnp.matmul(x, w) + noise
  w_init = sample = {"w": jnp.zeros((N, 1)), "sigma": jnp.array(10.0)}

  M = 10
  cs = 1000

  data_loader = NumpyDataLoader(x=x, y=y)
  batch_fn = data.random_reference_data(data_loader,
                                        cached_batches_count=cs,
                                        mb_size=M)

  def model(sample, observations):
    weights = sample["w"]
    predictors = observations["x"]
    return jnp.dot(predictors, weights)

  def likelihood(sample, observations):
    sigma = sample["sigma"]
    y = observations["y"]
    y_pred = model(sample, observations)
    return norm.logpdf(y - y_pred, scale=sigma)

  def prior(unused_sample):
    return 0.0

  # If the model is more complex, the strategy can be set to map for sequential
  # evaluation and pmap for parallel evaluation.
  potential_fn = potential.minibatch_potential(prior=prior,
                                               likelihood=likelihood,
                                               strategy="vmap")
  full_potential_fn = potential.full_potential(prior=prior,
                                               likelihood=likelihood,
                                               strategy="vmap")
  return data_loader, batch_fn, potential_fn, full_potential_fn, w, w_init


class TestAliasKolmogorovSmirnov:

  @pytest.mark.solver
  def test_rms_prop(self, kolmogorov_smirnov_setup):
    data_loader, batch_fn, potential_fn, _, init_sample, assert_fn =\
      kolmogorov_smirnov_setup

    solver = alias.sgld(
      potential_fn,
      data_loader,
      cache_size=1,
      batch_size=1,
      first_step_size=0.5,
      last_step_size=0.1,
      burn_in=100,
      accepted_samples=100,
      rms_prop=True
    )

    sampled = solver(init_sample, iterations=500)[0]["samples"]["variables"]
    assert_fn(sampled)

  @pytest.mark.solver
  def test_re_sgld(self, kolmogorov_smirnov_setup):
    data_loader, batch_fn, potential_fn, _, init_sample, assert_fn =\
      kolmogorov_smirnov_setup

    solver = alias.re_sgld(
      potential_fn,
      data_loader,
      cache_size=1,
      batch_size=1,
      first_step_size=0.5,
      last_step_size=0.1,
      burn_in=100,
      accepted_samples=100,
      temperature=1.5
    )

    sampled = solver(
      (init_sample, init_sample), iterations=500
    )[0]["samples"]["variables"]
    assert_fn(sampled)

  @pytest.mark.solver
  def test_amagold(self, kolmogorov_smirnov_setup):
    data_loader, batch_fn, potential_fn, full_pot_fn, init_sample, assert_fn =\
      kolmogorov_smirnov_setup

    solver = alias.amagold(
      potential_fn,
      full_pot_fn,
      data_loader,
      cache_size=1,
      batch_size=1,
      first_step_size=0.5,
      last_step_size=0.1,
      burn_in=100
    )

    sampled = solver(init_sample, iterations=200)[0]["samples"]["variables"]
    assert_fn(sampled)

  @pytest.mark.solver
  def test_sggmc(self, kolmogorov_smirnov_setup):
    data_loader, batch_fn, potential_fn, full_pot_fn, init_sample, assert_fn =\
      kolmogorov_smirnov_setup

    solver = alias.sggmc(
      potential_fn,
      full_pot_fn,
      data_loader,
      cache_size=1,
      batch_size=1,
      first_step_size=0.5,
      last_step_size=0.1,
      burn_in=100
    )

    sampled = solver(init_sample, iterations=200)[0]["samples"]["variables"]
    assert_fn(sampled)

  @pytest.mark.solver
  def test_obabo(self, kolmogorov_smirnov_setup):
    data_loader, batch_fn, potential_fn, _, init_sample, assert_fn =\
      kolmogorov_smirnov_setup

    solver = alias.obabo(
      potential_fn,
      data_loader,
      cache_size=1,
      batch_size=1,
      first_step_size=0.5,
      last_step_size=0.1,
      friction=10,
      burn_in=100,
      accepted_samples=100
    )

    sampled = solver(init_sample, iterations=500)[0]["samples"]["variables"]
    assert_fn(sampled)

  @pytest.mark.solver
  def test_sghmc(self, kolmogorov_smirnov_setup):
    data_loader, batch_fn, potential_fn, _, init_sample, assert_fn =\
      kolmogorov_smirnov_setup

    solver = alias.sghmc(
      potential_fn,
      data_loader,
      cache_size=1,
      batch_size=1,
      integration_steps=5,
      first_step_size=0.5,
      last_step_size=0.1,
      friction=0.995,
      burn_in=100,
      accepted_samples=100,
      adapt_noise_model=False,
    )

    sampled = solver(init_sample, iterations=500)[0]["samples"]["variables"]

    assert_fn(sampled)


class TestAliasLinearRegression:

  @pytest.mark.solver
  def test_rms_prop(self, problem):
    data_loader, batch_fn, potential_fn, _, w, w_init = problem

    solver = alias.sgld(
      potential_fn,
      data_loader,
      cache_size=512,
      batch_size=10,
      first_step_size=0.05,
      last_step_size=0.001,
      burn_in=20000,
      accepted_samples=4000,
      rms_prop=True
    )

    results = solver(w_init, iterations=50000)

    # Check that the standard deviation is close
    assert jnp.all(
      jnp.abs(results[0]["samples"]["variables"]["sigma"] - 0.5) < 0.5)

  @pytest.mark.solver
  def test_re_sgld(self, problem):
    data_loader, batch_fn, potential_fn, _, w, w_init = problem

    solver = alias.re_sgld(
      potential_fn,
      data_loader,
      cache_size=512,
      batch_size=10,
      first_step_size=0.0001,
      last_step_size=0.000005,
      burn_in=20000,
      accepted_samples=4000,
      temperature=100.0
    )

    results = solver((w_init, w_init), iterations=50000)

    # Check that the standard deviation is close
    assert jnp.all(
      jnp.abs(results[0]["samples"]["variables"]["sigma"] - 0.7) < 0.7)

  @pytest.mark.solver
  def test_amagold(self, problem):
    data_loader, batch_fn, potential_fn, full_potential_fn, w, w_init = problem

    solver = alias.amagold(
      potential_fn,
      full_potential_fn,
      data_loader,
      cache_size=512,
      batch_size=64,
      first_step_size=0.005,
      last_step_size=0.0005,
      burn_in=2000
    )

    results = solver(w_init, iterations=50000)

    # Check that the standard deviation is close
    assert jnp.all(
      jnp.abs(results[0]["samples"]["variables"]["sigma"] - 0.5) < 0.5)

  @pytest.mark.solver
  def test_sggmc(self, problem):
    data_loader, batch_fn, potential_fn, full_potential_fn, w, w_init = problem

    solver = alias.sggmc(
      potential_fn,
      full_potential_fn,
      data_loader,
      cache_size=512,
      batch_size=64,
      first_step_size=0.005,
      last_step_size=0.0005,
      burn_in=2000
    )

    results = solver(w_init, iterations=50000)

    # Check that the standard deviation is close
    assert jnp.all(
      jnp.abs(results[0]["samples"]["variables"]["sigma"] - 0.5) < 0.5)

  @pytest.mark.solver
  def test_obabo(self, problem):
    data_loader, batch_fn, potential_fn, _, w, w_init = problem

    solver = alias.obabo(
      potential_fn,
      data_loader,
      cache_size=512,
      batch_size=10,
      first_step_size=0.05,
      last_step_size=0.001,
      friction=1000,
      burn_in=2000
    )

    results = solver(w_init, iterations=50000)

    # Check that the standard deviation is close
    assert jnp.all(
      jnp.abs(results[0]["samples"]["variables"]["sigma"] - 0.5) < 0.5)

  @pytest.mark.solver
  def test_sghmc(self, problem):
    data_loader, batch_fn, potential_fn, _, w, w_init = problem

    solver = alias.sghmc(
      potential_fn,
      data_loader,
      cache_size=512,
      batch_size=10,
      integration_steps=5,
      first_step_size=0.05,
      last_step_size=0.001,
      friction=0.5,
      burn_in=5000,
      adapt_noise_model=True,
      diagonal_noise=False,
    )

    results = solver(w_init, iterations=10000)

    # Check that the standard deviation is close
    assert jnp.all(
      jnp.abs(results[0]["samples"]["variables"]["sigma"] - 0.5) < 0.5)
