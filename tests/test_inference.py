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
from jax import test_util

import pytest

from jax_sgmc import adaption
from jax_sgmc import data
from jax_sgmc import potential
from jax_sgmc import scheduler
from jax_sgmc import solver
from jax_sgmc import integrator

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

  data_loader = data.NumpyDataLoader(x=x, y=y)
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
  return data_loader, batch_fn, potential_fn, w, w_init

class TestSGLD:

  def test_default(self, problem):

    data_loader, batch_fn, potential_fn, w, w_init = problem

    default_integrator = integrator.langevin_diffusion(potential_fn,
                                                       batch_fn)
    default_step_size = scheduler.polynomial_step_size_first_last(first=0.001,
                                                                  last=0.000005)
    burn_in = scheduler.initial_burn_in(n=20000)
    default_random_thinning = scheduler.random_thinning(
      step_size_schedule=default_step_size,
      burn_in_schedule=burn_in,
      selections=4000)
    default_scheduler = scheduler.init_scheduler(step_size=default_step_size,
                                                 burn_in=burn_in,
                                                 thinning=default_random_thinning)
    default_sgld = solver.sgmc(default_integrator)
    default_run = solver.mcmc(default_sgld, default_scheduler)
    default = default_run(default_integrator[0](w_init), iterations=50000)

    # Check that the standard deviation is close
    assert jnp.all(jnp.abs(default[0]["samples"]["variables"]["sigma"] - 0.5)  < 0.5)

  def test_rms(self, problem):

    data_loader, batch_fn, potential_fn, w, w_init = problem

    rms_prop = adaption.rms_prop()
    rms_integrator = integrator.langevin_diffusion(potential_fn,
                                                   batch_fn,
                                                   rms_prop)
    rms_step_size = scheduler.polynomial_step_size_first_last(first=0.05,
                                                              last=0.001)
    burn_in = scheduler.initial_burn_in(n=20000)
    rms_random_thinning = scheduler.random_thinning(
      step_size_schedule=rms_step_size,
      burn_in_schedule=burn_in,
      selections=4000)
    rms_scheduler = scheduler.init_scheduler(step_size=rms_step_size,
                                             burn_in=burn_in,
                                             thinning=rms_random_thinning)
    rms_sgld = solver.sgmc(rms_integrator)
    rms_run = solver.mcmc(rms_sgld, rms_scheduler)
    rms = rms_run(rms_integrator[0](w_init), iterations=50000)

    # Check that the standard deviation is close
    assert jnp.all(jnp.abs(rms[0]["samples"]["variables"]["sigma"] - 0.5)  < 0.5)
