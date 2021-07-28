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

"""Popular solvers ready to use.

While jax-sgmc has been designed to be flexible, starting with full
flexibility can be complicated. Therefore, this file contains some popular
solvers with preset properties, which can be applied directly to the problem or
used as a guide to setup a custom solver.

Usage
------

This toy example performs a small bayesian linear regression problem on a toy
dataset. It contains all necessary steps to use a solver from ``alias.py``.

.. doctest::

  >>> import jax
  >>> import jax.numpy as jnp
  >>> from jax import random
  >>> from jax.scipy.stats import norm
  >>>
  >>> from jax_sgmc import data, alias, potential

Setup Data Loader
__________________

In this example we first have to generate some data. As the data is already in
the form of jnp-arrays, constructing a :class:`NumpyDataLoader` is the simplest
option. For many datasets such as mnist, using the `TFDataLoader` is recomended,
as it accepts datasets from `tensorflow_datasets`.

  >>> N = 4
  >>> samples = 1000  # Total samples
  >>> sigma = 0.5
  >>>
  >>> key = random.PRNGKey(0)
  >>> split1, split2, split3 = random.split(key, 3)
  >>>
  >>> w = random.uniform(split3, minval=-1, maxval=1, shape=(N, 1))
  >>> noise = sigma * random.normal(split2, shape=(samples, 1))
  >>> x = random.uniform(split1, minval=-10, maxval=10, shape=(samples, N))
  >>> x = jnp.stack([x[:, 0] + x[:, 1], x[:, 1], 0.1 * x[:, 2] - 0.5 * x[:, 3],
  ...                x[:, 3]]).transpose()
  >>> y = jnp.matmul(x, w) + noise
  >>>
  >>> data_loader = data.NumpyDataLoader(x=x, y=y)

Likelihood and Prior
____________________

The likelihood and prior wrap the model to be investigated. The potential
modules takes care of combining both into a single potential function, which
can be applied to a batch of data.

  >>> def model(sample, observations):
  ...   weights = sample["w"]
  ...   predictors = observations["x"]
  ...   return jnp.dot(predictors, weights)
  >>>
  >>> def likelihood(sample, observations):
  ...   sigma = sample["sigma"]
  ...   y = observations["y"]
  ...   y_pred = model(sample, observations)
  ...   return norm.logpdf(y - y_pred, scale=sigma)
  >>>
  >>> def prior(unused_sample):
  ...   return 0.0
  >>>
  >>> potential_fn = potential.minibatch_potential(prior=prior,
  ...                                              likelihood=likelihood,
  ...                                              strategy='vmap')
  >>> full_potential_fn = potential.full_potential(prior=prior,
  ...                                              likelihood=likelihood,
  ...                                              strategy='vmap')

"""

from functools import partial
from typing import Any

from jax_sgmc import data, potential, adaption, integrator, scheduler, solver, io

Pytree = Any

def sgld(potential_fn: potential.minibatch_potential,
         data_loader: data.DataLoader,
         cache_size: int = 512,
         batch_size: int = 32,
         first_step_size: float = 0.05,
         last_step_size: float = 0.001,
         burn_in: int = 0,
         accepted_samples: int = 1000,
         rms_prop: bool = False,
         alpha: float = 0.9,
         lmbd: float = 1e-5,
         save_to_numpy: bool = True):
  """Stochastic Gradient Langevin Dynamics.

  SGLD with a polynomial step size schedule and optional speed up via RMSprop-
  adaption [1].

  [1] https://arxiv.org/abs/1512.07666

    >>> rms_run = alias.sgld(potential_fn,
    ...                      data_loader,
    ...                      cache_size=512,
    ...                      batch_size=10,
    ...                      first_step_size=0.05,
    ...                      last_step_size=0.001,
    ...                      burn_in=20000,
    ...                      accepted_samples=4000,
    ...                      rms_prop=True)
    >>>
    >>> sample = {"w": jnp.zeros((N, 1)), "sigma": jnp.array(10.0)}
    >>> results = rms_run(sample, iterations=50000)[0]['samples']['variables']
    >>>
    >>> print(results['sigma'])
    [0.48556787 0.4838285  0.4860216  ... 0.49602574 0.49936494 0.49983683]

  Args:
    potential_fn: Stochastic potential over a minibatch of data
    data_loader: Data loader, e. g. numpy data loader
    cache_size: Number of mini_batches in device memory
    batch_size: Number of observations per batch
    first_step_size: First step size
    last_step_size: Final step size
    burn_in: Number of samples to skip before collecting samples
    accepted_samples: Total number of samples to collect, will be determined by
      random thinning if accepted samples < iterations - burn_in
    rms_prop: Whether to adapt a manifold via the RMSprop strategy
    alpha: Decay speed of previous manifold approximations
    lmbd: Stabilization parameter
    save_to_numpy: Save on host in numpy array instead of in device memory

  Returns:
    Returns a solver function which can be applied to multiple chains starting
    at ``init_sample``.

  """

  random_data = data.random_reference_data(data_loader, cache_size, batch_size)

  if rms_prop:
    rms_prop = adaption.rms_prop()
  else:
    rms_prop = None

  rms_integrator = integrator.langevin_diffusion(
    potential_fn, random_data, adaption=rms_prop)

  step_size_schedule = scheduler.polynomial_step_size_first_last(
    first=first_step_size, last=last_step_size)
  burn_in_schedule = scheduler.initial_burn_in(burn_in)
  random_thinning_schedule = scheduler.random_thinning(
    step_size_schedule, burn_in_schedule, selections=accepted_samples)
  schedule = scheduler.init_scheduler(
    step_size=step_size_schedule,
    burn_in=burn_in_schedule,
    thinning=random_thinning_schedule)

  if save_to_numpy:
    data_collector = io.MemoryCollector()
    saving = io.save(data_collector)
  else:
    saving = None

  sgld_solver = solver.sgmc(rms_integrator)
  mcmc = solver.mcmc(sgld_solver, schedule, strategy='map', saving=saving)

  def run_fn(*init_samples, iterations = 1000):
    init_with_adaption_kwargs = partial(
      sgld_solver[0],
      adaption_kwargs={'alpha': alpha, 'lmbd': lmbd})
    states = map(init_with_adaption_kwargs, init_samples)
    return mcmc(*states, iterations=iterations)
  return run_fn

def re_sgld(potential_fn: potential.minibatch_potential,
            data_loader: data.DataLoader,
            cache_size: int = 512,
            batch_size: int = 32,
            temperature: float = 1000.0,
            first_step_size: float = 0.05,
            last_step_size: float = 0.001,
            burn_in: int = 0,
            accepted_samples: int = 100,
            save_to_numpy: bool = True):
  """Replica Exchange Stochastic Gradient Langevin Diffusion.

  reSGLD simulates a tempered and a default chain in parallel, which
  exchange samples at random following a (biased) markov jump process [1].

  [1] https://arxiv.org/abs/2008.05367v3

    >>> resgld_run = alias.re_sgld(potential_fn,
    ...                            data_loader,
    ...                            cache_size=512,
    ...                            batch_size=10,
    ...                            first_step_size=0.0001,
    ...                            last_step_size=0.000005,
    ...                            burn_in=20000,
    ...                            accepted_samples=4000,
    ...                            temperature=100.0)
    >>>
    >>> sample = {"w": jnp.zeros((N, 1)), "sigma": jnp.array(2.0)}
    >>> init_samples = [(sample, sample), (sample, sample), (sample, sample)]
    >>>
    >>> results = resgld_run(*init_samples, iterations=50000)[0]['samples']['variables']
    >>>
    >>> print(results['sigma'])
    [0.7344817  0.78028935 0.70692766 ... 0.7185406  0.6215274  0.74618036]

  Args:
    potential_fn: Stochastic potential over a minibatch of data
    data_loader: Data loader, e. g. numpy data loader
    cache_size: Number of mini_batches in device memory
    batch_size: Number of observations per batch
    temperature: Temperature at which the helper chain should run
    first_step_size: First step size
    last_step_size: Final step size
    burn_in: Number of samples to skip before collecting samples
    accepted_samples: Total number of samples to collect, will be determined by
      random thinning if accepted samples < iterations - burn_in
    save_to_numpy: Save on host in numpy array instead of in device memory

  Returns:
    Returns a solver function which can be applied to multiple chains starting
    at ``start_chain_{idx}``.

  """

  random_data = data.random_reference_data(data_loader, cache_size, batch_size)

  resgld_integrator = integrator.langevin_diffusion(
    potential_fn, random_data)

  step_size_schedule = scheduler.polynomial_step_size_first_last(
    first=first_step_size, last=last_step_size)
  burn_in_schedule = scheduler.initial_burn_in(burn_in)
  random_thinning_schedule = scheduler.random_thinning(
    step_size_schedule, burn_in_schedule, selections=accepted_samples)
  temperature_schedule = scheduler.constant_temperature(1.0)
  schedule = scheduler.init_scheduler(
    step_size=step_size_schedule,
    burn_in=burn_in_schedule,
    thinning=random_thinning_schedule,
    temperature=temperature_schedule)

  if save_to_numpy:
    data_collector = io.MemoryCollector()
    saving = io.save(data_collector)
  else:
    saving = None

  resgld_solver = solver.parallel_tempering(resgld_integrator)
  mcmc = solver.mcmc(resgld_solver, schedule, strategy='map', saving=saving)

  def run_fn(*init_samples, iterations = 1000):
    states = map(resgld_solver[0], *zip(*init_samples))
    return mcmc(*states,
                iterations=iterations,
                schedulers=[{'temperature': {'tau': temperature}}])
  return run_fn

def amagold(stochastic_potential_fn: potential.StochasticPotential,
            full_potential_fn: potential.FullPotential,
            data_loader: data.DataLoader,
            cache_size: int = 512,
            batch_size: int = 32,
            integration_steps: int = 10,
            friction: float = 0.25,
            first_step_size: float = 0.05,
            last_step_size: float = 0.001,
            adaptive_step_size: bool = False,
            stabilization_constant: int = 10,
            decay_constant: float = 0.75,
            speed_constant: float = 0.05,
            target_acceptance_rate: float = 0.25,
            burn_in: int = 0,
            mass: Pytree = None,
            save_to_numpy: bool = True):
  """Amortized Metropolis Adjustment for Efficient Stochastic Gradient MCMC.

  The AMAGOLD solver constructs a skew-reversible markov chain, such that
  MH-correction steps can be applied periodically to sample from the correct
  distribution [1].

  [1] https://arxiv.org/abs/2003.00193

    >>> amagold_run = alias.amagold(potential_fn,
    ...                             full_potential_fn,
    ...                             data_loader,
    ...                             cache_size=512,
    ...                             batch_size=64,
    ...                             first_step_size=0.005,
    ...                             last_step_size=0.0005,
    ...                             burn_in=2000)
    >>>
    >>> sample = {"w": jnp.zeros((N, 1)), "sigma": jnp.array(2.0)}
    >>>
    >>> results = amagold_run(sample, iterations=5000)[0]['samples']['variables']
    >>>
    >>> print(results['sigma'])
    [0.5090904 0.5090904 0.5090904 ... 0.4953096 0.4953096 0.4981754]

  Args:
    stochastic_potential_fn: Stochastic potential over a minibatch of data
    full_potential_fn: Potential from full dataset
    data_loader: Data loader, e. g. numpy data loader
    cache_size: Number of mini_batches in device memory
    batch_size: Number of observations per batch
    integration_steps: Number of leapfrog-steps before each MH-correction step
    friction: Parameter between 0.0 and 1.0 controlling decay of momentum
    first_step_size: First step size for polynomial and adaptive step size
      schedule
    last_step_size: Final step size of the polynomial step size schedule
    adaptive_step_size: Adapt the step size to optimize the acceptance rate
      during burn in
    stabilization_constant: Larger numbers reduce the impact of the initial
      steps on the step size
    decay_constant: Larger values reduce impact of later steps
    speed_constant: Speed of adaption of the step size
    target_acceptance_rate: Target of the adption of the step sizes
    burn_in: Number of samples to skip before collecting samples
    mass: Diagonal mass for HMC-dynamics
    save_to_numpy: Save on host in numpy array instead of in device memory

  Returns:
    Returns a solver function which performs the AMAGOLD algorithm.

  """

  random_data = data.random_reference_data(data_loader, cache_size, batch_size)
  full_data_map = data.full_reference_data(data_loader, cache_size, batch_size)

  reversible_leapfrog = integrator.reversible_leapfrog(
    stochastic_potential_fn, random_data, integration_steps, friction, mass)
  amagold_solver = solver.amagold(
    reversible_leapfrog, full_potential_fn, full_data_map)

  if adaptive_step_size:
    step_size_schedule = scheduler.adaptive_step_size(
      burn_in=burn_in,
      initial_step_size=first_step_size,
      stabilization_constant=stabilization_constant,
      decay_constant=decay_constant,
      speed_constant=speed_constant,
      target_acceptance_rate=target_acceptance_rate)
  else:
    step_size_schedule = scheduler.polynomial_step_size_first_last(
      first=first_step_size,
      last=last_step_size)
  burn_in_schedule = scheduler.initial_burn_in(burn_in)
  schedule = scheduler.init_scheduler(
    step_size=step_size_schedule,
    burn_in=burn_in_schedule)

  if save_to_numpy:
    data_collector = io.MemoryCollector()
    saving = io.save(data_collector)
  else:
    saving = None

  mcmc = solver.mcmc(amagold_solver, schedule, strategy='map', saving=saving)

  def run_fn(*init_samples, iterations=1000):
    states = map(amagold_solver[0], init_samples)
    return mcmc(*states, iterations=iterations)
  return run_fn