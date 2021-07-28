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

"""

from functools import partial

from jax_sgmc import data, potential, adaption, integrator, scheduler, solver, io

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

    Usage:

    .. highlight:: python

    ::

      run = jax_sgmc.alias.sgld(...)
      results = run(*init_samples, iterations=1000)
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
    return mcmc(*states, iterations=iterations)['samples']
  return run_fn
