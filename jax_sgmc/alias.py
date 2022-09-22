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

While JaxSGMC has been designed to be flexible, starting with full
flexibility can be complicated. Therefore, this file contains some popular
solvers with preset properties, which can be applied directly to the problem or
used as a guide to set up a custom solver.
"""
import warnings
from functools import partial
from typing import Any, Union

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
         save_to_numpy: bool = True,
         progress_bar: bool = True):
  """Stochastic Gradient Langevin Dynamics.

  SGLD with a polynomial step size schedule and optional speed up via RMS-prop
  adaption [1].

  [1] https://arxiv.org/abs/1512.07666

  ::

    rms_run = alias.sgld(...)

    sample = {"w": jnp.zeros((N, 1)), "sigma": jnp.array(10.0)}
    results = rms_run(sample, init_model_state=0, iterations=50000)[0]['samples']['variables']

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
    progress_bar: Print the progress of the solver

  Returns:
    Returns a solver function which can be applied to multiple chains starting
    at ``init_sample``. If the likelihood is stateful, an initial state must be
    provided.

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
    thinning=random_thinning_schedule,
    progress_bar=progress_bar)

  if save_to_numpy:
    data_collector = io.MemoryCollector()
    saving = io.save(data_collector)
  else:
    saving = None

  sgld_solver = solver.sgmc(rms_integrator)
  mcmc = solver.mcmc(sgld_solver, schedule, strategy='map', saving=saving)

  def run_fn(*init_samples, init_model_state: Pytree = None, iterations = 1000):
    init_with_adaption_kwargs = partial(
      sgld_solver[0],
      adaption_kwargs={
        'alpha': alpha,
        'lmbd': lmbd,
      },
      init_model_state=init_model_state)
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
            save_to_numpy: bool = True,
            progress_bar: bool = True):
  """Replica Exchange Stochastic Gradient Langevin Diffusion.

  reSGLD simulates a tempered and a default chain in parallel, which
  exchange samples at random following a (biased) markov jump process [1].

  [1] https://arxiv.org/abs/2008.05367v3

  ::

    resgld_run = alias.re_sgld(...)

    sample = {"w": jnp.zeros((N, 1)), "sigma": jnp.array(2.0)}
    init_samples = [(sample, sample), (sample, sample), (sample, sample)]

    results = resgld_run(
      *init_samples, init_model_state=0, iterations=50000
      )[0]['samples']['variables']

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
    progress_bar: Print the progress of the solver


  Returns:
    Returns a solver function which can be applied to multiple chains starting
    at ``start_chain_{idx}``.

  """
  del progress_bar

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
    temperature=temperature_schedule,
    progress_bar=False)

  if save_to_numpy:
    data_collector = io.MemoryCollector()
    saving = io.save(data_collector)
  else:
    saving = None

  resgld_solver = solver.parallel_tempering(resgld_integrator)
  mcmc = solver.mcmc(resgld_solver, schedule, strategy='map', saving=saving)

  def run_fn(*init_samples, init_model_state: Pytree = None, iterations = 1000):
    init_resgld_fn = partial(
      resgld_solver[0],
      init_model_state=init_model_state)
    states = map(init_resgld_fn, *zip(*init_samples))
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
            first_step_size: float = 0.001,
            last_step_size: float = 0.001,
            adaptive_step_size: bool = False,
            stabilization_constant: int = 10,
            decay_constant: float = 0.75,
            speed_constant: float = 0.05,
            target_acceptance_rate: float = 0.25,
            burn_in: int = 0,
            accepted_samples: Union[int, None] = None,
            mass: Pytree = None,
            save_to_numpy: bool = True,
            progress_bar: bool = True):
  """Amortized Metropolis Adjustment for Efficient Stochastic Gradient MCMC.

  The AMAGOLD solver constructs a skew-reversible markov chain, such that
  MH-correction steps can be applied periodically to sample from the correct
  distribution [1].

  [1] https://arxiv.org/abs/2003.00193

  ::

    amagold_run = alias.amagold(...)

    sample = {"w": jnp.zeros((N, 1)), "sigma": jnp.array(2.0)}

    results = amagold_run(
      sample, init_model_state=0, iterations=5000
      )[0]['samples']['variables']

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
    accepted_samples: Total number of samples to collect, will be determined by
      random thinning if accepted samples < iterations - burn_in.
      If None, no thinning wil be applied.
    mass: Diagonal mass for HMC-dynamics
    save_to_numpy: Save on host in numpy array instead of in device memory
    progress_bar: Print the progress of the solver

  Returns:
    Returns a solver function which can be applied to multiple chains starting
    at ``init_sample``.

  """

  random_data = data.random_reference_data(data_loader, cache_size, batch_size)
  full_data_map = data.full_reference_data(data_loader, cache_size, batch_size)

  reversible_leapfrog = integrator.reversible_leapfrog(
    stochastic_potential_fn, random_data, integration_steps, friction, mass)
  amagold_solver = solver.amagold(
    reversible_leapfrog, full_potential_fn, full_data_map)

  burn_in_schedule = scheduler.initial_burn_in(burn_in)
  if adaptive_step_size:
    step_size_schedule = scheduler.adaptive_step_size(
      burn_in=burn_in,
      initial_step_size=first_step_size,
      stabilization_constant=stabilization_constant,
      decay_constant=decay_constant,
      speed_constant=speed_constant,
      target_acceptance_rate=target_acceptance_rate)
    random_thinning_schedule = None
    assert accepted_samples is None, ('Thinning currently not supported for'
                                      ' adaptive step size.')
  else:
    step_size_schedule = scheduler.polynomial_step_size_first_last(
      first=first_step_size,
      last=last_step_size)
    if accepted_samples is None:
      random_thinning_schedule = None
    else:
      random_thinning_schedule = scheduler.random_thinning(
        step_size_schedule, burn_in_schedule, selections=accepted_samples)
  schedule = scheduler.init_scheduler(
    step_size=step_size_schedule,
    burn_in=burn_in_schedule,
    thinning=random_thinning_schedule,
    progress_bar=progress_bar)

  if save_to_numpy:
    data_collector = io.MemoryCollector()
    saving = io.save(data_collector)
  else:
    saving = None

  mcmc = solver.mcmc(amagold_solver, schedule, strategy='map', saving=saving)

  def run_fn(*init_samples, init_model_state: Pytree = None, iterations=1000):
    init_amagold_fn = partial(
      amagold_solver[0],
      init_model_state = init_model_state)
    states = map(init_amagold_fn, init_samples)
    return mcmc(*states, iterations=iterations)
  return run_fn

def sggmc(stochastic_potential_fn: potential.StochasticPotential,
          full_potential_fn: potential.FullPotential,
          data_loader: data.DataLoader,
          cache_size: int = 512,
          batch_size: int = 32,
          integration_steps: int = 10,
          friction_coefficient: float = 1.0,
          first_step_size: float = 0.001,
          last_step_size: float = 0.001,
          adaptive_step_size: bool = False,
          stabilization_constant: int = 10,
          decay_constant: float = 0.75,
          speed_constant: float = 0.05,
          target_acceptance_rate: float = 0.25,
          burn_in: int = 0,
          accepted_samples: Union[int, None] = None,
          mass: Pytree = None,
          save_to_numpy: bool = True,
          progress_bar: bool = True):
  """Stochastic gradient guided monte carlo.

  The SGGMC solver is based on the OBABO integrator, which is reversible
  when using stochastic gradients. Moreover, the calculation of the full
  potential is only necessary once per MH-correction step, which can be applied
  after multiple iterations [1].

  [1] https://arxiv.org/abs/2102.01691

  ::

    sggmc_run = alias.sggmc(...)

    sample = {"w": jnp.zeros((N, 1)), "sigma": jnp.array(2.0)}

    results = sggmc_run(
      sample, init_model_state=0, iterations=5000
      )[0]['samples']['variables']

  Args:
    stochastic_potential_fn: Stochastic potential over a minibatch of data
    full_potential_fn: Potential from full dataset
    data_loader: Data loader, e. g. numpy data loader
    cache_size: Number of mini_batches in device memory
    batch_size: Number of observations per batch
    integration_steps: Number of leapfrog-steps before each MH-correction step
    friction_coefficient: Positive parameter controling amount of refreshed
     momentum
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
    accepted_samples: Total number of samples to collect, will be determined by
      random thinning if accepted samples < iterations - burn_in.
      If None, no thinning wil be applied.
    mass: Diagonal mass for HMC-dynamics
    save_to_numpy: Save on host in numpy array instead of in device memory
    progress_bar: Print the progress of the solver

  Returns:
    Returns a solver function which can be applied to multiple chains starting
    at ``init_sample``.

  """

  random_data = data.random_reference_data(data_loader, cache_size, batch_size)
  full_data_map = data.full_reference_data(data_loader, cache_size, batch_size)

  obabo = integrator.obabo(
    stochastic_potential_fn, random_data, integration_steps,
    friction_coefficient, mass)
  sggmc_solver = solver.sggmc(
    obabo, full_potential_fn, full_data_map)

  burn_in_schedule = scheduler.initial_burn_in(burn_in)
  if adaptive_step_size:
    step_size_schedule = scheduler.adaptive_step_size(
      burn_in=burn_in,
      initial_step_size=first_step_size,
      stabilization_constant=stabilization_constant,
      decay_constant=decay_constant,
      speed_constant=speed_constant,
      target_acceptance_rate=target_acceptance_rate)
    random_thinning_schedule = None
    assert accepted_samples is None, ('Thinning currently not supported for'
                                      ' adaptive step size.')
  else:
    step_size_schedule = scheduler.polynomial_step_size_first_last(
      first=first_step_size,
      last=last_step_size)
    if accepted_samples is None:
      random_thinning_schedule = None
    else:
      random_thinning_schedule = scheduler.random_thinning(
        step_size_schedule, burn_in_schedule, selections=accepted_samples)
  schedule = scheduler.init_scheduler(
    step_size=step_size_schedule,
    burn_in=burn_in_schedule,
    thinning=random_thinning_schedule,
    progress_bar=progress_bar)

  if save_to_numpy:
    data_collector = io.MemoryCollector()
    saving = io.save(data_collector)
  else:
    saving = None

  mcmc = solver.mcmc(sggmc_solver, schedule, strategy='map', saving=saving)

  def run_fn(*init_samples, init_model_state: Pytree = None, iterations=1000):
    init_sggmc_fn = partial(sggmc_solver[0], init_model_state=init_model_state)
    states = map(init_sggmc_fn, init_samples)
    return mcmc(*states, iterations=iterations)
  return run_fn

def sghmc(potential_fn: potential.minibatch_potential,
          data_loader: data.DataLoader,
          cache_size: int = 512,
          batch_size: int = 32,
          integration_steps: int = 10,
          friction: Union[float, Pytree] = 1.0,
          mass: Pytree = None,
          first_step_size: float = 0.05,
          last_step_size: float = 0.001,
          burn_in: int = 0,
          accepted_samples: int = 1000,
          adapt_noise_model: bool = False,
          diagonal_noise: bool = True,
          save_to_numpy: bool = True,
          progress_bar: bool = True):
  """Stochastic Gradient Hamiltonian Monte Carlo.

  SGHMC improves the exploratory power of SGLD by introducing momentum [1].

  [1] https://arxiv.org/abs/1402.4102

  ::

    sghmc_run = alias.sghmc(...)

    sample = {"w": jnp.zeros((N, 1)), "sigma": jnp.array(2.0)}
    results = sghmc_run(sample, init_model_state=0, iterations=5000)[0]['samples']['variables']

  Args:
    potential_fn: Stochastic potential over a minibatch of data
    data_loader: Data loader, e. g. numpy data loader
    cache_size: Number of mini_batches in device memory
    batch_size: Number of observations per batch
    integration_steps: Number of leapfrog steps before resampling the momentum
    friction: Friction to counteract noise introduced by stochastic gradients.
      Can be specified for each variable or for all variables (scalar value)
    mass: Diagonal mass to be used for hamiltonian dynamics
    first_step_size: First step size
    last_step_size: Final step size
    burn_in: Number of samples to skip before collecting samples
    accepted_samples: Total number of samples to collect, will be determined by
      random thinning if accepted samples < iterations - burn_in
    adapt_noise_model: Estimate the gradient noise to speed up the convergence.
    diagonal_noise: Restrict the noise estimate to be diagonal.
    save_to_numpy: Save on host in numpy array instead of in device memory
    progress_bar: Print the progress of the solver

  Returns:
    Returns a solver function which can be applied to multiple chains starting
    at ``init_sample``.

  """

  random_data = data.random_reference_data(data_loader, cache_size, batch_size)

  if adapt_noise_model:
    noise_model = adaption.fisher_information(minibatch_potential=potential_fn,
                                              diagonal=diagonal_noise)
  else:
    noise_model = None

  friction_leapfrog = integrator.friction_leapfrog(
    potential_fn, random_data, friction=friction, const_mass=mass,
    steps=integration_steps, noise_model=noise_model)

  step_size_schedule = scheduler.polynomial_step_size_first_last(
    first=first_step_size, last=last_step_size)
  burn_in_schedule = scheduler.initial_burn_in(burn_in)
  random_thinning_schedule = scheduler.random_thinning(
    step_size_schedule, burn_in_schedule, selections=accepted_samples)
  schedule = scheduler.init_scheduler(
    step_size=step_size_schedule,
    burn_in=burn_in_schedule,
    thinning=random_thinning_schedule,
    progress_bar=progress_bar)

  if save_to_numpy:
    data_collector = io.MemoryCollector()
    saving = io.save(data_collector)
  else:
    saving = None

  sghmc_solver = solver.sgmc(friction_leapfrog)
  mcmc = solver.mcmc(sghmc_solver, schedule, strategy='map', saving=saving)

  def run_fn(*init_samples, init_model_state: Pytree = None, iterations = 1000):
    init_sghmc_fn = partial(sghmc_solver[0], init_model_state=init_model_state)
    states = map(init_sghmc_fn, init_samples)
    return mcmc(*states, iterations=iterations)
  return run_fn

def obabo(potential_fn: potential.minibatch_potential,
          data_loader: data.DataLoader,
          cache_size: int = 512,
          batch_size: int = 32,
          integration_steps: int = 10,
          friction: Union[float, Pytree] = 1.0,
          mass: Pytree = None,
          first_step_size: float = 0.05,
          last_step_size: float = 0.001,
          burn_in: int = 0,
          accepted_samples: int = 1000,
          save_to_numpy: bool = True,
          progress_bar: bool = True):
  """Langevin Monte Carlo with partial momentum refreshment.

  [1] https://arxiv.org/abs/2102.01691

  ::

    sghmc_run = alias.obabo(...)

    sample = {"w": jnp.zeros((N, 1)), "sigma": jnp.array(2.0)}
    results = sghmc_run(sample, init_model_state=0, iterations=5000)[0]['samples']['variables']

  Args:
    potential_fn: Stochastic potential over a minibatch of data
    data_loader: Data loader, e. g. numpy data loader
    cache_size: Number of mini_batches in device memory
    batch_size: Number of observations per batch
    integration_steps: Number of leapfrog steps before resampling the momentum
    friction: Positive parameter controling amount of refreshed momentum
    mass: Diagonal mass to be used for hamiltonian dynamics
    first_step_size: First step size
    last_step_size: Final step size
    burn_in: Number of samples to skip before collecting samples
    accepted_samples: Total number of samples to collect, will be determined by
      random thinning if accepted samples < iterations - burn_in
    save_to_numpy: Save on host in numpy array instead of in device memory
    progress_bar: Print the progress of the solver

  Returns:
    Returns a solver function which can be applied to multiple chains starting
    at ``init_sample``.

  """

  random_data = data.random_reference_data(data_loader, cache_size, batch_size)

  obabo_integrator = integrator.obabo(
    potential_fn=potential_fn, batch_fn=random_data, steps=integration_steps,
    friction=friction, const_mass=mass
  )

  step_size_schedule = scheduler.polynomial_step_size_first_last(
    first=first_step_size, last=last_step_size)
  burn_in_schedule = scheduler.initial_burn_in(burn_in)
  random_thinning_schedule = scheduler.random_thinning(
    step_size_schedule, burn_in_schedule, selections=accepted_samples)
  schedule = scheduler.init_scheduler(
    step_size=step_size_schedule,
    burn_in=burn_in_schedule,
    thinning=random_thinning_schedule,
    progress_bar=progress_bar)

  if save_to_numpy:
    data_collector = io.MemoryCollector()
    saving = io.save(data_collector)
  else:
    saving = None

  sghmc_solver = solver.sgmc(obabo_integrator)
  mcmc = solver.mcmc(sghmc_solver, schedule, strategy='map', saving=saving)

  def run_fn(*init_samples, init_model_state: Pytree = None, iterations = 1000):
    init_sghmc_fn = partial(sghmc_solver[0], init_model_state=init_model_state)
    states = map(init_sghmc_fn, init_samples)
    return mcmc(*states, iterations=iterations)
  return run_fn
