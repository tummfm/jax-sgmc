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

"""Solvers for Stochastic Gradient Bayesian Inference."""

from functools import partial
from typing import Callable, Any, Tuple, NamedTuple, Dict, Union

from jax import lax, jit, random, named_call
import jax.numpy as jnp
from jax_sgmc import io, util, data, integrator, potential

PyTree = Any
Array = util.Array

class AMAGOLDState(NamedTuple):
  """State of the AMAGOLD solver.

  Attributes:
    full_data_state: Cache state for full data mapping function.
    potential: True potential of the current sample
    integrator_state: State of the reversible leapfrog integrator
    key: PRNGKey for MH correction step
    acceptance_ratio: Acceptance ratio used in last step.
    mass_state: State of the mass adaption

  """
  full_data_state: data.CacheState
  potential: Array
  integrator_state: integrator.LeapfrogState
  key: Array
  acceptance_ratio: Array
  mass_state: PyTree

class SGGMCState(NamedTuple):
  """State of the AMAGOLD solver.

  Attributes:
    full_data_state: Cache state for full data mapping function.
    potential: True potential of the current sample
    integrator_state: State of the reversible leapfrog integrator
    key: PRNGKey for MH correction step
    acceptance_ratio: Acceptance ratio used in last step.
    mass_state: State of the mass adaption

  """
  full_data_state: data.CacheState
  potential: Array
  integrator_state: integrator.ObaboState
  key: Array
  acceptance_ratio: Array
  mass_state: PyTree

def mcmc(solver,
         scheduler,
         strategy="map",
         saving=None,
         loading=None):
  """Runs the solver for multiple chains and saves the collected samples

  Args:
    solver: Computes the next state form a given state.
    scheduler: Schedules solver parameters such as temperature and burn in.
    strategy: Run multiple chains in parallel or vectorized
    saving: Save samples via host_callback (if saving requires much memory)
    loading: Restore a previously saved checkpoint (scheduler state and solver
      state)

  Returns:
    Returns function which runs the solver for a given number of iterations.

  """

  if saving is None:
    # Pass the state
    init_saving, save, postprocess_saving = io.no_save()
  else:
    init_saving, save, postprocess_saving = saving

  scheduler_init, scheduler_next, scheduler_get = scheduler
  _, solver_update, solver_get = solver

  @partial(jit, static_argnums=(0,))
  def _uninitialized_update(unused_static_information, state, _):
    solver_state, scheduler_states, saving_state = state
    if len(scheduler_states) == 1:
      schedule = scheduler_get(scheduler_states[0])
      schedules = [schedule]
    else:
      schedules = util.list_vmap(scheduler_get)(*scheduler_states)
    new_state, stats = solver_update(solver_state, *schedules)
    # Kepp the sample if it is not subject to burn in or thinning.
    keep = jnp.logical_and(schedules[0].burn_in, schedules[0].accept)
    saving_state, saved = save(
      saving_state,
      keep,
      solver_get(new_state),
      scheduler_state=scheduler_states,
      solver_state=solver_state)
    # Update the scheduler with additional information from the integrator, such
    # as the acceptance ratio of AMAGOLD
    if stats is not None:
      next_fn = partial(scheduler_next, **stats)
    else:
      next_fn = scheduler_next

    if len(scheduler_states) == 1:
      scheduler_state = next_fn(scheduler_states[0])
      scheduler_states = [scheduler_state]
    else:
      scheduler_states = util.list_vmap(next_fn)(*scheduler_states)
    return (new_state, scheduler_states, saving_state), saved

  # Scan over the update function. Postprocessing the samples is required if
  # saving is None to sort out not accepted samples. Samples are not accepted
  # due tu burn in or thinning.

  # Initialize a single chain
  def _init(init_state, iterations, schedulers=None):
    main_scheduler, static_information = scheduler_init(iterations)
    # Some solvers might require additional schedulers, such as parallel
    # tempering.
    if schedulers is not None:
      helper_schedulers = [scheduler_init(iterations, **kwargs)[0]
                           for kwargs in schedulers]
      scheduler_states = [main_scheduler] + helper_schedulers
    else:
      scheduler_states = [main_scheduler]
    if loading is None:
      # Define what the saving module is expected to save
      saving_state = init_saving(
        solver_get(init_state),
        (init_state, scheduler_states),
        static_information)
    else:
      raise NotImplementedError("Loading of checkpoints is currently not "
                                "supported.")
    # Return tuple of static and non-static information
    return (iterations, static_information), (init_state, scheduler_states, saving_state)

  @partial(jit, static_argnums=0)
  def _run(static, init_state):
    iterations, static_information = static
    _update = partial(_uninitialized_update, static_information)
    (_, _, saving_state), saved = lax.scan(
      _update,
      init_state,
      jnp.arange(iterations))
    return saving_state, saved

  def run(*states: Union[PyTree, Tuple[PyTree]],
          schedulers: Union[Dict, Tuple[Dict]] = None,
          iterations: int = 1e5):
    if strategy == "map":
      def run_single():
        for state in states:
          init_args = _init(state, iterations, schedulers=schedulers)
          results = _run(*init_args)
          yield postprocess_saving(*results)
      return list(run_single())
    else:
      # Initialize the states sequentially
      static_info, init_states = zip(
        *[_init(state, iterations, schedulers=schedulers) for state in states])
      # Static information is the same for all chains
      if strategy == "pmap":
        mapped_run = util.list_pmap(partial(_run, static_info[0]))
      elif strategy == "vmap":
        mapped_run = jit(util.list_vmap(partial(_run, static_info[0])))
      else:
        raise NotImplementedError(f"Strategy {strategy} is unknown. ")
      saving_states, saved_values = zip(*mapped_run(*init_states))
      return list(map(postprocess_saving, saving_states, saved_values))
  return run


def sgmc(integrator) -> Tuple[Callable, Callable, Callable]:
  """Initializes the standard SGLD - sampler.

  This sampler simply integrates without acceptance/rejection or parallel
  tempering.

  Args:
    integrator: sgld or leapfrog.

  Returns:
    Returns a solver which depends only on external variables from the
    scheduler, such as the step size.

  """

  init_integrator, update_integrator, get_integrator = integrator

  def init(*args, **kwargs):
    return init_integrator(*args, **kwargs)

  @partial(named_call, name='update_step')
  def update(state, schedule):

    # No statistics such as acceptance ratio
    return update_integrator(state, schedule), None

  def get(state) -> Dict[str, PyTree]:
    return get_integrator(state)

  return init, update, get


def parallel_tempering(integrator,
                       sa_schedule: Callable = lambda n: 1 / n
                       ) -> Tuple[Callable, Callable, Callable]:
  """Exchange samples from a normal and tempered chain.

  This solver runs an additional tempered chain, from which no samples are
  drawn. The normal chain and the additional chain exchange samples at random
  by a reversible jump process [1].

  [1] https://arxiv.org/abs/2008.05367v3

  Args:
    integrator: standard langevin diffusion integrator
    sa_schedule: learning rate schedule to estimate the standard deviation of
      the stochastic potential

  Returns:
    Returns the reSGLD solver.

  """

  init_integrator, update_integrator, get_integrator = integrator

  # Todo: Maybe also provide pmap?

  # Todo: Currently not supported because of missing stop_vmap implementation
  # vmapped_update = util.list_vmap(lambda args: update_integrator(*args))

  def init(normal_sample: PyTree,
           tempered_sample: PyTree,
           ssq_init: Array = jnp.array(0.0),
           key: Array = random.PRNGKey(0),
           F: Array = jnp.array(1.0),
           **kwargs):
    key, split1, split2 = random.split(key, 3)

    normal_chain = init_integrator(normal_sample, key=split1, **kwargs)
    hot_chain = init_integrator(tempered_sample, key=split2, **kwargs)

    return normal_chain, hot_chain, ssq_init, F, 0, key

  # Hot schedule is an additional schedule, which has no influence on the
  # saved samples.
  @partial(named_call, name='resgld_swap_step')
  def update(state, normal_schedule, hot_schedule):
    normal_chain, hot_chain, ssq_estimate, F, step, key = state

    step += 1

    # Todo: Replace with vmapped / pmapped update
    normal_chain = update_integrator(normal_chain, normal_schedule)
    hot_chain = update_integrator(hot_chain, hot_schedule)

    # Update of std_estimate
    step_size = sa_schedule(step)
    ssq_estimate = ((1 - step_size) * ssq_estimate
                     + step_size * normal_chain.variance)

    # Swapping step
    temps = 1 / normal_schedule.temperature - 1 / hot_schedule.temperature
    correction = temps * ssq_estimate / F
    log_s =  temps * (normal_chain.potential - hot_chain.potential - correction)

    key, split = random.split(key)
    log_u = jnp.log(random.uniform(split))

    # Swap the chains at random
    (normal_chain, hot_chain) = lax.cond(
      log_u < log_s,
      lambda cold_hot: cold_hot,
      lambda cold_hot: (cold_hot[1], cold_hot[0]),
      (normal_chain, hot_chain))

    return (normal_chain, hot_chain, ssq_estimate, F, step, key), None

  # Return only the results of the normally tempered chain
  def get(state) -> Dict[str, PyTree]:
    return get_integrator(state[0])

  return init, update, get

def amagold(integrator_fn,
            full_potential_fn: potential.FullPotential,
            full_data_map: data.OrderedBatch,
            mass_adaption: Callable = None
            ) -> Tuple[Callable, Callable, Callable]:
  """Initializes AMAGOLD integration.

  Args:
    integrator_fn: Reversible leapfrog integrator.
    full_potential_fn: Function to calculate true potential.
    full_data_map: Tuple returned by :func:`jax_sgmc.data.full_reference_data``
    mass_adaption: Function to adapt a constant mass during the burn in phase.

  Returns:
    Returns the AMAGOLD solver, a combination of reversible stochastic
    hamiltonian dynamics and amortized MH corrections steps.

  """

  init_integrator, update_integrator, get_integrator = integrator_fn
  init_full_data, full_data_map_fn, _ = full_data_map
  if mass_adaption:
    init_mass, update_mass, get_mass = mass_adaption

  def init(init_sample: PyTree,
           key: Array = random.PRNGKey(0),
           initial_mass: PyTree = None,
           full_data_kwargs: dict = None,
           **kwargs) -> AMAGOLDState:
    if mass_adaption:
      mass_state = init_mass(init_sample, initial_mass)
      mass = get_mass(mass_state)
    else:
      mass_state = None
      mass = None

    if not full_data_kwargs:
      full_data_kwargs = {}
    full_data_state = init_full_data(**full_data_kwargs)

    init_model_state = kwargs.get('init_model_state')

    potential, (full_data_state, model_state) = full_potential_fn(
      init_sample,
      full_data_state,
      full_data_map_fn,
      state=init_model_state)

    kwargs['init_model_state'] = model_state

    key, split = random.split(key)
    # Todo: Init with correct covariance
    integrator_state = init_integrator(init_sample, key=key, mass=mass, **kwargs)

    state = AMAGOLDState(
      integrator_state=integrator_state,
      potential=potential,
      full_data_state=full_data_state,
      key=split,
      acceptance_ratio=(jnp.array(0.0), jnp.array(0.0)),
      mass_state=mass_state)
    return state

  @partial(named_call, name='amagold_mh_step')
  def update(state: AMAGOLDState, schedule):
    if mass_adaption:
      mass = get_mass(state.mass_state)
    else:
      mass = None

    key, split = random.split(state.key, 2)
    proposal = update_integrator(
      state.integrator_state,
      schedule,
      mass=mass)

    # MH correction step
    new_potential, (full_data_state, new_model_state) = full_potential_fn(
      proposal.positions,
      state.full_data_state,
      full_data_map_fn,
      state=proposal.model_state)

    # Limit acceptance ratio to 1.0
    log_alpha = state.potential - new_potential + proposal.potential
    log_alpha = jnp.where(log_alpha > 0.0, 0.0, log_alpha)
    slice = random.uniform(split)

    # If true, the step is accepted
    mh_integrator_state, new_potential, direction, mh_model_state = lax.cond(
      jnp.log(slice) < log_alpha,
      lambda new_old: new_old[0],
      lambda new_old: new_old[1],
      ((proposal, new_potential, 1.0, new_model_state),
       (state.integrator_state, state.potential, -1.0,
        state.integrator_state.model_state)))

    new_integrator_state = integrator.LeapfrogState(
      positions=mh_integrator_state.positions,
      momentum=util.tree_scale(direction, mh_integrator_state.momentum),
      model_state=mh_model_state,
      potential=mh_integrator_state.potential,
      # These parameters must be provided from the updated state, otherwise
      # the noise and the random data is not resampled
      data_state=proposal.data_state,
      key=proposal.key)

    # Adapt the mass on the accepted sample
    if mass_adaption:
      mass_state = update_mass(state.mass_state, mh_integrator_state.positions)
    else:
      mass_state = None

    new_state = AMAGOLDState(
      key=key,
      integrator_state=new_integrator_state,
      potential=new_potential,
      full_data_state=full_data_state,
      acceptance_ratio=(jnp.exp(log_alpha), schedule.step_size),
      mass_state=mass_state)

    stats = {'acceptance_ratio': jnp.exp(log_alpha)}

    return new_state, stats

  def get(state: AMAGOLDState) -> Dict[str, PyTree]:
    int_dict = get_integrator(state.integrator_state)
    int_dict['acceptance_ratio'] = state.acceptance_ratio[0]
    int_dict['step_size'] = state.acceptance_ratio[1]
    return int_dict

  return init, update, get

def sggmc(integrator_fn,
          full_potential_fn: potential.FullPotential,
          full_data_map: data.OrderedBatch,
          mass_adaption: Callable = None
          ) -> Tuple[Callable, Callable, Callable]:
  """Gradient Guided Monte Carlo using Stochastic Gradients.

  The OBABO integration scheme is reversible even when using stochastic
  gradients and provides second order accuracy. Therefore, a MH-acceptance step
  can be applied to sample from the correct posterior distribution.

  [1] https://arxiv.org/abs/2102.01691

  Args:
    integrator_fn: Reversible leapfrog integrator.
    full_potential_fn: Function to calculate true potential.
    mass_adaption: Function to predict the mass during warmup.

  Returns:
    Returns the SGGMC solver.

  """

  init_integrator, update_integrator, get_integrator = integrator_fn
  init_full_data, full_data_map_fn, _ = full_data_map
  if mass_adaption:
    init_mass, update_mass, get_mass = mass_adaption

  def init(init_sample: PyTree,
           key: Array = random.PRNGKey(0),
           initial_mass: PyTree = None,
           full_data_kwargs: dict = None,
           **kwargs) -> SGGMCState:

    if not full_data_kwargs:
      full_data_kwargs = {}
    full_data_state = init_full_data(**full_data_kwargs)

    init_model_state = kwargs.get('init_model_state')

    full_pot, (full_data_state, new_model_state) = full_potential_fn(
      init_sample,
      full_data_state,
      full_data_map_fn,
      state=init_model_state)

    # Pass the new model state to the integrator
    kwargs['init_model_state'] = new_model_state

    key, split = random.split(key)
    integrator_state = init_integrator(init_sample, key=key, **kwargs)

    if mass_adaption:
      mass_state = init_mass(init_sample, initial_mass)
    else:
      mass_state = None

    state = SGGMCState(
      integrator_state=integrator_state,
      potential=full_pot,
      full_data_state=full_data_state,
      key=split,
      mass_state=mass_state,
      acceptance_ratio=(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)))
    return state

  @partial(named_call, name='sggmc_mh_step')
  def update(state: SGGMCState, schedule) -> Tuple[SGGMCState, Dict]:
    # Get the current mass
    if mass_adaption:
      mass = get_mass(state.mass_state)
    else:
      mass = None

    key, split = random.split(state.key, 2)
    proposal = update_integrator(
      state.integrator_state,
      schedule,
      mass=mass)

    # MH correction step
    new_potential, (full_data_state, new_model_state) = full_potential_fn(
      proposal.positions,
      state.full_data_state,
      full_data_map_fn,
      state=proposal.model_state)

    log_alpha = - 1.0 / schedule.temperature * (
      new_potential - state.potential
      + proposal.kinetic_energy_end - proposal.kinetic_energy_start)
    # Limit the probability to 1.0 to ensure correct calculation of acceptance
    # ratio statistics.
    log_alpha = jnp.where(log_alpha <= 0, log_alpha, 0.0)

    slice = random.uniform(split)

    # If true, the step is accepted
    mh_integrator_state, new_potential, mh_model_state = lax.cond(
      jnp.log(slice) < log_alpha,
      lambda new_old: new_old[0],
      lambda new_old: new_old[1],
      ((proposal, new_potential, new_model_state),
       (state.integrator_state, state.potential,
        state.integrator_state.model_state)))

    new_integrator_state = integrator.ObaboState(
      positions=mh_integrator_state.positions,
      momentum=mh_integrator_state.momentum,
      model_state=mh_model_state,
      potential=mh_integrator_state.potential,
      # These parameters must be provided from the updated state, otherwise
      # the noise and the random data is not resampled
      data_state=proposal.data_state,
      key=proposal.key,
      kinetic_energy_start=0.0,
      kinetic_energy_end=0.0)

    # Adapt the mass
    if mass_adaption:
      mass_state = update_mass(state.mass_state, mh_integrator_state.positions)
    else:
      mass_state = None

    new_state = SGGMCState(
      key=key,
      integrator_state=new_integrator_state,
      potential=new_potential,
      full_data_state=full_data_state,
      mass_state=mass_state,
      acceptance_ratio=(jnp.exp(log_alpha), schedule.step_size, proposal.kinetic_energy_end-proposal.kinetic_energy_start))

    stats = {'acceptance_ratio': jnp.exp(log_alpha)}

    return new_state, stats

  def get(state: SGGMCState) -> Dict[str, PyTree]:
    int_dict = get_integrator(state.integrator_state)
    int_dict['acceptance_ratio'] = state.acceptance_ratio[0]
    int_dict['step_size'] = state.acceptance_ratio[1]
    int_dict['kinetic_energy'] = state.acceptance_ratio[2]
    int_dict['potential'] = state.potential
    return int_dict

  return init, update, get
