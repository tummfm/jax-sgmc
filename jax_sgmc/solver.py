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
from typing import Callable, Any, Tuple, NamedTuple, Dict

from jax import lax, jit, random
import jax.numpy as jnp
from jax_sgmc import io, util, data, potential, integrator
from jax_sgmc.util import host_callback
from jax_sgmc.adaption import covariance

PyTree = Any
Array = util.Array

class AMAGOLDState(NamedTuple):
  """State of the AMAGOLD solver.

  Attributes:
    full_data_state: Cache state for full data mapping function.
    potential: True potential of the current sample
    integrator_state: State of the reversible leapfrog integrator
    key: PRNGKey for MH correction step
    direction: Integrate with positive (1.0) or negative (-1.0) momentum

  """
  full_data_state: data.CacheState
  potential: Array
  integrator_state: integrator.leapfrog_state
  key: Array
  direction: Array

class SGGMCState(NamedTuple):
  """State of the AMAGOLD solver.

  Attributes:
    full_data_state: Cache state for full data mapping function.
    potential: True potential of the current sample
    integrator_state: State of the reversible leapfrog integrator
    key: PRNGKey for MH correction step
    acceptance_ratio: Acceptance ratio used in last step.

  """
  full_data_state: data.CacheState
  potential: Array
  integrator_state: integrator.obabo_state
  key: Array
  acceptance_ratio: Array

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
    solver_state, scheduler_state, saving_state = state
    schedule = scheduler_get(scheduler_state)
    new_state, stats = solver_update(solver_state, schedule)
    # Kepp the sample if it is not subject to burn in or thinning.
    keep = jnp.logical_and(schedule.burn_in, schedule.accept)
    saving_state, saved = save(
      saving_state,
      keep,
      solver_get(new_state),
      scheduler_state=scheduler_state,
      solver_state=solver_state)
    # Update the scheduler with additional information from the integrator, such
    # as the acceptance ratio of AMAGOLD
    if stats is not None:
      scheduler_state = scheduler_next(scheduler_state, **stats)
    else:
      scheduler_state = scheduler_next(scheduler_state)
    return (new_state, scheduler_state, saving_state), saved

  # Scan over the update function. Postprocessing the samples is required if
  # saving is None to sort out not accepted samples. Samples are not accepted
  # due tu burn in or thinning.

  # Intialize a single chain
  def _init(init_state, iterations):
    scheduler_state, static_information = scheduler_init(iterations)
    if loading is None:
      # Define what the saving module is expected to save
      saving_state = init_saving(
        solver_get(init_state),
        (init_state, scheduler_state),
        static_information)
    else:
      raise NotImplementedError("Loading of checkpoints is currently not "
                                "supported.")
    # Return tuple of static and non-static information
    return (iterations, static_information), (init_state, scheduler_state, saving_state)

  @partial(jit, static_argnums=0)
  def _run(static, init_state):
    iterations, static_information = static
    _update = partial(_uninitialized_update, static_information)
    (_, _, saving_state), saved = lax.scan(
      _update,
      init_state,
      jnp.arange(iterations))
    return saving_state, saved

  def run(*states, iterations: int = 1e5):
    # The same schedule for all chains.
    if strategy == "map":
      def run_single():
        for state in states:
          init_args = _init(state, iterations)
          results = _run(*init_args)
          yield postprocess_saving(*results)
      chains = list(run_single())
    else:
      # Initialize the states sequentially
      static_info, init_states = zip(*[_init(state, iterations)
                                       for state in states])
      # Static information is the same for all chains
      if strategy == "pmap":
        mapped_run = util.list_pmap(partial(_run, static_info[0]))
      elif strategy == "vmap":
        # Todo: Fix unnecessary vmap calls. This could be another application of
        #       the stop_vmap primitive
        # Vmap of cond is transformed to lax.select. Thus, true branch and false
        # branch is run, such that all samples are saved and for every sample an
        # index update is performed.
        # raise NotImplementedError("Very inefficient.")
        print("Run vmapped")
        mapped_run = jit(util.list_vmap(partial(_run, static_info[0])))
      else:
        raise NotImplementedError(f"Strategy {strategy} is unknown. ")
      saving_states, saved_values = zip(*mapped_run(*init_states))
      return list(map(postprocess_saving, saving_states, saved_values))

    # No need to return a list for a single chain result
    if len(chains) == 1:
      return chains.pop()
    else:
      return chains
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

  def update(state, schedule):
    # No statistics such as acceptance ratio
    return update_integrator(state, schedule), None

  def get(state):
    return get_integrator(state)

  return init, update, get


def amagold(integrator_fn,
            full_potential_fn) -> Tuple[Callable, Callable, Callable]:
  """Initializes AMAGOLD integration.

  Args:
    integrator_fn: Reversible leapfrog integrator.
    full_potential_fn: Function to calculate true potential.

  Returns:
    Returns the AMAGOLD solver, a combination of reversible stochastic
    hamiltonian dynamics and amortized MH corrections steps.

  """

  init_integrator, update_integrator, get_integrator = integrator_fn

  def init(init_sample: PyTree,
           full_data_state: Any,
           key: Array = random.PRNGKey(0),
           **kwargs) -> AMAGOLDState:

    potential, (full_data_state, _) = full_potential_fn(init_sample,
                                                        full_data_state)
    key, split = random.split(key)
    # Todo: Init with correct covariance
    integrator_state = init_integrator(init_sample, key=key, cov=jnp.array(1.0), **kwargs)

    state = AMAGOLDState(
      integrator_state=integrator_state,
      potential=potential,
      full_data_state=full_data_state,
      key=split,
      direction=jnp.array(1.0)
    )
    return state

  def update(state: AMAGOLDState, schedule):
    key, split = random.split(state.key, 2)
    proposal = update_integrator(
      state.integrator_state,
      schedule,
      direction=state.direction,
      cov=jnp.array(1.0))

    # MH correction step
    new_potential, (full_data_state, _) = full_potential_fn(
      proposal.positions,
      state.full_data_state)

    alpha = jnp.exp(state.potential - new_potential + proposal.potential)
    log_alpha = state.potential - new_potential + proposal.potential
    slice = random.uniform(split)

    host_callback.id_print(alpha, what="Acceptance ratio")
    host_callback.id_print(new_potential, what="New potential")
    host_callback.id_print(state.potential, what="Old potential")
    host_callback.id_print(proposal.potential, what="Energy acc")
    host_callback.id_print(jnp.log(slice) < log_alpha, what="Accepted")
    host_callback.id_print(slice, what="slice")

    # If true, the step is accepted
    mh_integrator_state, new_potential, direction = lax.cond(
      jnp.log(slice) < log_alpha,
      lambda new_old: new_old[0],
      lambda new_old: new_old[1],
      ((proposal, new_potential, 1.0),
       (state.integrator_state, state.potential, -1.0)))

    new_integrator_state = integrator.leapfrog_state(
      positions=mh_integrator_state.positions,
      momentum=mh_integrator_state.momentum,
      model_state=mh_integrator_state.model_state,
      potential=mh_integrator_state.potential,
      # These parameters must be provided from the updated state, otherwise
      # the noise and the random data is not resampled
      data_state=proposal.data_state,
      key=proposal.key
    )

    host_callback.id_print(direction, what="Direction")

    new_state = AMAGOLDState(
      key=key,
      integrator_state=new_integrator_state,
      potential=new_potential,
      full_data_state=full_data_state,
      direction=direction
    )

    stats = {'acceptance_ratio': alpha}

    return new_state, stats

  def get(state):
    return get_integrator(state.integrator_state)

  return init, update, get

def sggmc(integrator_fn,
          full_potential_fn) -> Tuple[Callable, Callable, Callable]:
  """Guided Gradient Monte Carlo using Stochastic Gradients.

  The OBABO integration scheme is reversible even when using stochastic
  gradients and provides second order accuracy. Therefore, a MH-acceptance step
  can be applied to sample from the correct posterior distribution.

  [1] https://arxiv.org/abs/2102.01691

  Args:
    integrator_fn: Reversible leapfrog integrator.
    full_potential_fn: Function to calculate true potential.

  Returns:
    Returns the SGGMC solver.

  """

  init_integrator, update_integrator, get_integrator = integrator_fn

  def init(init_sample: PyTree,
           full_data_state: Any,
           key: Array = random.PRNGKey(0),
           **kwargs) -> SGGMCState:

    potential, (full_data_state, _) = full_potential_fn(init_sample,
                                                        full_data_state)
    key, split = random.split(key)
    integrator_state = init_integrator(init_sample, key=key, **kwargs)

    state = SGGMCState(
      integrator_state=integrator_state,
      potential=potential,
      full_data_state=full_data_state,
      key=split,
      acceptance_ratio=jnp.array(0.0)
    )
    return state

  def update(state: SGGMCState, schedule) -> Tuple[SGGMCState, Dict]:
    # Todo: Change cov
    cov = jnp.array(1.0)

    key, split = random.split(state.key, 2)
    proposal = update_integrator(
      state.integrator_state,
      schedule,
      cov=cov)

    # MH correction step
    new_potential, (full_data_state, _) = full_potential_fn(
      proposal.positions,
      state.full_data_state)

    log_alpha = - 1.0 / schedule.temperature * (
      new_potential - state.potential + proposal.kinetic_energy_end - proposal.kinetic_energy_start)
    # Limit the probability to 1.0 to ensure correct calculation of acceptance
    # ratio statistics.
    log_alpha = jnp.where(log_alpha <= 0, log_alpha, 0.0)

    slice = random.uniform(split)

    # If true, the step is accepted
    mh_integrator_state, new_potential = lax.cond(
      jnp.log(slice) < log_alpha,
      lambda new_old: new_old[0],
      lambda new_old: new_old[1],
      ((proposal, new_potential),
       (state.integrator_state, state.potential)))

    new_integrator_state = integrator.obabo_state(
      positions=mh_integrator_state.positions,
      momentum=mh_integrator_state.momentum,
      model_state=mh_integrator_state.model_state,
      potential=mh_integrator_state.potential,
      # These parameters must be provided from the updated state, otherwise
      # the noise and the random data is not resampled
      data_state=proposal.data_state,
      key=proposal.key,
      kinetic_energy_start=0.0,
      kinetic_energy_end=0.0
    )

    new_state = SGGMCState(
      key=key,
      integrator_state=new_integrator_state,
      potential=new_potential,
      full_data_state=full_data_state,
      acceptance_ratio=jnp.exp(log_alpha)
    )

    stats = {'acceptance_ratio': jnp.exp(log_alpha)}

    return new_state, stats

  def get(state: SGGMCState) -> Dict:
    int_dict = get_integrator(state.integrator_state)
    int_dict['acceptance_ratio'] = state.acceptance_ratio
    return int_dict

  return init, update, get
