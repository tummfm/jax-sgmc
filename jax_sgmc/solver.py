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
from typing import Callable, Any, Tuple

from jax import lax, jit
import jax.numpy as jnp
from jax_sgmc import io, util

PyTree = Any

# Todo: Implement kernel
# Todo: Implement chain runner
# Todo: Implement saving module

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

  # Todo: Remove saved
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
        raise NotImplementedError("Very inefficient.")
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
