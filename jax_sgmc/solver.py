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

from typing import Callable, Any, Tuple

from jax import lax, jit
import jax.numpy as jnp
from jax_sgmc import io

PyTree = Any

# Todo: Implement kernel
# Todo: Implement chain runner
# Todo: Implement saving module

def mcmc(solver,
         scheduler,
         strategy="map",
         saving=None):
  """Runs the solver for multiple chains and saves the collected samples

  Args:
    solver: Computes the next state form a given state.
    scheduler: Schedules solver parameters such as temperature and burn in.
    strategy: Run multiple chains in parallel or vectorized
    saving: Save samples via host_callback (if saving requires much memory)

  Returns:
    Returns function which runs the solver for a given number of iterations.

  """

  if saving is None:
    # Pass the state
    init_saving, save, postprocess_saving = io.no_save()

  scheduler_init, scheduler_next, scheduler_get = scheduler
  _, solver_update, solver_get = solver

  @jit
  def _update(state, _):
    solver_state, scheduler_state = state
    schedule = scheduler_get(scheduler_state)
    new_state, stats = solver_update(solver_state, schedule)
    saved = save(schedule, solver_get(new_state))
    # Update the scheduler with additional information from the integrator, such
    # as the acceptance ratio of AMAGOLD
    if stats is not None:
      scheduler_state = scheduler_next(scheduler_state, **stats)
    else:
      scheduler_state = scheduler_next(scheduler_state)
    return (new_state, scheduler_state), saved

  # Scan over the update function. Postprocessing the samples is required if
  # saving is None to sort out not accepted samples. Samples are not accepted
  # due tu burn in or thinning.
  def run(*states, iterations: int = 1e5):
    if strategy == "map":
      def run_single():
        for state in states:
          init_saving()
          scheduler_state = scheduler_init(iterations)
          _, saved = lax.scan(_update,
                              (state, scheduler_state),
                              jnp.arange(iterations))
          yield postprocess_saving(saved)
      chains = list(run_single())
    else:
      raise NotImplementedError("Parallel execution is currently not "
                                "supported")
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
