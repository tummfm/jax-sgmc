"""Solvers for Stochastic Gradient Bayesian Inference."""

from typing import Callable, Any, Optional, Tuple

from jax import lax
import jax.numpy as jnp

PyTree = Any

from jax_sgmc import io
from jax_sgmc.util import host_callback


# Todo: Implement kernel
# Todo: Implement chain runner
# Todo: Implement saving module

def mcmc(solver,
         scheduler,
         strategy="map",
         init_strategy="sample",
         saving=None):
  """Extends the solver  .

  Args:
    solver: Computes the next state form a given state.
    scheduler: Schedules solver parameters such as temperature and burn in.


  """

  if saving is None:
    # Pass the state
    init_saving, save, postprocess_saving = io.no_save()

  scheduler_init, scheduler_next, scheduler_get = scheduler
  _, solver_update, solver_get = solver

  def _update(state, _):
    solver_state, scheduler_state = state
    schedule = scheduler_get(scheduler_state)
    new_state = solver_update(solver_state, schedule)
    saved = save(schedule, solver_get(new_state))
    scheduler_state = scheduler_next(scheduler_state)
    return (new_state, scheduler_state), saved

  def run(*states, iterations: int = 1e5):
    if len(states) == 1:
      # Initializes the scheduler and run for the number of iterations
      scheduler_state = scheduler_init(iterations)
      _, saved = lax.scan(_update,
                          (states[0], scheduler_state),
                          jnp.arange(iterations))
      return postprocess_saving(saved)
    else:
      raise NotImplementedError("Currently only a single chain can be "
                                "evaluated in parallel.")

  return run


def sgld(integrator) -> Tuple[Callable, Callable, Callable]:
  """Initializes the standard SGLD - sampler.

  This sampler simply integrates without acceptance/rejection or parallel
  tempering.
  """

  init_integrator, update_integrator, get_integrator = integrator

  def init(*args, **kwargs):
    return init_integrator(*args, **kwargs)

  def update(state, schedule):
    return update_integrator(state, schedule)

  def get(state):
    return get_integrator(state)

  return init, update, get
