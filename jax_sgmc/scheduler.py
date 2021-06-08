"""Handles the independent variables of the update equqation.

The scheduler organizes the independent variables of the update equation, such
as the temperature and the step size.
A scheduler must be combined from specific schedulers, only considering a single
variable.

Initializing a specific scheduler returns the triple of functions:

::

  (init_fn, update_fn, get_fn) = init_specific_scheduler(*args, **kwargs)

The complete scheduler can then be initialized by passing the triples of the
specific schedulers.

::

  (init_fn, update_fn, get_fn) = init_scheduler(step_size=step_size_triple,
                                                temperature=temp_triple)

Uninitialized specific schedulers simply return a default value when not passed
to the scheduler.

"""

# Todo: Correct typing

from collections import namedtuple

from typing import Callable, Tuple

import jax.numpy as jnp
from jax import lax
from jax import random

from jax_sgmc.util import Array

specific_scheduler = namedtuple("specific_scheduler",
                                ["init_fn",
                                 "update_fn",
                                 "get_fn"])
"""Bundles the specific scheduler as described above.

Attributes:
  init_fn: Function to initialize the specific scheduler
  update_fn: Calculate the next state of the scheduler
  get_fn: Get the scheduled parameter

"""

schedule = namedtuple("schedule",
                      ["step_size",
                       "temperature",
                       "burn_in"])
"""Auxillary variables for integrator.

Attributes:
  step_size: Learning rate
  temperature: Scaling the magnitude of the additional noise
  burn_in: Bool, whether current step can be accepted

"""

scheduler_state = namedtuple("scheduler_state",
                             ["state",
                              "step_size_state",
                              "temperature_state",
                              "burn_in_state"])
"""Collects the states of the specific schedulers.

Attributes:
  self: Own variables, such as current iteration
  step_size_state: State of the step size scheduler
  temperature_state: State of the temperature scheduler
  burn_in_state: State of the burn in scheduler

"""

# Todo: Where to place burn in?
#       ++ Burn in has dependency on step size
#       -- Difficult to manage for thinning

# The scheduler combines the specific scheduler. This makes it easier to
# implement only rarely used auxillary variables by providing default values.
# The update functions are collected at a central state.

def init_scheduler(step_size: specific_scheduler = None,
                   temperature: specific_scheduler = None,
                   burn_in: specific_scheduler = None
                   ) -> Tuple[Callable, Callable, Callable]:
  """Initialize the scheduler.

  The scheduler combines the specific schedules for each variable. It updates
  them and gets them at a central place and makes it possible to combine them or
  provide default values.

  Args:
    step_size: Triplet from step-size scheduler initialization
    temperature: Triplet from temperature scheduler initialization
    burn_in: Triplet from burn-in scheduler initialization

  Returns:
    Returns a triplet of ``(init_fn, update_fn, get_fn)``.
  """

  if step_size is None:
    step_size = polynomial_step_size(1, 0, 0.0)
  if temperature is None:
    temperature = constant_temperature(1.0)
  if burn_in is None:
    burn_in = initial_burn_in(0)

  init_step_size, update_step_size, get_step_size = step_size
  init_temp, update_temp, get_temp = temperature
  init_burn_in, update_burn_in, get_burn_in = burn_in

  def init_fn(iterations: int) -> scheduler_state:
    step_size_state = init_step_size(iterations)

  # Todo: Complete

  def update_fn(state: scheduler_state, **kwargs) -> scheduler_state:
    iteration, = scheduler_state.state # pylint: disable=E0633
    current_iteration = iteration + 1

    current_step_size = update_step_size(iteration)

    new_self = (current_iteration,)

  def get_fn(state: scheduler_state) -> schedule:
    pass

  return init_fn, update_fn, get_fn


################################################################################
#
# Temperature scheduling
#
################################################################################

# Todo: Implement constant temperature scheduler

def constant_temperature(tau: jnp.float32=1.0) -> specific_scheduler:
  """Scale the added noise with an unchanged constant.

  Args:
    tau: Scale of the added noise

  Returns:
    Returns a triplet as described above.

  """

  def init_fn(iterations:int) -> None:
    return None

  def update_fn(state: None, **unused_kwargs) -> None:
    return None

  def get_fn(state: None, iteration: int, **unused_kwargs) -> Array:
    return tau

  return specific_scheduler(init_fn, update_fn, get_fn)


def cyclic_temperature(beta: jnp.float32=1.0, k: int=1) -> specific_scheduler:
  """Cylic switch of the temperature between 0.0 and 1.0.

  Switches temperature form 0.0 (SGD) to 1.0 (SGLD) when ratio of initial step
  size and current step size drops below beta. This scheduler is intended to
  be used with the cyclic step size scheduler.

  Args:
    beta: Ratio of current step size to inital step size when transition to SGLD
    k: Number of cycles

  Returns:
    Returns a triplet as described above
  """
  pass

################################################################################
#
# Temperature scheduling
#
################################################################################

# Todo: Implement basic schedulers:

def polynomial_step_size(a: jnp.float32=1.0,
                         b:jnp.float32=1.0,
                         gamma:jnp.float32=0.33
                         ) -> specific_scheduler:
  """Polynomial descresing step size schedule.

  Implements the original proposal of a polynomial step size schedule
  :math:`\epsilon = a(b + n)^{\gamma}`.

  Args:
    a: Scale of all step sizes
    b: Stabilization constant
    gamma: Decay constant

  Returns:
    Returns triplet as described above.

  """

  # The internal state is just an array, which holds the step size for all the
  # iterations

  def init_fn(iterations: int) -> Array:
    n = jnp.arange(iterations)
    unscaled = jnp.power(b + n, -gamma)
    scaled = jnp.multiply(a, unscaled)
    return scaled

  def update_fn(state: Array) -> Array:
    return state

  def get_fn(state: Array, iteration:int, **unused_kwargs) -> Array:
    del unused_kwargs
    return state[iteration]

  return init_fn, update_fn, get_fn


def polynomial_step_size_first_last(first: jnp.float32=1.0,
                                    last: jnp.float32=1.0,
                                    gamma: jnp.float32=0.33
                                    ) -> specific_scheduler:
  """Initializes polynomial step size schedule via first and last step."""

  # Calculates the required coefficients of the polynomial
  def find_ab(its):
    ginv = jnp.power(gamma, -1.0)
    fpow = jnp.power(first, -ginv) # pylint: disable=E1130
    lpow = jnp.power(last, -ginv) # pylint: disable=E1130
    apow = jnp.divide(lpow - fpow, its - 1)
    a = jnp.power(apow, -gamma)
    b = jnp.power(jnp.divide(first, a), -ginv) # pylint: disable=E1130
    return a, b

  def init_fn(iterations: int) -> Array:
    a, b = find_ab(iterations)
    init_fn, _, _ = polynomial_step_size(a, b, gamma)
    return init_fn(iterations)

  def update_fn(state: Array) -> Array:
    return state

  def get_fn(state: Array, iteration:int, **unused_kwargs) -> Array:
    del unused_kwargs
    return state[iteration]

  return specific_scheduler(init_fn, update_fn, get_fn)


def cyclic_step_size(alpha: jnp.float32=1.0, k: int=1):
  """Step size cyclically decreasing from alpha.

  Implements a step size schedule folllowing:

  .. math::

     \\epsilon = \\frac{\\alpha}{2}\\left[ a \\right]

  """
  raise NotImplementedError

################################################################################
#
# Implement burn in scheduling
#
################################################################################

# Burn in: Return 1.0 if the sample should be accepted and 0.0 otherwise

def cyclic_burn_in(beta: jnp.float32=1.0, k:int=1):
  raise NotImplementedError


def initial_burn_in(n: int=0):
  """Discard the first n steps."""

  def init_fn(iterations: int) -> None:
    return None

  def update_fn(state: None, **unused_kwargs) -> None:
    return None

  def get_fn(state: None, iteration: int, **unused_kwargs) -> Array:
    return jnp.where(n < iteration, 0.0, 1.0)

  return specific_scheduler(init_fn, update_fn, get_fn)

################################################################################
#
# Implement thinning
#
################################################################################

def random_thinning(step_size_schedule: specific_scheduler,
                    burn_in_schedule: specific_scheduler):
  """Randomly select n elements with probability weighted by the step size."""

  def init_fn(iterations: int,
              selections: int,
              key=random.PRNGKey(0)) -> Array:
    step_size_state = step_size_schedule.init_fn(iterations)
    burn_in_state = burn_in_schedule.init_fn(iterations)

    def update_fn(state, _):
      step_size_state, burn_in_state = state
      step_size = step_size_schedule.get_fn(step_size_state)
      burn_in = burn_in_schedule.get_fn(burn_in_state)
      probability = step_size * burn_in
      new_state = (step_size_schedule.update_fn(step_size_state),
                   burn_in_schedule.update_fn(burn_in_state))
      return new_state, probability

    _, probs = lax.scan(update_fn, (step_size_state, burn_in_state), iterations)

    # Draw the iterations which should be accepted
    accepted_its = random.choice(key,
                                 jnp.arange(iterations),
                                 shape=(selections,),
                                 replace=False,
                                 p=probs)
    return accepted_its

  def update_fn(state: Array, **kwargs) -> Array:
    return state

  def get_fn(state: Array, iteration: int, **unused_kwargs) -> jnp.bool_:
    accepted = jnp.where(jnp.any(iteration == state), True, False)
    return accepted

  return specific_scheduler(init_fn, update_fn, get_fn)
