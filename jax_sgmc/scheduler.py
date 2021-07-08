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
                                ["init",
                                 "update",
                                 "get"])
"""Bundles the specific scheduler as described above.

Attributes:
  init_fn: Function to initialize the specific scheduler
  update_fn: Calculate the next state of the scheduler
  get_fn: Get the scheduled parameter

"""

schedule = namedtuple("schedule",
                      ["step_size",
                       "temperature",
                       "burn_in",
                       "accept",
                       "friction"])
"""Auxillary variables for integrator.

Attributes:
  step_size: Learning rate
  temperature: Scaling the magnitude of the additional noise
  burn_in: Bool, whether current step can be accepted
  accept: Bool, whether current sample should be saved
  friction: Friction for SGHMC

"""

scheduler_state = namedtuple("scheduler_state",
                             ["state",
                              "step_size_state",
                              "temperature_state",
                              "burn_in_state",
                              "thinning_state"])
"""Collects the states of the specific schedulers.

Attributes:
  self: Own variables, such as current iteration
  step_size_state: State of the step size scheduler
  temperature_state: State of the temperature scheduler
  burn_in_state: State of the burn in scheduler
  thinnin_state: State of thinning

"""

static_information = namedtuple("static_information",
                                ["samples_collected"])
"""Information which is constant during the run.

Attributes:
  samples_collected: Number of samples saved.
"""

# Todo: Change scheduler to allow passing kwargs to init function, which do not
#       change the static properties.

# The scheduler combines the specific scheduler. This makes it easier to
# implement only rarely used auxillary variables by providing default values.
# The update functions are collected at a central state.

def init_scheduler(step_size: specific_scheduler = None,
                   temperature: specific_scheduler = None,
                   burn_in: specific_scheduler = None,
                   thinning: specific_scheduler = None,
                   friction: jnp.array = 0.25
                   ) -> Tuple[Callable, Callable, Callable]:
  """Initialize the scheduler.

  The scheduler combines the specific schedules for each variable. It updates
  them and gets them at a central place and makes it possible to combine them or
  provide default values.

  Args:
    step_size: Triplet from step-size scheduler initialization
    temperature: Triplet from temperature scheduler initialization
    burn_in: Triplet from burn-in scheduler initialization
    thinning: Triplet from thinning scheduler initialization

  Returns:
    Returns a triplet of ``(init_fn, update_fn, get_fn)``.
  """

  # Define the default values
  if step_size is None:
    step_size = (polynomial_step_size(1, 1, 0.0))
  if temperature is None:
    temperature = constant_temperature(1.0)
  if burn_in is None:
    burn_in = initial_burn_in(0)
  if thinning is None:
    # Accept all samples, save all samples
    thinning = specific_scheduler(
      lambda iterations: (None, iterations),
      lambda *args, **kwargs: None,
      lambda *args, **kwargs: True)

  def init_fn(iterations: int) -> Tuple[scheduler_state, static_information]:
    # Initialize all the specific schedulers
    state = (0,) # Start with iteration 0
    thinning_state, total_samples = thinning.init(iterations)
    init_state = scheduler_state(
      state=state,
      step_size_state=step_size.init(iterations),
      temperature_state=temperature.init(iterations),
      burn_in_state=burn_in.init(iterations),
      thinning_state=thinning_state
    )
    static = static_information(
      samples_collected=total_samples
    )
    return init_state, static

  def update_fn(state: scheduler_state, **kwargs) -> scheduler_state:
    # Keep track of current iteration
    iteration, = state.state
    current_iteration = iteration + 1
    # Update the states
    step_size_state = step_size.update(state.step_size_state,
                                       iteration,
                                       **kwargs)
    temperature_state = temperature.update(state.temperature_state,
                                           iteration,
                                           **kwargs)
    burn_in_state = burn_in.update(state.burn_in_state,
                                   iteration,
                                   **kwargs)
    thinning_state = thinning.update(state.thinning_state,
                                     iteration,
                                     **kwargs)
    state = (current_iteration,)
    updated_scheduler_state = scheduler_state(
      state=state,
      step_size_state=step_size_state,
      temperature_state=temperature_state,
      burn_in_state=burn_in_state,
      thinning_state=thinning_state
    )
    return updated_scheduler_state

  def get_fn(state: scheduler_state, **kwargs) -> schedule:
    iteration, = state.state
    current_step_size = step_size.get(state.step_size_state,
                                      iteration,
                                      **kwargs)
    current_temperature = temperature.get(state.temperature_state,
                                          iteration,
                                          **kwargs)
    current_burn_in = burn_in.get(state.burn_in_state,
                                  iteration,
                                  **kwargs)
    current_thinning = thinning.get(state.thinning_state,
                                    iteration,
                                    **kwargs)
    current_schedule = schedule(
      step_size=jnp.array(current_step_size),
      temperature=jnp.array(current_temperature),
      burn_in=jnp.array(current_burn_in),
      accept=jnp.array(current_thinning),
      friction=friction
    )
    return current_schedule

  return init_fn, update_fn, get_fn


################################################################################
#
# Temperature
#
################################################################################

# Todo: Implement constant temperature scheduler

def constant_temperature(tau: Array=1.0) -> specific_scheduler:
  """Scale the added noise with an unchanged constant.

  Args:
    tau: Scale of the added noise

  Returns:
    Returns a triplet as described above.

  """

  def init_fn(unused_iterations:int) -> None:
    return None

  def update_fn(unused_state: None,
                unused_iteration: int,
                **unused_kwargs) -> None:
    return None

  def get_fn(unused_state: None,
             unused_iteration: int,
             **unused_kwargs
             ) -> Array:
    return tau

  return specific_scheduler(init_fn, update_fn, get_fn)


def cyclic_temperature(beta: Array=1.0, k: int=1) -> specific_scheduler:
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
  raise NotImplementedError

################################################################################
#
# Step Size
#
################################################################################

# Todo: Implement basic schedulers:

def polynomial_step_size(a: Array=1.0,
                         b: Array=1.0,
                         gamma:Array=0.33
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
  assert gamma >= 0, f"Gamma must be positive: gamma = {gamma}"
  assert a > 0, f"a must be positive: a = {a}"
  assert b > 0, f"b must be greater than zero: b = {b}"

  # The internal state is just an array, which holds the step size for all the
  # iterations

  def init_fn(iterations: int) -> Array:
    n = jnp.arange(iterations)
    unscaled = jnp.power(b + n, -gamma)
    scaled = jnp.multiply(a, unscaled)
    return scaled

  def update_fn(state: Array, unused_iteration: int, **unused_kwargs) -> Array:
    return state

  def get_fn(state: Array, iteration: int, **unused_kwargs) -> Array:
    del unused_kwargs
    return state[iteration]

  return specific_scheduler(init_fn, update_fn, get_fn)


def polynomial_step_size_first_last(first: Array=1.0,
                                    last: Array=1.0,
                                    gamma: Array=0.33
                                    ) -> specific_scheduler:
  """Initializes polynomial step size schedule via first and last step.

  Args:
    first: Step size in the first iteration
    last: Step size in the last iteration
    gamma: Rate of decay

  Returns:
    Returns a polynomial step size schedule defined via the first and the last
    step size.

  """
  # Check for valid parameters
  assert gamma > 0, f"Gamma must be bigger than 0, is {gamma}"
  assert first > last, f"The first step size must be larger than the last: "\
                       f"{first} !> {last}"

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

  def update_fn(state: Array, unused_iteration: int, **unused_kwargs) -> Array:
    return state

  def get_fn(state: Array, iteration: int, **unused_kwargs) -> Array:
    del unused_kwargs
    return state[iteration]

  return specific_scheduler(init_fn, update_fn, get_fn)


def cyclic_step_size(alpha: Array=1.0, k: int=1):
  """Step size cyclically decreasing from alpha.

  Implements a step size schedule folllowing:

  .. math::

     \\epsilon = \\frac{\\alpha}{2}\\left[ a \\right]

  """
  raise NotImplementedError

################################################################################
#
# Burn In
#
################################################################################

# Burn in: Return 1.0 if the sample should be accepted and 0.0 otherwise

def cyclic_burn_in(beta: Array=1.0, k:int=1):
  """Discard samples at the beginning of each cycle.

  Args:
    beta: Ratio of current and initial step size up to which burn in should be
      applied
    k: Number of cycles

  Returns:
    Returns a burn in schedule, which applies burn in to the beginning of each
    cycle.

  """
  raise NotImplementedError


def initial_burn_in(n: int=0):
  """Discard the first n steps.

  Args:
    n: Count of inital steps which should be discarded

  Returns:
    Returns specific scheduler.

  """

  def init_fn(unused_iterations: int) -> None:
    return None

  def update_fn(unused_state: None,
                unused_iteration: int,
                **unused_kwargs) -> None:
    return None

  def get_fn(unused_state: None, iteration: int, **unused_kwargs) -> Array:
    return jnp.where(n <= iteration, 1.0, 0.0)

  return specific_scheduler(init_fn, update_fn, get_fn)

################################################################################
#
# Thinning
#
################################################################################

# Thinning provides information about the number of samples which will be saved.

def random_thinning(step_size_schedule: specific_scheduler,
                    burn_in_schedule: specific_scheduler,
                    selections: Array,
                    key: Array = None):
  """Random thinning weighted by the step size.

  Randomly select samples not subject to burn in. The probability of selection
  is proportional to the step size to deal with the issue of the decaying step
  size. This only works for static step size and burn in schedules.

  Args:
    step_size_schedule: Static step size schedule
    burn_in_schedule: Static burn in schedule
    selections: Number of selected samples
    key: PRNGKey for drawing selections

  Returns:
    Returns a scheduler marking the accepted samples.

  """

  if key is None:
    key = random.PRNGKey(0)

  def init_fn(iterations: int) -> Tuple[Array, Array]:

    step_size_state = step_size_schedule.init(iterations)
    burn_in_state = burn_in_schedule.init(iterations)

    def update_fn(state, iteration):
      step_size_state, burn_in_state = state
      step_size = step_size_schedule.get(step_size_state, iteration)
      burn_in = burn_in_schedule.get(burn_in_state, iteration)
      probability = step_size * burn_in
      new_state = (step_size_schedule.update(step_size_state, iteration),
                   burn_in_schedule.update(burn_in_state, iteration))
      return new_state, probability

    _, probs = lax.scan(update_fn,
                        (step_size_state, burn_in_state),
                        jnp.arange(iterations))

    # Check that a sufficient number of elements can be drawn
    # assert jnp.count_nonzero(probs) >= selections, "Cannot select enough values"
    # Draw the iterations which should be accepted
    accepted_its = random.choice(key,
                                 jnp.arange(iterations),
                                 shape=(selections,),
                                 replace=False,
                                 p=probs)
    return accepted_its, selections

  def update_fn(state: Array, unused_iteration: int, **unused_kwargs) -> Array:
    return state

  def get_fn(state: Array, iteration: int, **unused_kwargs) -> jnp.bool_:
    accepted = jnp.where(jnp.any(iteration == state), True, False)
    return accepted

  return specific_scheduler(init_fn, update_fn, get_fn)
