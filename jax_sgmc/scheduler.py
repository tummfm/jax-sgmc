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

"""Schedules parameters of the integrator and solver.

The scheduler organizes the independent variables of the update equation, such
as the temperature and the step size, which are organized by multiple specific
schedulers.

"""

# Todo: Correct typing

from collections import namedtuple
from typing import Callable, Tuple

import jax.numpy as jnp
from jax import lax
from jax import random
from jax.experimental import host_callback as hcb

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
                       "accept"])
"""Auxiliary variables for integrator.

Attributes:
  step_size: Learning rate
  temperature: Scaling the magnitude of the additional noise
  burn_in: Bool, whether current step can be accepted
  accept: Bool, whether current sample should be saved

"""

scheduler_state = namedtuple("scheduler_state",
                             ["state",
                              "step_size_state",
                              "temperature_state",
                              "burn_in_state",
                              "thinning_state",
                              "progress_bar_state"])
"""Collects the states of the specific schedulers.

Attributes:
  state: State of the base scheduler, e.g. to keep track of current iteration
  step_size_state: State of the step size scheduler
  temperature_state: State of the temperature scheduler
  burn_in_state: State of the burn in scheduler
  thinning_state: State of thinning
  progress_bar_state: State of the progress bar
"""

static_information = namedtuple("static_information",
                                ["samples_collected"])
"""Information which is constant during the run.

Attributes:
  samples_collected: Number of samples saved.
"""

# The scheduler combines the specific scheduler. This makes it easier to
# implement only rarely used auxiliary variables by providing default values.
# The update functions are collected at a central state.

def init_scheduler(step_size: specific_scheduler = None,
                   temperature: specific_scheduler = None,
                   burn_in: specific_scheduler = None,
                   thinning: specific_scheduler = None,
                   progress_bar: bool = True,
                   progress_bar_steps: Array = 20
                   ) -> Tuple[Callable, Callable, Callable]:
  """Initializes the scheduler.

  The scheduler combines the specific schedules for each variable. It updates
  them and gets them at a central place and makes it possible to combine them or
  provide default values.

  Args:
    step_size: Triplet from step-size scheduler initialization
    temperature: Triplet from temperature scheduler initialization
    burn_in: Triplet from burn-in scheduler initialization
    thinning: Triplet from thinning scheduler initialization
    progress_bar: Show the percentage of completed steps

  Returns:
    Returns a triplet of ``(init_fn, update_fn, get_fn)``.
  """

  # Define the default values
  if step_size is None:
    step_size = polynomial_step_size(a=1, b=1, gamma=0.0)
  if temperature is None:
    temperature = constant_temperature(tau=1.0)
  if burn_in is None:
    burn_in = initial_burn_in(n=0)
  if thinning is None:
    # Accept all samples, save all samples
    thinning = specific_scheduler(
      lambda iterations: (None, iterations),
      lambda *args, **kwargs: None,
      lambda *args, **kwargs: True)

  if progress_bar:
    init_progress_bar, update_progress_bar = _progress_bar(burn_in, thinning)

  def init_fn(iterations: int,
              **scheduler_kwargs
              ) -> Tuple[scheduler_state, static_information]:

    # Initialize all the specific schedulers
    state = (0, iterations) # Start with iteration 0
    thinning_state, total_samples = thinning.init(
      iterations,
      **scheduler_kwargs.get('thinning', {}))
    burn_in_state, collected_samples = burn_in.init(
      iterations,
      **scheduler_kwargs.get('burn_in', {}))
    # If not thinning is provided, collect all samples not subject to burn in
    total_samples = min(total_samples, collected_samples)

    if progress_bar:
      pg_steps = scheduler_kwargs.get("progress_bar_steps", progress_bar_steps)
      pg_enabled = scheduler_kwargs.get("enabled", jnp.array(progress_bar))

      progress_bar_state = init_progress_bar(
        jnp.array(iterations), total_samples, pg_steps, pg_enabled)
    else:
      progress_bar_state = None

    init_state = scheduler_state(
      state=state,
      step_size_state=step_size.init(
        iterations,
        **scheduler_kwargs.get('step_size', {})),
      temperature_state=temperature.init(
        iterations,
        **scheduler_kwargs.get('temperature', {})),
      burn_in_state=burn_in_state,
      thinning_state=thinning_state,
      progress_bar_state=progress_bar_state)
    static = static_information(
      samples_collected=total_samples)
    return init_state, static

  def update_fn(state: scheduler_state, **kwargs) -> scheduler_state:
    # Keep track of current iteration
    iteration, total_iterations = state.state
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

    if progress_bar:
      # The burn in and thinning state are required to count the number of
      # collected samples
      progress_bar_state = update_progress_bar(
        state.progress_bar_state,
        iteration,
        burn_in_state,
        thinning_state,
        **kwargs)
    else:
      progress_bar_state = None

    new_scheduler_state = (current_iteration, total_iterations)
    updated_scheduler_state = scheduler_state(
      state=new_scheduler_state,
      step_size_state=step_size_state,
      temperature_state=temperature_state,
      burn_in_state=burn_in_state,
      thinning_state=thinning_state,
      progress_bar_state=progress_bar_state)
    return updated_scheduler_state

  def get_fn(state: scheduler_state, **kwargs) -> schedule:
    iteration, total_iterations = state.state
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
    )
    return current_schedule

  return init_fn, update_fn, get_fn

################################################################################
#
# Temperature
#
################################################################################

def constant_temperature(tau: Array = 1.0) -> specific_scheduler:
  """Scales the added noise with an unchanged constant.

  Args:
    tau: Scale of the added noise

  Returns:
    Returns a triplet as described above.

  """

  def init_fn(iterations: int,
              tau: Array = tau
              ) -> Array:
    del iterations
    return tau

  def update_fn(state: Array,
                iteration: int,
                **kwargs
                ) -> Array:
    del iteration, kwargs
    return state

  def get_fn(state: Array,
             iteration: int,
             **kwargs
             ) -> Array:
    del iteration, kwargs
    return state

  return specific_scheduler(init_fn, update_fn, get_fn)


def cyclic_temperature(beta: Array=1.0, k: int=1) -> specific_scheduler:
  """Cyclic switch of the temperature between 0.0 and 1.0.

  Switches temperature form 0.0 (SGD) to 1.0 (SGLD) when ratio of initial step
  size and current step size drops below beta. This scheduler is intended to
  be used with the cyclic step size scheduler.

  Args:
    beta: Ratio of current step size to initial step size when transition to SGLD
    k: Number of cycles

  Returns:
    Returns a triplet as described above
  """
  raise NotImplementedError

################################################################################
#
# Progress bar
#
################################################################################

def _progress_bar(burn_in: specific_scheduler,
                  thinning: specific_scheduler):
  """Prints the progress of the solver.

  Args:
    burn_in: Burn in scheduler to count accepted samples
    thinning: Thinning scheduler to count accepted samples

  """

  def _print_fn(info, _):
    percentage = round(
      int(info['current_iteration']) / int(info['total_iterations']) * 100)
    total_samples = int(info["total_samples"])
    collected_samples = int(info["collected_samples"])
    current_iteration = int(info["current_iteration"])
    total_iterations = int(info["total_iterations"])

    print(f"[Step {current_iteration}/{total_iterations}]"
          f"({percentage}%) Collected {collected_samples} of "
          f"{total_samples} samples...")

  def init_fn(iterations: Array,
              num_samples: Array,
              steps: Array = jnp.array(20),
              enabled: Array = jnp.array(True)
              ) -> Tuple[Array, Array, Array, Array, Array]:
    # Set already collected samples to zero
    init_state = iterations, num_samples, jnp.zeros(1), steps, enabled
    return init_state

  def step_fn(state: Tuple[Array, Array, Array, Array, Array],
              iteration: Array,
              burn_in_state,
              thinning_state,
              **kwargs
              ):
    iterations, tot_samples, collected_samples, steps, enabled = state

    # A sample is going to be saved if it is not subject to burn in and accepted
    sample_burn_in = burn_in.get(burn_in_state, iteration, **kwargs)
    sample_accepted = thinning.get(thinning_state, iteration, **kwargs)
    saved = sample_burn_in * sample_accepted
    collected_samples += saved

    info = {
      "total_iterations": iterations,
      "current_iteration": iteration,
      "total_samples": tot_samples,
      "collected_samples": collected_samples,
      "kwargs": kwargs
    }

    # Calculate number of steps until the progress should be printed out
    num_its = jnp.int_(jnp.floor(iterations / steps))

    # Return the number of collected samples as result of id_tap
    collected_samples = lax.cond(
      jnp.logical_and(jnp.mod(iteration, num_its) == 0, enabled),
      lambda arg: hcb.id_tap(_print_fn, arg, result=collected_samples),
      lambda arg: info["collected_samples"],
      info
    )

    new_state = iterations, tot_samples, collected_samples, steps, enabled
    return new_state

  return init_fn, step_fn

################################################################################
#
# Step Size
#
################################################################################

def adaptive_step_size(burn_in = 0,
                       initial_step_size = 0.05,
                       stabilization_constant = 100,
                       decay_constant = 0.75,
                       speed_constant = 0.05,
                       target_acceptance_rate=0.02):
  """Dual averaging scheme to tune step size for schemes with MH-step.

  The adaptive step size uses the dual averaging scheme to optimize the
  acceptance rate, as proposed by [1].

  [1] https://arxiv.org/abs/1111.4246

  Args:
    burn_in: Initial iterations, in which the step size should be tuned
    initial_step_size: Initial value of the step size
    speed_constant: Bigger constant stabilizes adaption against initial
      iterations
    decay_constant: Controls decay of learning rate of the step size
    speed_constant: Weights acceptance ratio statistics
    target_acceptance_rate: Desired acceptance rate

  Returns:
    Returns a specific step size scheduler.

  """

  def init(iterations: int,
           burn_in=burn_in,
           initial_step_size=initial_step_size,
           stabilization_constant=stabilization_constant,
           decay_constant=decay_constant,
           speed_constant=speed_constant,
           target_acceptance_rate=target_acceptance_rate):
    del iterations

    x_bar = jnp.log(initial_step_size)
    h_bar = 0.0
    init_state = (
      burn_in, x_bar, h_bar, target_acceptance_rate, stabilization_constant,
      decay_constant, speed_constant, jnp.log(10 * initial_step_size))
    return init_state

  def update(state: Array, iteration: int, acceptance_ratio=0.0, **kwargs):
    del kwargs
    burn_in, x_bar, h_bar, alpha, t0, kappa, gamma, mu = state
    m = iteration + 1

    h_bar *= (1 - 1/(m + t0))
    h_bar += 1/(m + t0) * (target_acceptance_rate - acceptance_ratio)

    x = mu - jnp.sqrt(m) / gamma * h_bar

    lr = jnp.power(m, -kappa)

    x_bar_old = x_bar
    x_bar *= (1 - lr)
    x_bar += lr * x

    # Only update during burn in
    x_bar = jnp.where(iteration < burn_in, x_bar, x_bar_old)

    return burn_in, x_bar, h_bar, alpha, t0, kappa, gamma, mu

  def get(state: Array, iteration: int, **kwargs):
    del iteration, kwargs
    return jnp.exp(state[1])

  return specific_scheduler(init, update, get)


def polynomial_step_size(a: Array = 1.0,
                         b: Array = 1.0,
                         gamma: Array = 0.33
                         ) -> specific_scheduler:
  """Polynomial decreasing step size schedule.

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

  def init_fn(iterations: int,
              a: Array = a,
              b: Array = b,
              gamma: Array = gamma
              ) -> Array:
    assert gamma >= 0, f"Gamma must be positive: gamma = {gamma}"
    assert a > 0, f"a must be positive: a = {a}"
    assert b > 0, f"b must be greater than zero: b = {b}"

    n = jnp.arange(iterations)
    unscaled = jnp.power(b + n, -gamma)
    scaled = jnp.multiply(a, unscaled)
    return scaled

  def update_fn(state: Array, iteration: int, **kwargs) -> Array:
    del iteration, kwargs
    return state

  def get_fn(state: Array, iteration: int, **kwargs) -> Array:
    del kwargs
    return state[iteration]

  return specific_scheduler(init_fn, update_fn, get_fn)


def polynomial_step_size_first_last(first: [float, Array] = 1.0,
                                    last: [float, Array] = 1.0,
                                    gamma: [float, Array] = 0.33
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

  # Calculates the required coefficients of the polynomial
  def find_ab(its, gamma, first, last):
    ginv = jnp.power(gamma, -1.0)
    fpow = jnp.power(first, -ginv) # pylint: disable=E1130
    lpow = jnp.power(last, -ginv) # pylint: disable=E1130
    apow = jnp.divide(lpow - fpow, its - 1)
    a = jnp.power(apow, -gamma)
    b = jnp.power(jnp.divide(first, a), -ginv) # pylint: disable=E1130
    return a, b

  def init_fn(iterations: int,
              first: [float, Array] = first,
              last: [float, Array] = last,
              gamma: [float, Array] = gamma
              ) -> Array:
    # Check for valid parameters
    assert gamma > 0, f"Gamma must be bigger than 0, is {gamma}"
    assert first >= last, f"The first step size must be larger than the last:" \
                         f" {first} !>= {last}"
    a, b = find_ab(iterations, gamma, first, last)
    init_fn, _, _ = polynomial_step_size(a=a, b=b, gamma=gamma)
    return init_fn(iterations)

  def update_fn(state: Array, iteration: int, **kwargs) -> Array:
    del iteration, kwargs
    return state

  def get_fn(state: Array, iteration: int, **kwargs) -> Array:
    del kwargs
    return state[iteration]

  return specific_scheduler(init_fn, update_fn, get_fn)

################################################################################
#
# Burn In
#
################################################################################

# Burn in: Return 1.0 if the sample should be accepted and 0.0 otherwise

def cyclic_burn_in(beta: Array=1.0, k:int=1):
  """Discards samples at the beginning of each cycle.

  Args:
    beta: Ratio of current and initial step size up to which burn in should be
      applied
    k: Number of cycles

  Returns:
    Returns a burn in schedule, which applies burn in to the beginning of each
    cycle.

  """
  raise NotImplementedError

def initial_burn_in(n: Array = 0) -> specific_scheduler:
  """Discards the first n steps.

  Args:
    n: Count of initial steps which should be discarded

  Returns:
    Returns specific scheduler.

  """

  def init_fn(iterations: int, n: Array = n) -> Tuple[Array, Array]:
    return n, iterations - n

  def update_fn(state: Array, iteration: int, **kwargs) -> Array:
    del iteration, kwargs
    return state

  def get_fn(state: Array, iteration: int, **kwargs) -> Array:
    del kwargs
    return jnp.where(state <= iteration, 1.0, 0.0)

  return specific_scheduler(init_fn, update_fn, get_fn)

################################################################################
#
# Thinning
#
################################################################################

# Thinning provides information about the number of samples which will be saved.

def random_thinning(step_size_schedule: specific_scheduler,
                    burn_in_schedule: specific_scheduler,
                    selections: int,
                    key: Array = None
                    ) -> specific_scheduler:
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

  def init_fn(iterations: int,
              step_size_schedule: specific_scheduler = step_size_schedule,
              burn_in_schedule: specific_scheduler = burn_in_schedule,
              selections: int = selections,
              key: Array = key
              ) -> Tuple[Array, Array]:
    if key is None:
      key = random.PRNGKey(0)

    step_size_state = step_size_schedule.init(iterations)
    burn_in_state, _ = burn_in_schedule.init(iterations)

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
    assert jnp.count_nonzero(probs) >= selections, "Cannot select enough values"

    # Draw the iterations which should be accepted
    accepted_its = random.choice(key,
                                 jnp.arange(iterations),
                                 shape=(selections,),
                                 replace=False,
                                 p=probs)
    return accepted_its, selections

  def update_fn(state: Array, iteration: int, **kwargs) -> Array:
    del iteration, kwargs
    return state

  def get_fn(state: Array, iteration: int, **kwargs) -> jnp.bool_:
    del kwargs
    accepted = jnp.where(jnp.any(iteration == state), True, False)
    return accepted

  return specific_scheduler(init_fn, update_fn, get_fn)
