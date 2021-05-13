# Todo: Implement a class to organize step_size and tempering

from collections import namedtuple

from typing import Iterable

import jax.numpy as jnp

from jax_sgmc.util import Array

update_parameters = namedtuple("update_parameters",
                               ["step_size",
                                "temperature"])
"""State of auxillary variables for integrator.

Attributes:
  step_size: Learning rate
  temperature: Scaling the magnitude of the additional noise

"""

class Scheduler:
  """Schedules the step size and temperature."""

  def get_schedule(self, step_count=1, strict=False) -> Iterable:
    pass

# Todo: Implement a handler to variably set the step count, e. g. by using
#       information about the step size

class StaticScheduler(Scheduler):
  """Schedules the pre-run determineted step size and temperature.

  Attributes:
    step_size: Learning rate of the update steps
    temperature: Scale of the added noise
    current_index: Index to iterate over step-size and temperature array

  """
  def __init__(self, step_size: Array, temperature: Array):
    self.step_size = step_size
    self.temperature = temperature

    self.current_index = 0

  def get_schedule(self, step_count=1, strict=False) -> Iterable:
    """Returns a schedule for a number of steps.

    Arguments:
      step_count: Number of steps in the iterator
      strict: Throw an exception when all steps are already stepped. Otherwise,
        infinitely repeat the last step size and temperature.

    """

    start_index = self.current_index
    end_index = self.current_index + step_count
    selection_indices = jnp.arange(start_index, end_index)

    # If not strict, simply repeat the last index to reselect the last step size
    # a multiple of times

    if strict:
      assert end_index <= self.step_size.size
    else:
      selection_indices = jnp.where(selection_indices < self.step_size.size,
                                    selection_indices,
                                    self.step_size.size -1)

    selected_step_sizes = self.step_size[selection_indices,]
    selected_temperatures = self.temperature[selection_indices,]

    for step_size, temperature in zip(selected_step_sizes,
                                     selected_temperatures):

      yield update_parameters(step_size=step_size,
                              temperature=temperature)

