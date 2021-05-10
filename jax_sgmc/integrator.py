"""Defines integrators which form the core of the solvers."""

from collections import namedtuple

from typing import AnyStr, Callable

from jax_sgmc.data import ReferenceData
from jax_sgmc.util import Array

leapfrog_state = namedtuple("leapfrog_state",
                            ["positions", "momentum", "potential"])

# T as static or dynamic parameter?
# Low level like this or directly as a class?

def reversible_leapfrog(key: Array,
                        T: Array,
                        data: ReferenceData,
                        potential_strategy: AnyStr='map'
                        ) -> Callable[[leapfrog_state], leapfrog_state]:
  """Initializes a reversible leapfrog integrator.

  AMAGOLD requires a reversible leapfrog integrator with half step at the
  beginning and end.

  Args:
    key: PRNGKey to generate noise
    T: Number of leapfrog steps until acceptance step is run
    data: Reference data object to draw minibatches while integrating.
    initial_state: Usually the last state of the previous integration run.
    potential_strategy: See potential module

  Returns:
    Returns a function running the time reversible leapfrog integrator for T
    steps.

  """

  # We need to initialize the potential module to evaluate the potential for a
  # minibatch of data.

  # Should there be an init function and should we pass back initial states or
  # are they obvious?

  def integrate(leapfrog_state: leapfrog_state) -> leapfrog_state:

    # Here it could be possible to evaluate multiple chains at once. The overal
    # method is not jit-able, so we cannot move out the parallelization /
    # vectorization on a higher level.

    pass


  return integrate


def friction_leapfrog(key: Array,
                      T: Array,
                      data: ReferenceData,
                      potential_strategy: AnyStr='map'
                      ) -> Callable[[leapfrog_state], leapfrog_state]:
  """Initializes the original SGHMC leapfrog integrator.

  Args:
    key: PRNGKey to generate noise
    T: Number of leapfrog steps until acceptance step is run
    data: Reference data object to draw minibatches while integrating.
    initial_state: Usually the last state of the previous integration run.
    potential_strategy: See potential module

  Returns:
    Returns a function running the non conservative leapfrog integrator for T
    steps.

  """

  # We need to initialize the potential module to evaluate the potential for a
  # minibatch of data.

  # Should there be an init function and should we pass back initial states or
  # are they obvious?

  def integrate(leapfrog_state: leapfrog_state) -> leapfrog_state:
    # Here it could be possible to evaluate multiple chains at once. The overal
    # method is not jit-able, so we cannot move out the parallelization /
    # vectorization on a higher level.

    pass

  return integrate

  # Todo: Implement Langevin diffusin integrator
  # Todo: Find a general method to deal with direct hessian inversion (e. g.
  # fisher scoring) or iterative inversion. -> Two different solvers
  # For langevin diffusion is might be possible to implement an adaption step.

def langevin_diffusion(key: Array,
                       T: Array,
                       data: ReferenceData,
                       potential_strategy: AnyStr='map'
                       ) -> Callable[[leapfrog_state], leapfrog_state]:
  """Initializes langevin diffusion integrator."""

  # We need to initialize the potential module to evaluate the potential for a
  # minibatch of data.

  # Should there be an init function and should we pass back initial states or
  # are they obvious?

  def integrate(leapfrog_state: leapfrog_state) -> leapfrog_state:
    # Here it could be possible to evaluate multiple chains at once. The overal
    # method is not jit-able, so we cannot move out the parallelization /
    # vectorization on a higher level.

    pass

  return integrate
