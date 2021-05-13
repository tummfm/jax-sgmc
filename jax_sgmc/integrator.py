"""Defines integrators which form the core of the solvers."""

from collections import namedtuple

from typing import AnyStr, Callable, Any, Tuple, Iterable, List

from jax import random, jit, tree_unflatten, tree_flatten

import jax.numpy as jnp

from jax_sgmc.data import ReferenceData, mini_batch
from jax_sgmc.util import Array, tree_scale, tree_add, tree_multiply
from jax_sgmc.scheduler import update_parameters

leapfrog_state = namedtuple("leapfrog_state",
                            ["positions", "momentum", "potential"])

langevin_state = namedtuple("langevin_state",
                            ["latent_variables",
                             "key",
                             "adapt_state"])
"""State of the langevin diffusion integrator.

Attributes:
  latent_variables: Current latent variables
  key: PRNGKey
  adapt_state: Containing quantities such as momentum for adaption

"""

PyTree = Any
LangevinIntegrator = Callable[[langevin_state, Iterable], langevin_state]

def random_tree(key, a):
  """Build a tree shaped like a where all nodes are normal distributed.

  Arguments:
    key: PRNGKey
    a: PyTree defining the shape of the output

  Returns:
    Tree shaped like a with normal distributed leaves.

  """
  leaves, tree_def = tree_flatten(a)
  splits = random.split(key, len(leaves))
  noise_leaves = [random.normal(split, leave.shape)
                  for split, leave in zip(splits, leaves)]
  noise_tree = tree_unflatten(tree_def, noise_leaves)
  return noise_tree

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

def langevin_diffusion(
        init_sample: PyTree,
        data: ReferenceData,
        stochastic_gradient: Callable[[PyTree, mini_batch], PyTree],
        key: Array=None,
        adaption=None,
        strategy='map'
        ) -> Tuple[langevin_state, LangevinIntegrator]:
  """Initializes langevin diffusion integrator.

  Arguments:
    key: PRNGKey
    init_sample: Latent variables to start with
    data: Reference Data object
    stochastic_gradient: Stochastic gradient function
    adaption: A function to adapt the state and get the the parameters M, Gamma
      and D

  Returns:
    Returns a tuple consisting of the integrator function and the initial state.

  """

  # We need to define an update function. The update function acts on the
  # flattened version of the pytree, as operations such as elementwise product,
  # are currently not supported. This is probably going to change with the
  # introduction of the tree vectorizing transformation
  # --> https://github.com/google/jax/pull/3263

  @jit
  def update_fn(state: langevin_state,
                parameters: update_parameters,
                reference_data: mini_batch):
    key, split = random.split(state.key)

    noise = random_tree(split, state.latent_variables)
    gradient = stochastic_gradient(state.latent_variables,
                                   reference_data)

    if adaption is None:
      scaled_gradient = tree_scale(- parameters.step_size, gradient)
      print(scaled_gradient)
      scaled_noise = tree_scale(jnp.sqrt(2 * parameters.temperature
                                * parameters.step_size),
                                noise)
      update_step = tree_add(scaled_gradient, scaled_noise)
    else:
      # Adapt the state
      new_adapt_state, manifold, drift = adaption(state.adapt_state,
                                                  stochastic_gradient=gradient,
                                                  sample=state.latent_variables)

      # Todo: Implement the adaption update step with tree_map
      # Todo: Think of a way to support matrix-vector and vector-vector product
      #       on pytrees.

      update_step = jnp.zeros_like(state.latent_variables)

    new_sample = tree_add(state.latent_variables, update_step)
    new_state = langevin_state(key=key,
                               latent_variables=new_sample,
                               adapt_state=None)

    return new_state

  # if strategy == 'map':
  #   def update_fn(state: List[langevin_state],
  #                 parameters: update_parameters,
  #                 refernce_data: List[mini_batch]):
  #     pass

  # The integrate function iterates over a schedule, which provides information
  # such as step size and temperature

  # Todo: Change to support e. g. adaptive step_size?
  # Todo: Support parallel chain evaluation

  def integrate(state: List[langevin_state], schedule: Iterable) -> langevin_state:
    # Todo: Here it could be possible to evaluate multiple chains at once. The
    #       overal method is not jit-able, so we cannot move out the
    #       parallelization / vectorization on a higher level.

    for step in schedule:
      mini_batch = data.get_random_batch()

      print(step)
      state = update_fn(state, step, mini_batch)

    return state

  # Initializing the initial state here makes it easier to add additional
  # variables which might be only necessary in special cases

  if key is None:
    key = random.PRNGKey(0)

  init_state = langevin_state(key=key, latent_variables=init_sample, adapt_state=None)

  return init_state, integrate
