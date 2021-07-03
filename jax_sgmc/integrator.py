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

"""Defines integrators which form the core of the solvers."""

from collections import namedtuple

from typing import AnyStr, Callable, Any, Tuple, Iterable, Dict

from jax import random, tree_unflatten, tree_flatten, grad

import jax.numpy as jnp

from jax_sgmc.adaption import AdaptionStrategy
from jax_sgmc.potential import StochasticPotential
from jax_sgmc.data import RandomBatch, CacheState
from jax_sgmc.util import Array, tree_scale, tree_add, tree_multiply, tree_matmul
from jax_sgmc.scheduler import schedule

leapfrog_state = namedtuple("leapfrog_state",
                            ["positions", "momentum", "potential"])

langevin_state = namedtuple("langevin_state",
                            ["latent_variables",
                             "model_state",
                             "key",
                             "adapt_state",
                             "data_state",
                             "potential"])
"""State of the langevin diffusion integrator.

Attributes:
  latent_variables: Current latent variables
  key: PRNGKey
  adapt_state: Containing quantities such as momentum for adaption
  data_state: State of the reference data cache
  model_state: Variables not considered during inference
  potential: Stochastic potential from last evaluation
"""

# Todo: Correct typing:
#       - ReferenceData class is replaced by batch_fn()
#       - MiniBatchState either in integrator state or seperate

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
  noise_leaves = [random.normal(split, leaf.shape)
                  for split, leaf in zip(splits, leaves)]
  noise_tree = tree_unflatten(tree_def, noise_leaves)
  return noise_tree

# T as static or dynamic parameter?
# Low level like this or directly as a class?

def reversible_leapfrog(key: Array,
                        T: Array,
                        data: CacheState,
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
                      data: CacheState,
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

  # Todo: Find a general method to deal with direct hessian inversion (e. g.
  #   fisher scoring) or iterative inversion. -> Two different solvers
  #   For langevin diffusion is might be possible to implement an adaption step.

def langevin_diffusion(
        potential_fn: StochasticPotential,
        batch_fn: RandomBatch,
        adaption: AdaptionStrategy = None,
) -> Tuple[Callable, Callable, Callable]:
  """Initializes langevin diffusion integrator.

  Arguments:
    potential_fn: Likelihood and prior applied over a minibatch of data
    batch_fn: Function to draw a mini-batch of reference data
    adaption: Adaption of manifold for faster inference

  Returns:
    Returns a tuple consisting of ``ini_fn``, ``update_fn``, ``get_fn``.
    The init_fn takes the arguments

    - key: Initial PRNGKey
    - adaption_kwargs: Additional arguments to determine the initial manifold
      state
    - batch_kwargs: Determine the state of the random data chain

  """
  if adaption is not None:
    adapt_init, adapt_update, adapt_get = adaption
  batch_init, batch_get = batch_fn
  stochastic_gradient = grad(potential_fn, has_aux=True)

  # We need to define an update function. All array oprations must be
  # implemented via tree_map. This is probably going to change with the
  # introduction of the tree vectorizing transformation
  # --> https://github.com/google/jax/pull/3263

  # This function is intended to generate initial states. Jax key,
  # adaption, etc. can be initialized to a default value if not explicitely
  # provided

  def init_fn(init_sample: PyTree,
              key: Array = random.PRNGKey(0),
              adaption_kwargs: Dict = None,
              batch_kwargs: Dict = None,
              init_model_state: PyTree = None):
    """Initializes the initial state of the integrator.

    Args:
      init_sample: Initial latent variables
      key: Initial PRNGKey
      adaption_kwargs: Determine the inital state of the adaption
      batch_kwargs: Determine the inital state of the random data chain

    Returns:
      Returns the initial state of the integrator.

    """

    # Initializing the initial state here makes it easier to add additional
    # variables which might be only necessary in special case
    if adaption_kwargs is None:
      adaption_kwargs = {}
    if batch_kwargs is None:
      batch_kwargs = {}

    # Adaption is not required in the most general case
    if adaption is None:
      adaption_state = None
    else:
      adaption_state = adapt_init(init_sample, **adaption_kwargs)

    reference_data_state = batch_init(**batch_kwargs)

    init_state = langevin_state(
      key=key,
      latent_variables=init_sample,
      adapt_state=adaption_state,
      data_state=reference_data_state,
      model_state=init_model_state,
      potential=jnp.array(0.0)
    )

    return init_state

  # Returns the important parameters of a state and excludes. Makes information
  # hiding possible

  def get_fn(state: langevin_state):
    """Returns the latent variables."""
    return {"variables": state.latent_variables,
            "likelihood": -state.potential}

  # Update according to the integrator update rule

  def update_fn(state: langevin_state,
                parameters: schedule):
    """Updates the integrator state according to a schedule.

    Args:
      state: Integrator state
      parameters: Schedule containig step_size and temperature

    Returns:
      Returns a new step calculated by applying langevin diffusion.
    """
    key, split = random.split(state.key)
    data_state, mini_batch = batch_get(state.data_state, information=True)

    noise = random_tree(split, state.latent_variables)
    gradient, new_model_state = stochastic_gradient(
      state.latent_variables,
      mini_batch,
      state=state.model_state)
    potential, _ = potential_fn(
      state.latent_variables,
      mini_batch,
      state=state.model_state
    )

    scaled_gradient = tree_scale(-parameters.step_size, gradient)
    scaled_noise = tree_scale(
      jnp.sqrt(2 * parameters.temperature * parameters.step_size), noise)

    if adaption is None:
      update_step = tree_add(scaled_gradient, scaled_noise)
      adapt_state = None
    else:
      # Update the adaption
      adapt_state = adapt_update(
        state.adapt_state,
        state.latent_variables,
        gradient,
        mini_batch)
      # Get the adaption
      manifold = adapt_get(
        adapt_state,
        state.latent_variables,
        gradient,
        mini_batch)

      if manifold.ndim == 1:
        adapted_gradient = tree_multiply(manifold.g_inv, scaled_gradient)
        adapted_noise = tree_multiply(manifold.sqrt_g_inv, scaled_noise)
      else:
        adapted_gradient = tree_matmul(manifold.g_inv, scaled_gradient)
        adapted_noise = tree_matmul(manifold.sqrt_g_inv, scaled_noise)

      scaled_gamma = tree_scale(parameters.temperature, manifold.gamma)
      update_step = tree_add(
        tree_add(scaled_gamma, adapted_gradient),
        adapted_noise
      )

    # Conclude the variable update by adding the step to the current samples
    new_sample = tree_add(state.latent_variables, update_step)
    new_state = langevin_state(
      key=key,
      latent_variables=new_sample,
      adapt_state=adapt_state,
      data_state=data_state,
      model_state=new_model_state,
      potential=jnp.array(potential, dtype=state.potential.dtype)
    )

    return new_state

  return init_fn, update_fn, get_fn
