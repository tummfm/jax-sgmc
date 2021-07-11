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
from functools import partial

from typing import AnyStr, Callable, Any, Tuple, Iterable, Dict, Union

import numpy as onp

from jax import random, tree_unflatten, tree_flatten, grad, lax

import jax.numpy as jnp

from jax_sgmc.adaption import AdaptionStrategy, covariance
from jax_sgmc.potential import StochasticPotential
from jax_sgmc.data import RandomBatch, CacheState
from jax_sgmc.util import Array, tree_scale, tree_add, tree_multiply, tree_matmul, tree_dot
from jax_sgmc.scheduler import schedule

leapfrog_state = namedtuple("leapfrog_state",
                            ["positions",
                             "momentum",
                             "potential",
                             "model_state",
                             "data_state",
                             "key"])

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

def reversible_leapfrog(
          potential_fn: StochasticPotential,
          batch_fn: RandomBatch,
          steps: Array = 10
          ) -> Tuple[Callable, Callable, Callable]:
  """Initializes a reversible leapfrog integrator.

  AMAGOLD requires a reversible leapfrog integrator with half step at the
  beginning and end.

  Args:
    potential_fn: Likelihood and prior applied over a minibatch of data
    batch_fn: Function to draw a mini-batch of reference data
    steps: Number of intermediate leapfrog steps

  Returns:
    Returns a function running the time reversible leapfrog integrator for T
    steps.

  """

  init_data, get_data = batch_fn
  stochastic_gradient = grad(potential_fn, has_aux=True)

  def _position_update(scale, cov, position, momentum):
    if isinstance(cov, covariance):
      if cov.ndim == 1:
        scaled_momentum = tree_multiply(cov.inv_cov, momentum)
      elif cov.ndim == 2:
        scaled_momentum = tree_matmul(cov.inv_cov, momentum)
      else:
        raise ValueError(f"Covariance cannot have dimension greater than 2, "
                         f"dim is {cov.ndim}")
    else:
      scaled_momentum = tree_scale(1 / cov, momentum)
    # Scale is 0.5 of step size for half momentum update, otherwise it is just
    # the step size.
    scaled_momentum = tree_scale(scale, scaled_momentum)
    return tree_add(position, scaled_momentum)

  def _cov_scaled_noise(split, cov, tree):
    noise = random_tree(split, tree)
    if isinstance(cov, covariance):
      if cov.ndim == 1:
        noise = tree_multiply(covariance.cov_sqrt, noise)
      elif cov.ndim == 2:
        noise = tree_matmul(covariance.cov_sqrt, noise)
      else:
        raise ValueError(f"Covariance cannot have dimension greater than 2, "
                         f"dim is {cov.ndim}")
    else:
      noise = tree_scale(jnp.sqrt(cov), noise)
    return noise

  def _body_fun(state: leapfrog_state,
                step: jnp.array,
                parameters: schedule = None,
                cov: Union[Array, covariance] = jnp.array(1.0)):
    # Full step not required in first iteration because of the half step at the
    # beginning
    positions = lax.cond(step == 0,
                         lambda pos: pos,
                         lambda pos: _position_update(
                           parameters.step_size,
                           cov,
                           pos,
                           state.momentum),
                         state.positions)

    key, split = random.split(state.key)
    noise = _cov_scaled_noise(split, cov, state.momentum)
    scaled_noise = tree_scale(
      jnp.sqrt(4 * parameters.friction * parameters.step_size),
      noise)

    data_state, mini_batch = get_data(state.data_state, information=True)
    gradient, model_state = stochastic_gradient(
      positions,
      mini_batch,
      state=state.model_state
    )

    decayed_momentum = tree_scale(
      1 - parameters.step_size * parameters.friction,
      state.momentum
    )
    negative_scaled_gradient = tree_scale(
      -1.0 * parameters.step_size,
      gradient
    )
    unscaled_momentum = tree_add(
      tree_add(decayed_momentum, negative_scaled_gradient),
      scaled_noise
    )
    updated_momentum = tree_scale(
      1 / (1 + parameters.step_size * parameters.friction),
      unscaled_momentum
    )

    # Accumulate the energy
    momentum_sum = tree_add(state.momentum, updated_momentum)
    if isinstance(cov, covariance):
      if cov.ndim == 1:
        scaled_gradient = tree_multiply(covariance.inv_cov, gradient)
      elif cov.ndim == 2:
        scaled_gradient = tree_matmul(covariance.inv_cov, gradient)
      else:
        raise ValueError(f"Covariance cannot have dimension greater than 2, "
                         f"dim is {cov.ndim}")
    else:
      scaled_gradient = tree_scale(1 / cov, gradient)
    unscaled_energy = tree_dot(momentum_sum, scaled_gradient)
    acc_pot = state.potential + 0.5 * parameters.step_size * unscaled_energy

    new_state = leapfrog_state(
      potential=acc_pot,
      key=key,
      positions=positions,
      momentum=updated_momentum,
      data_state=data_state,
      model_state=model_state
    )

    return new_state, None

  def init_fn(init_sample: PyTree,
              key: Array = None,
              batch_kwargs: Dict = None,
              cov: Union[Array, covariance] = jnp.array(1.0),
              init_model_state: PyTree = None):
    """Initializes the initial state of the integrator.

    Args:
      init_sample: Initial latent variables
      key: Initial PRNGKey
      batch_kwargs: Determine the inital state of the random data chain
      cov: Initial covariance.
      init_model_state: State of the model.

    Returns:
      Returns the initial state of the integrator.

    """

    # Initializing the initial state here makes it easier to add additional
    # variables which might be only necessary in special case
    if batch_kwargs is None:
      batch_kwargs = {}

    reference_data_state = init_data(**batch_kwargs)

    # Sample initial momentum
    if key is None:
      key = random.PRNGKey(0)
    key, split = random.split(key)
    momentum = _cov_scaled_noise(split, cov, init_sample)

    init_state = leapfrog_state(
      potential=jnp.array(0.0),
      key=key,
      positions=init_sample,
      momentum=momentum,
      data_state=reference_data_state,
      model_state=init_model_state
    )

    return init_state

  def integrate(state: leapfrog_state,
                parameters: schedule,
                cov: Union[Array, covariance] = jnp.array(1.0),
                direction: Array = jnp.array(1.0)) -> leapfrog_state:

    # # Resample momentum to make process reversible (otherwise skew-reversible)
    key, split = random.split(state.key)
    momentum = _cov_scaled_noise(split, cov, state.momentum)

    # Change direction if last step has been rejected
    #vmomentum = tree_scale(direction, state.momentum)

    # Half step for leapfrog integration
    positions = _position_update(
      0.5 * parameters.step_size, cov, state.positions, momentum)

    # Do the leapfrog steps
    state = leapfrog_state(
      positions=positions,
      momentum=momentum,
      key=key,
      potential=jnp.array(0.0),
      model_state=state.model_state,
      data_state=state.data_state)

    # Leapfrog integration
    state, _ = lax.scan(
      partial(_body_fun, parameters=parameters, cov=cov),
      state,
      onp.arange(steps))

    # Final half step
    positions = _position_update(
      0.5 * parameters.step_size, cov, state.positions, state.momentum)

    final_state = leapfrog_state(
      positions=positions,
      momentum=state.momentum,
      key=state.key,
      potential=state.potential,
      model_state=state.model_state,
      data_state=state.data_state)

    return final_state

  def get_fn(state: leapfrog_state):
    """Returns the latent variables."""
    # Todo: This is not truly the likelihood
    return {"variables": state.positions,
            "energy": state.potential}


  return init_fn, integrate, get_fn


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
