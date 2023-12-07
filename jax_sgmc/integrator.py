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

from functools import partial
from typing import Callable, Any, Tuple, Dict, NamedTuple

import jax.numpy as jnp
import numpy as onp
from jax import random, tree_unflatten, tree_flatten, grad, lax, tree_util, \
  value_and_grad, named_call

from jax_sgmc.adaption import AdaptionStrategy, MassMatrix
from jax_sgmc.data import RandomBatch, CacheState
from jax_sgmc.potential import StochasticPotential
from jax_sgmc.scheduler import schedule
from jax_sgmc.util import Array, tree_scale, tree_add, tensor_matmul, \
  tree_dot, Tensor, tree_multiply

PyTree = Any


class LeapfrogState(NamedTuple):
  """State of the reversible and friction leapfrog integrator.

  Attributes:
    positions: Latent variables in hamiltonian dynamics formulation
    momentum: Momentum with the same shape as the latent variables
    potential: Accumulated energy to calculate MH-correction step
    model_state: State of the model in the likelihood
    data_state: State of the random data function
    key: PRNGKey
  """
  positions: PyTree
  momentum: PyTree
  potential: Array
  model_state: PyTree
  data_state: CacheState
  key: Array
  extra_fields: PyTree = None


class ObaboState(NamedTuple):
  """State of the OBABO integrator.

  Attributes:
    positions: Latent variables in hamiltonian dynamics formulation
    momentum: Momentum with the same shape as the latent variables
    potential: Stochastic potential
    model_state: State of the model in the likelihood
    data_state: State of the random data function
    key: PRNGKey
    kinetic_energy_start: Kinetic energy after the first 1/4-step
    kinetic_energy_end: Kinetic energy after the last 3/4-step
  """
  positions: PyTree
  momentum: PyTree
  potential: Array
  model_state: PyTree
  data_state: CacheState
  key: Array
  kinetic_energy_start: Array
  kinetic_energy_end: Array


class LangevinState(NamedTuple):
  """State of the langevin diffusion integrator.

  Attributes:
    latent_variables: Current latent variables
    key: PRNGKey
    adapt_state: Containing quantities such as momentum for adaption
    data_state: State of the reference data cache
    model_state: Variables not considered during inference
    potential: Stochastic potential from last evaluation
    variance: Variance of stochastic potential over mini-batch
  """
  latent_variables: PyTree
  model_state: PyTree
  key: Array
  adapt_state: Any
  data_state: CacheState
  potential: Array
  variance: Array


def init_mass(mass) -> MassMatrix:
  """Initializes a diagonal mass tensor.

  Args:
    mass: Diagonal mass which has the same tree structure as the sample.

  Returns:
    Returns a diagonal mass matrix.
  """
  inv_mass = tree_util.tree_map(
    lambda x: jnp.power(x, -1.0),
    mass)
  sqrt_mass = tree_util.tree_map(
    jnp.sqrt,
    mass)
  inv_mass = Tensor(ndim=1, tensor=inv_mass)
  sqrt_mass = Tensor(ndim=1, tensor=sqrt_mass)
  return MassMatrix(inv=inv_mass, sqrt=sqrt_mass)


def random_tree(key, a):
  """Build a tree shaped like a where all nodes are normally distributed.

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


def obabo(potential_fn: StochasticPotential,
          batch_fn: RandomBatch,
          steps: Array = 10,
          friction: Array = 1.0,
          const_mass: PyTree = None,
          ) -> Tuple[Callable, Callable, Callable]:
  """Initializes the OBABO integration scheme.

  The OBABO integration scheme is reversible even when using stochastic
  gradients and provides second order accuracy.

  [1] https://arxiv.org/abs/2102.01691

  Args:
    potential_fn: Likelihood and prior applied over a minibatch of data
    batch_fn: Function to draw a mini-batch of reference data
    steps: Number of integration steps.
    friction: Controls impact of momentum from previous step
    const_mass: Mass matrix if no matrix is adapted. Must have the same tree
      structure as the sample

  Returns:
    Returns a function running the time OBABO integrator for T
    steps.

  """

  init_data, get_data, _ = batch_fn
  stochastic_gradient = value_and_grad(potential_fn, argnums=0, has_aux=True)

  # Calculate the inverse and the square root
  if const_mass:
    const_mass = init_mass(const_mass)
  else:
    const_mass = None

  # Helper functions to calculate the kinetic energy, update the position,
  # refresh and update the momentum.

  def _kinetic_energy(mass, momentum):
    scaled_momentum = tensor_matmul(mass.inv, momentum)
    return 0.5 * tree_dot(momentum, scaled_momentum)

  def _position_update(scale, mass, position, momentum):
    scaled_momentum = tensor_matmul(mass.inv, momentum)
    scaled_momentum = tree_scale(scale, scaled_momentum)
    return tree_add(position, scaled_momentum)

  # Half step if scale == 0.5 * step_size
  def _momentum_update(scale, gradient, momentum):
    scaled_gradient = tree_scale(-1.0 * scale, gradient)
    return tree_add(scaled_gradient, momentum)

  # Add noise to momentum
  def _momentum_resampling(parameters, mass, momentum, split):
    noise = random_tree(split, momentum)
    scaled_noise = tensor_matmul(mass.sqrt, noise)
    permanence = jnp.exp(-friction * parameters.step_size)
    momentum_noise = tree_scale(
      jnp.sqrt((1 - permanence) * parameters.temperature),
      scaled_noise)
    decayed_momentum = tree_scale(jnp.sqrt(permanence), momentum)
    return tree_add(decayed_momentum, momentum_noise)

  # A single OBABO-step of the integrator
  def _leapfrog_steps(state: ObaboState,
                      step: jnp.array,
                      mass: PyTree,
                      parameters: schedule = None):
    del step
    key, split1, split2 = random.split(state.key, num=3)

    refreshed_momentum = _momentum_resampling(
      parameters,
      mass,
      state.momentum,
      split1)

    # The kinetic energy from the first step is necessary to calculate the
    # acceptance probability
    # start_energy = lax.select(
    #   step == 0,
    #   _kinetic_energy(mass, refreshed_momentum),
    #   state.kinetic_energy_start)
    start_energy = state.kinetic_energy_start + _kinetic_energy(mass, refreshed_momentum)

    # Momentum update with stochastic gradient
    data_state, mini_batch = get_data(state.data_state, information=True)
    (pot_before, model_state), gradient = stochastic_gradient(
      state.positions,
      mini_batch,
      state=state.model_state)
    first_updated_momentum = _momentum_update(
      0.5 * parameters.step_size,
      gradient,
      refreshed_momentum)

    # Position update with momentum
    updated_positions = _position_update(
      parameters.step_size,
      mass,
      state.positions,
      first_updated_momentum)

    # Momentum update with stochastic gradient
    data_state, mini_batch = get_data(data_state, information=True)
    (pot_after, model_state), gradient = stochastic_gradient(
      updated_positions,
      mini_batch,
      state=model_state )
    second_updated_momentum = _momentum_update(
      0.5 * parameters.step_size,
      gradient,
      first_updated_momentum)

    final_refreshed_momentum = _momentum_resampling(
      parameters,
      mass,
      second_updated_momentum,
      split2)

    # The kinetic energy of the last step is necessary to
    # calculate the acceptance probability for the MH step.
    end_energy = state.kinetic_energy_end + _kinetic_energy(mass, second_updated_momentum)

    new_state = ObaboState(
      potential=0.5 * (pot_before + pot_after),
      positions=updated_positions,
      momentum=final_refreshed_momentum,
      key=key,
      data_state=data_state,
      model_state=model_state,
      kinetic_energy_start=start_energy,
      kinetic_energy_end=end_energy)

    return new_state, None

  def init_fn(init_sample: PyTree,
              key: Array = None,
              batch_kwargs: Dict = None,
              init_model_state: PyTree = None):
    """Initializes the initial state of the integrator.

    Args:
      init_sample: Initial latent variables
      key: Initial PRNGKey
      batch_kwargs: Determine the initial state of the random data chain
      init_model_state: State of the model.
      mass: Mass matrix

    Returns:
      Returns the initial state of the integrator.

    """

    # Initializing the initial state here makes it easier to add additional
    # variables which might be only necessary in special case
    if batch_kwargs is None:
      batch_kwargs = {}

    reference_data_state = init_data(**batch_kwargs)

    if key is None:
      key = random.PRNGKey(0)

    momentum = tree_util.tree_map(jnp.zeros_like, init_sample)

    init_state = ObaboState(
      kinetic_energy_start=jnp.array(0.0),
      kinetic_energy_end=jnp.array(0.0),
      potential=jnp.array(0.0),
      key=key,
      positions=init_sample,
      momentum=momentum,
      data_state=reference_data_state,
      model_state=init_model_state)

    return init_state

  @partial(named_call, name='obabo_integration')
  def integrate(state: ObaboState,
                parameters: schedule,
                mass: PyTree = None
                ) -> ObaboState:

    # If the mass is not adapted, take the constant mass if provided
    if const_mass is None:
      cms = init_mass(tree_util.tree_map(jnp.ones_like, state.positions))
    else:
      cms = const_mass

    if mass is None:
      mass = cms

    # Leapfrog integration
    state, _ = lax.scan(
      partial(_leapfrog_steps, parameters=parameters, mass=mass),
      state,
      onp.arange(steps))

    return state

  def get_fn(state: LeapfrogState) -> Dict[str, PyTree]:
    """Returns the latent variables."""
    return {"variables": state.positions,
            "energy": state.potential,
            "model_state": state.model_state}

  return init_fn, integrate, get_fn


def reversible_leapfrog(potential_fn: StochasticPotential,
                        batch_fn: RandomBatch,
                        steps: int = 10,
                        friction: [float, Array] = 0.25,
                        const_mass: PyTree = None
                        ) -> Tuple[Callable, Callable, Callable]:
  """Initializes a reversible leapfrog integrator.

  AMAGOLD requires a reversible leapfrog integrator with half step at the
  beginning and end.

  Args:
    potential_fn: Likelihood and prior applied over a minibatch of data
    batch_fn: Function to draw a mini-batch of reference data
    steps: Number of intermediate leapfrog steps
    friction: Decay of momentum to counteract induced noise due to stochastic
     gradients
    const_mass: Mass matrix to be used when no mass matrix is adapted

  Returns:
    Returns a function running the time reversible leapfrog integrator for T
    steps.

  """

  init_data, get_data, _ = batch_fn
  stochastic_gradient = grad(potential_fn, has_aux=True)

  # Calculate the inverse and the square root
  if const_mass:
    const_mass = init_mass(const_mass)
  else:
    const_mass = None

  def _position_update(scale, mass, position, momentum):
    scaled_momentum = tensor_matmul(mass.inv, momentum)
    # Scale is 0.5 of step size for half momentum update, otherwise it is just
    # the step size.
    scaled_momentum = tree_scale(scale, scaled_momentum)
    return tree_add(position, scaled_momentum)

  def _cov_scaled_noise(split, mass, tree):
    noise = random_tree(split, tree)
    noise = tensor_matmul(mass.sqrt, noise)
    return noise

  def _energy(old_momentum, new_momentum, mass, gradient, scale):
    # Accumulate the energy
    momentum_sum = tree_add(old_momentum, new_momentum)
    scaled_gradient = tensor_matmul(mass.inv, gradient)
    unscaled_energy = tree_dot(momentum_sum, scaled_gradient)
    return scale * unscaled_energy

  def _body_fun(state: LeapfrogState,
                step: jnp.array,
                parameters: schedule,
                mass: MassMatrix):
    # Full step not required in first iteration because of the half step at the
    # beginning
    positions = lax.cond(step == 0,
                         lambda pos: pos,
                         lambda pos: _position_update(
                           parameters.step_size,
                           mass,
                           pos,
                           state.momentum),
                         state.positions)

    key, split = random.split(state.key)
    noise = _cov_scaled_noise(split, mass, state.momentum)
    scaled_noise = tree_scale(
      jnp.sqrt(4 * friction * parameters.step_size),
      noise)

    data_state, mini_batch = get_data(state.data_state, information=True)
    gradient, model_state = stochastic_gradient(
      positions,
      mini_batch,
      state=state.model_state)

    decayed_momentum = tree_scale(
      1 - parameters.step_size * friction,
      state.momentum)
    negative_scaled_gradient = tree_scale(
      -1.0 * parameters.step_size,
      gradient)
    unscaled_momentum = tree_add(
      tree_add(decayed_momentum, negative_scaled_gradient),
      scaled_noise)
    updated_momentum = tree_scale(
      1 / (1 + parameters.step_size * friction),
      unscaled_momentum)

    energy = _energy(
      state.momentum,
      updated_momentum,
      mass,
      gradient,
      0.5 * parameters.step_size)
    accumulated_energy = energy + state.potential

    new_state = LeapfrogState(
      potential=accumulated_energy,
      key=key,
      positions=positions,
      momentum=updated_momentum,
      data_state=data_state,
      model_state=model_state)

    return new_state, None

  def init_fn(init_sample: PyTree,
              key: Array = None,
              batch_kwargs: Dict = None,
              init_model_state: PyTree = None,
              mass: PyTree = None):
    """Initializes the initial state of the integrator.

    Args:
      init_sample: Initial latent variables
      key: Initial PRNGKey
      batch_kwargs: Determine the initial state of the random data chain
      init_cov: Initial covariance.
      init_model_state: State of the model.

    Returns:
      Returns the initial state of the integrator.

    """

    # Initializing the initial state here makes it easier to add additional
    # variables which might be only necessary in special case
    if batch_kwargs is None:
      batch_kwargs = {}

    reference_data_state = init_data(**batch_kwargs)

    # Use constant mass if provided
    if not mass:
      if const_mass:
        mass = const_mass
      else:
        mass = init_mass(tree_util.tree_map(jnp.ones_like, init_sample))

    # Sample initial momentum
    if key is None:
      key = random.PRNGKey(0)
    key, split = random.split(key)
    momentum = _cov_scaled_noise(split, mass, init_sample)

    init_state = LeapfrogState(
      potential=jnp.array(0.0),
      key=key,
      positions=init_sample,
      momentum=momentum,
      data_state=reference_data_state,
      model_state=init_model_state)

    return init_state

  @partial(named_call, name='leapfrog_integration')
  def integrate(state: LeapfrogState,
                parameters: schedule,
                mass: PyTree = None
                ) -> LeapfrogState:

    # Use default values if mass matrix not provided
    if not mass:
      if const_mass:
        mass = const_mass
      else:
        mass = init_mass(tree_util.tree_map(jnp.ones_like, state.positions))

    # Half step for leapfrog integration
    positions = _position_update(
      0.5 * parameters.step_size, mass, state.positions, state.momentum)

    # Do the leapfrog steps
    state = LeapfrogState(
      positions=positions,
      momentum=state.momentum,
      key=state.key,
      potential=jnp.array(0.0),
      model_state=state.model_state,
      data_state=state.data_state)

    # Leapfrog integration
    state, _ = lax.scan(
      partial(_body_fun, parameters=parameters, mass=mass),
      state,
      onp.arange(steps))

    # Final half step
    positions = _position_update(
      0.5 * parameters.step_size, mass, state.positions, state.momentum)

    final_state = LeapfrogState(
      positions=positions,
      momentum=state.momentum,
      key=state.key,
      potential=state.potential,
      model_state=state.model_state,
      data_state=state.data_state)

    return final_state

  def get_fn(state: LeapfrogState) -> Dict[str, PyTree]:
    return {"variables": state.positions,
            "energy": state.potential,
            "model_state": state.model_state}

  return init_fn, integrate, get_fn


def friction_leapfrog(potential_fn: StochasticPotential,
                      batch_fn: RandomBatch,
                      steps: int = 10,
                      friction: Array = 0.25,
                      const_mass: PyTree = None,
                      noise_model = None
                      ) -> Tuple[Callable, Callable, Callable]:
  """Initializes the original SGHMC leapfrog integrator.

  Original SGHMC from [1].

  [1] https://arxiv.org/pdf/1402.4102.pdf

  Args:
    potential_fn: Likelihood and prior applied over a minibatch of data
    batch_fn: Function to draw a mini-batch of reference data
    steps: Number of intermediate leapfrog steps
    friction: Decay of momentum to counteract induced noise due to stochastic
     gradients
    const_mass: Mass matrix of the hamiltonian process
    noise_model: Stateless adaption of the noise (e. g. via the empirical fisher
      information)

  Returns:
    Returns a function running the non-conservative leapfrog integrator for T
    steps.

  """

  init_data, get_data, _ = batch_fn
  stochastic_gradient = value_and_grad(potential_fn, has_aux=True)
  if noise_model:
    init_noise_model, update_noise_model, get_noise_model = noise_model

  # Calculate the inverse and the square root
  if const_mass:
    const_mass = init_mass(const_mass)
  else:
    const_mass = None

  def _body_fun(state: LeapfrogState,
                step: jnp.array,
                parameters: schedule,
                friction: PyTree,
                mass: MassMatrix):
    del step
    # Update the position with the momentum
    scaled_momentum = tensor_matmul(mass.inv, state.momentum)
    position_update = tree_scale(parameters.step_size, scaled_momentum)
    new_positions = tree_add(state.positions, position_update)

    # Update the momentum in three steps
    # 1. Momentum decays from friction
    scaled_friction = tree_scale(-parameters.step_size, friction)
    momentum_decay = tree_multiply(scaled_friction, scaled_momentum)
    new_momentum = tree_add(state.momentum, momentum_decay)

    # 2. Momentum changes due to forces (gradient of stochastic potential)
    data_state, mini_batch = get_data(state.data_state, information=True)
    (pot, model_state), gradient = stochastic_gradient(
      new_positions,
      mini_batch,
      state=state.model_state)
    scaled_gradient = tree_scale(-parameters.step_size, gradient)
    new_momentum = tree_add(new_momentum, scaled_gradient)

    # 3. Injection of noise
    key, split = random.split(state.key)
    noise = random_tree(split, state.momentum)
    if noise_model:
      noise_state = update_noise_model(
        state.extra_fields,
        new_positions,
        gradient,
        friction,
        mini_batch=mini_batch,
        step_size=parameters.step_size)

      noise_correction = get_noise_model(
        noise_state,
        new_positions,
        gradient,
        friction,
        mini_batch=mini_batch,
        step_size=parameters.step_size,
        model_state=state.model_state)
      reduced_noise = tensor_matmul(noise_correction.cb_diff_sqrt, noise)
      scaled_noise = tree_scale(jnp.sqrt(2 * parameters.step_size), reduced_noise)
    else:
      noise_state = None
      scaled_noise = tree_scale(jnp.sqrt(2 * parameters.step_size), noise)
      scaled_noise = tree_multiply(friction, scaled_noise)
    new_momentum = tree_add(new_momentum, scaled_noise)

    new_state = LeapfrogState(
      potential=pot,
      key=key,
      positions=new_positions,
      momentum=new_momentum,
      data_state=data_state,
      model_state=model_state,
      extra_fields=noise_state)

    return new_state, None

  def init_fn(init_sample: PyTree,
              key: Array = None,
              batch_kwargs: Dict = None,
              init_model_state: PyTree = None):
    """Initializes the initial state of the integrator.

    Args:
      init_sample: Initial latent variables
      key: Initial PRNGKey
      batch_kwargs: Determine the initial state of the random data chain
      init_model_state: State of the model.

    Returns:
      Returns the initial state of the integrator.

    """

    # Initializing the initial state here makes it easier to add additional
    # variables which might be only necessary in special case
    if batch_kwargs is None:
      batch_kwargs = {}

    if noise_model:
      noise_state = init_noise_model(init_sample)
    else:
      noise_state = None

    reference_data_state = init_data(**batch_kwargs)

    # Only shape of momentum is important, as momentum is resampled in each
    # integration step
    if key is None:
      key = random.PRNGKey(0)

    momentum = init_sample

    init_state = LeapfrogState(
      potential=jnp.array(0.0),
      key=key,
      positions=init_sample,
      momentum=momentum,
      data_state=reference_data_state,
      model_state=init_model_state,
      extra_fields=noise_state)

    return init_state

  @partial(named_call, name='leapfrog_integration')
  def integrate(state: LeapfrogState,
                parameters: schedule,
                mass: PyTree = None
                ) -> LeapfrogState:

    # Use default values if mass matrix not provided
    if not mass:
      if const_mass:
        mass = const_mass
      else:
        mass = init_mass(tree_util.tree_map(jnp.ones_like, state.positions))

    # If the friction is scalar
    if tree_util.treedef_is_leaf(tree_util.tree_structure(friction)):
      multiscalar_friction = tree_util.tree_map(
        partial(jnp.full_like, fill_value=friction), state.positions)
    else:
      multiscalar_friction = friction

    # Resample momentum
    key, split = random.split(state.key)
    noise = random_tree(split, state.momentum)
    momentum = tensor_matmul(mass.sqrt, noise)

    # Do the leapfrog steps
    state = LeapfrogState(
      positions=state.positions,
      momentum=momentum,
      key=key,
      potential=jnp.array(0.0),
      model_state=state.model_state,
      data_state=state.data_state,
      extra_fields=state.extra_fields)
    final_state, _ = lax.scan(
      partial(_body_fun,
              parameters=parameters,
              mass=mass,
              friction=multiscalar_friction),
      state,
      onp.arange(steps))

    return final_state

  def get_fn(state: LeapfrogState) -> Dict[str, PyTree]:
    """Returns the latent variables."""
    return {"variables": state.positions,
            "energy": state.potential,
            "model_state": state.model_state}

  return init_fn, integrate, get_fn

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
  batch_init, batch_get, _ = batch_fn
  stochastic_gradient = value_and_grad(potential_fn, argnums=0, has_aux=True)

  # We need to define an update function. All array operations must be
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
    """Initializes the state of the integrator.

    Args:
      init_sample: Initial latent variables
      key: Initial PRNGKey
      adaption_kwargs: Determine the initial state of the adaption
      batch_kwargs: Determine the initial state of the random data chain

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

    init_state = LangevinState(
      key=key,
      latent_variables=init_sample,
      adapt_state=adaption_state,
      data_state=reference_data_state,
      model_state=init_model_state,
      potential=jnp.array(0.0),
      variance=jnp.array(1.0)
    )

    return init_state

  # Returns the important parameters of a state and excludes. Makes information
  # hiding possible

  def get_fn(state: LangevinState) -> Dict[str, PyTree]:
    """Returns the latent variables."""
    return {"variables": state.latent_variables,
            "likelihood": -state.potential,
            "model_state": state.model_state}

  # Update according to the integrator update rule

  @partial(named_call, name='langevin_diffusion_step')
  def update_fn(state: LangevinState,
                parameters: schedule):
    """Updates the integrator state according to a schedule.

    Args:
      state: Integrator state
      parameters: Schedule containing step_size and temperature

    Returns:
      Returns a new step calculated by applying langevin diffusion.
    """
    key, split = random.split(state.key)
    data_state, mini_batch = batch_get(state.data_state, information=True)

    noise = random_tree(split, state.latent_variables)
    (potential, (likelihoods, new_model_state)), gradient = stochastic_gradient(
      state.latent_variables,
      mini_batch,
      state=state.model_state,
      likelihoods=True)
    variance = jnp.var(likelihoods)

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

      adapted_gradient = tensor_matmul(manifold.g_inv, scaled_gradient)
      adapted_noise = tensor_matmul(manifold.sqrt_g_inv, scaled_noise)
      scaled_gamma = tree_scale(parameters.step_size, manifold.gamma.tensor)

      update_step = tree_add(
        tree_add(scaled_gamma, adapted_gradient),
        adapted_noise)

    # Conclude the variable update by adding the step to the current samples
    new_sample = tree_add(state.latent_variables, update_step)
    new_state = LangevinState(
      key=key,
      latent_variables=new_sample,
      adapt_state=adapt_state,
      data_state=data_state,
      model_state=new_model_state,
      potential=jnp.array(potential, dtype=state.potential.dtype),
      variance=variance)

    return new_state

  return init_fn, update_fn, get_fn
