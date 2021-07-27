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

"""Adapt conditioning matrix

Initialize adaption

::

  state = adaption_init(init_sample, *args, **kwargs)


Update the adaption state

::

  state = adaption_update(state, sample, sample_grad, mini_batch; *args, **kwargs)

Get the adapted manifold

::

  manifold = adaption_get(state, sample, sample_grad, mini_batch, *args, **kwargs)

Mostly, only the second argument is important, otherwise, argument 1. and 3.
allow to derive further quantities, such as the stochastic hessian of the
potential.

"""

import functools

from typing import Any, Callable, Tuple, NamedTuple

from collections import namedtuple

from jax import tree_util, flatten_util, jit, named_call, lax, vmap, grad
import jax.numpy as jnp

from jax_sgmc.util import Array, Tensor, host_callback
from jax_sgmc.data import MiniBatch
from jax_sgmc.util import host_callback

# Todo: Correctly specify the return type

PartialFn = Any
PyTree = Any

Adaption = Callable[[Any, Any],
  Tuple[Callable[[Array, Any,Any], PyTree],
                 Callable[[PyTree, Array, Array, PyTree, Any, Any], PyTree],
                 Callable[[PyTree, Array, Array, PyTree, Any, Any],
                 Tuple[Array, Array, Array]]]]
AdaptionState = Tuple[PyTree, PartialFn, PartialFn]
AdaptionStrategy = [Callable[[PyTree], AdaptionState],
                    Callable[[AdaptionState, PyTree, PyTree, Any, Any],
                             AdaptionState],
                    Callable[[AdaptionState, PyTree, PyTree, Any, Any],
                             PyTree]]

# Todo: Initializing the adaption shold return an init_function, update_function
#       and get_function

adaption_state = namedtuple(
  "adaption_state",
  ["state",
   "ravel_fn",
   "unravel_fn",
   "flat_potential"])
"""State of the manifold adaption.

This tuple stores functions to ravel and unravel the parameter and gradient
pytree in addition to the adaption state.

Attributes:
  state: State of the adaption strategy
  ravel_fn: Jax-partial function to transform pytree to 1D array
  unravel_fn: Jax-partial function to undo ravelling of pytree
  flat_potential: Potential function on the flattened pytree
"""

class Manifold(NamedTuple):
  """Adapted manifold.

  Attributes:
    g_inv: Inverse manifold.
    sqrt_g_inv: Square root of inverse manifold.
    gamma: Diffusion to correct for positional dependence of manifold.

  """
  g_inv: Tensor
  sqrt_g_inv: Tensor
  gamma: Tensor

class MassMatrix(NamedTuple):
  """Mass matrix for HMC. """
  inv: PyTree
  sqrt: PyTree

class NoiseModel(NamedTuple):
  """Approximation of the gradient noise. """
  cb_diff_sqrt: PyTree
  b_sqrt: PyTree

# Todo: Make tree hashable and add caching
def get_unravel_fn(tree: PyTree):
  """Calculates the unravel function.

  Args:
    tree: Parameter pytree

  Returns:
    Returns a jax Partial object such that the function can be passed as
    valid argument.

  """
  _, unravel_fn = flatten_util.ravel_pytree(tree)
  return tree_util.Partial(jit(unravel_fn))

def adaption(quantity: namedtuple = tuple):
  return functools.partial(_adaption, quantity=quantity)

def _adaption(adaption_fn: Adaption, quantity: namedtuple = tuple):
  """Decorator to make adaption strategies operate on 1D arrays.

  Positional arguments are flattened while keyword arguments are passed
  unchanged.

  """

  @functools.wraps(adaption_fn)
  def pytree_adaption(*args, **kwargs) -> AdaptionStrategy:
    init, update, get = adaption_fn(*args, **kwargs)
    # Name call for debugging
    named_update = named_call(update, name='update_adaption_state')
    named_get = named_call(get, name='get_adapted_manifold')
    @functools.wraps(init)
    def new_init(x0: PyTree,
                 *init_args,
                 **init_kwargs) -> adaption_state:
      # Calculate the flattened state and the ravel and unravel fun
      ravel_fn = tree_util.Partial(
        jit(lambda tree: flatten_util.ravel_pytree(tree)[0]))
      unravel_fn = get_unravel_fn(x0)

      x0_flat = ravel_fn(x0)
      init_flat = map(ravel_fn, init_args)
      state = init(x0_flat, *init_flat, **init_kwargs)

      # Wrap the potential in a flatten function if potential is provided as
      # kwarg
      minibatch_potential = kwargs.get("minibatch_potential")
      if minibatch_potential is not None:
        @tree_util.Partial
        def flat_potential(sample, mini_batch, **kwargs):
          sample_tree = unravel_fn(sample)
          return minibatch_potential(sample_tree, mini_batch, **kwargs)
      else:
        flat_potential = None

      return adaption_state(state=state,
                            ravel_fn=ravel_fn,
                            unravel_fn=unravel_fn,
                            flat_potential=flat_potential)

    @functools.wraps(update)
    def new_update(state: adaption_state,
                   *update_args,
                   mini_batch: PyTree = None,
                   **update_kwargs):
      # Flat the params and the gradient
      flat_args = map(state.ravel_fn, update_args)

      # Update with flattened arguments
      if state.flat_potential is None:
        new_state = named_update(
          state.state,
          *flat_args,
          **update_kwargs)
      else:
        assert mini_batch, "Adaption strategy requires mini-batch"
        new_state = named_update(
          state.state,
          *flat_args,
          mini_batch,
          state.flat_potential,
          **update_kwargs)

      updated_state = adaption_state(
        state=new_state,
        ravel_fn=state.ravel_fn,
        unravel_fn=state.unravel_fn,
        flat_potential=state.flat_potential)
      return updated_state

    @functools.wraps(get)
    def new_get(state: adaption_state,
                *get_args,
                mini_batch: PyTree = None,
                **get_kwargs
                ) -> quantity:
      # Flat the params and the gradient
      flat_args = map(state.ravel_fn, get_args)

      # Get with flattened arguments
      if state.flat_potential is None:
        adapted_quantities = named_get(
          state.state,*flat_args, **get_kwargs)
      else:
        adapted_quantities = named_get(
          state.state, *flat_args,
          mini_batch=mini_batch, flat_potential=state.flat_potential,
          **get_kwargs)

      def unravel_quantities():
        for q in adapted_quantities:
          if q.ndim == 1:
            yield Tensor(ndim=1, tensor=state.unravel_fn(q))
          else:
            yield Tensor(ndim=2, tensor=q)

      return quantity(*unravel_quantities())
    return new_init, new_update, new_get
  return pytree_adaption

static_conditioning_state = namedtuple(
  "static_conditioning_state",
  ["matrix"]
)

# def static_conditioning(sample, matrix
#                         ) -> Tuple[Callable[[Any, Any], Tuple[Any, Any, Any]], Any]:
#   """Condition the model by a static matrix.
#
#   This is the default adaption. The simplest form is to choose the matrix to be
#   the identity.
#
#   Arguments:
#     sample: Defines the shape of the samples.
#     Gamma: The preconditining matrix. Must allow multiplication of the potential
#       gradient with itself.
#
#   Returns:
#     Returns a function which adapts its internal states and returns the
#     preconditioning matrix and the drift vector consisting of Gamme term and
#     possible additional drifts.
#
#   """
#
#   # Get a zero pytree

@adaption(quantity=Manifold)
def rms_prop() -> AdaptionStrategy:
  """RMSprop adaption.

  Adapt a diagonal matrix to the local curvature requiring only the stochastic
  gradient.

  Returns:
    Returns RMSprop adaption strategy.

  [1] https://arxiv.org/abs/1512.07666
  """

  def init(sample: Array,
           alpha: Array = 0.9,
           lmbd: Array = 1e-5):
    """Initializes RMSprop algorithm.

    Args:
      sample: Initial sample to derive the sample size
      alpha: Adaption speed
      lmbd: Stabilization constant

    Returns:
      Returns the inital adaption state
    """
    v = jnp.ones_like(sample)
    return (v, alpha, lmbd)

  def update(state: Tuple[Array, Array, Array],
             sample: Array,
             sample_grad: Array,
             *args: Any,
             **kwargs: Any):
    """Updates the RMSprop adaption.

    Args:
      state: Adaption state
      sample_grad: Stochastic gradient

    Returns:
      Returns adapted RMSprop state.
    """
    del sample, args, kwargs

    v, alpha, lmbd = state
    new_v = alpha * v + (1 - alpha) * jnp.square(sample_grad)
    return new_v, alpha, lmbd

  def get(state: Tuple[Array, Array, Array],
          sample: Array,
          sample_grad: Array,
          *args: Any,
          **kwargs: Any):
    """Calculates the current manifold of the RMSprop adaption.

    Args:
      state: Current RMSprop adaption state

    Returns:
      Returns a manifold tuple with ``ndim == 1``.
    """
    del sample, sample_grad, args, kwargs

    v, _, lmbd = state
    g = jnp.power(lmbd + jnp.sqrt(v), -1.0)
    return g, jnp.sqrt(g), jnp.zeros_like(g)

  return init, update, get


@adaption(quantity=MassMatrix)
def mass_matrix(diagonal=True, burn_in=1000):
  """Adapt the mass matrix for HMC.

  Args:
    diagonal: Restrict the adapted matrix to be diagonal
    burn_in: Number of steps in which the matrix should be updated

  Returns:
    Returns an adaption strategy for the mass matrix.

  """

  def _update_matrix(args):
    iteration, ssq, _, _ = args
    if diagonal:
      inv = ssq / iteration
      sqrt = jnp.sqrt(iteration / ssq)
      host_callback.id_print(inv, what="Covariance")
    else:
      inv = ssq / iteration
      eigw, eigv = jnp.linalg.eigh(ssq / iteration)
      # Todo: More effective computation
      sqrt = jnp.matmul(jnp.transpose(eigv), jnp.matmul(jnp.diag(jnp.sqrt(eigw)), eigv))
    return inv, sqrt

  def init(sample: Array, init_cov: Array):
    iteration = 0
    mean = jnp.zeros_like(sample)
    if diagonal:
      ssq = jnp.zeros_like(sample)
    else:
      ssq = jnp.zeros((sample.size, sample.size))

    if init_cov is None:
      init_cov = jnp.ones_like(sample)

    if diagonal:
      m_inv = init_cov
      m_sqrt = 1 / jnp.sqrt(init_cov)
    else:
      m_inv = jnp.diag(init_cov)
      m_sqrt = jnp.diag(1 / jnp.sqrt(init_cov))

    return iteration, mean, ssq, m_inv, m_sqrt

  def update(state: Tuple[Array, Array, Array, Array, Array],
             sample: Array,
             *args: Any,
             **kwargs: Any):
    del args, kwargs
    iteration, mean, ssq, m_inv, m_sqrt = state

    iteration += 1
    new_mean = (iteration - 1) / iteration * mean + 1 / iteration * sample
    if diagonal:
      ssq += jnp.multiply(sample - mean, sample - new_mean)
    else:
      ssq += jnp.outer(sample - mean, sample - new_mean)

    # Only update once
    new_m_inv, new_m_sqrt = lax.cond(
      iteration == burn_in,
      _update_matrix,
      lambda arg: (arg[2], arg[3]),
      (iteration, ssq, m_inv, m_sqrt))

    return iteration, new_mean, ssq, new_m_inv, new_m_sqrt

  def get(state: Tuple[Array, Array, Array, Array, Array]):
    _, _, _, m_inv, m_sqrt = state

    return m_inv, m_sqrt

  return init, update, get

@adaption(quantity=NoiseModel)
def fisher_information(minibatch_potential: Callable = None,
                       diagonal = True
                       ) -> AdaptionStrategy:
  """Adapt empirical fisher information.

  Use the empirical fisher information as a noise model for SGHMC. The empirical
  fisher information is approximated according to [1].

  Returns:
    Returns noise model approximation strategy.

  [1] https://arxiv.org/abs/1206.6380
  """
  assert minibatch_potential, "Fisher information requires potential function."

  def init(*args):
    return None

  def update(*args,
             **kwargs):
    del args, kwargs
    return None

  def get(state,
          sample: Array,
          sample_grad: Array,
          *args: Any,
          mini_batch: MiniBatch,
          flat_potential,
          friction: Array = jnp.array(1.0),
          step_size: Any = jnp.array(1.0),
          **kwargs: Any):
    del state, args, kwargs

    _, mb_information = mini_batch
    N = mb_information.observation_count
    n = mb_information.batch_size

    # Unscale the gradient to get mean
    sample_grad /= N

    def potential_at_obs(smp, obs_idx):
      _, (likelihoods, _) = flat_potential(smp, mini_batch, likelihoods=True)
      return likelihoods[obs_idx]

    grad_diff_idx = grad(potential_at_obs, argnums=0)

    @functools.partial(vmap, out_axes=0)
    def sqd(idx):
      if diagonal:
        return jnp.square(grad_diff_idx(sample, idx) - sample_grad)
      else:
        diff = grad_diff_idx(sample, idx) - sample_grad
        return jnp.outer(diff, diff)

    ssq = jnp.sum(sqd(jnp.arange(mb_information.batch_size)), axis=0)

    if diagonal:
      v = 1 / (n - 1) * ssq
      b = 0.5 * step_size * v

      # Correct for negative eigenvalues
      correction = friction - b
      smallest_positive = jnp.min(jnp.where(correction <= 0, jnp.inf, correction))
      positive_correction = jnp.where(correction <= 0, smallest_positive, correction)

      # Apply the corrections to b
      b_corrected = friction - positive_correction

      noise_scale_sqrt = jnp.sqrt(positive_correction)
      scale_sqrt = jnp.sqrt(b_corrected)
    else:
      v = 1 / (n - 1) * ssq
      b = 0.5 * step_size * v
      eigw_cb, eigv_cb = jnp.linalg.eigh(friction * jnp.eye(v.shape[0]) - b)
      noise_scale_sqrt = jnp.matmul(jnp.transpose(eigv_cb), jnp.matmul(jnp.diag(jnp.sqrt(eigw_cb)), eigv_cb))

      eigw_b, eigv_b = jnp.linalg.eigh(b)
      # Todo: More effective computation
      scale_sqrt = jnp.matmul(jnp.transpose(eigv_b), jnp.matmul(jnp.diag(jnp.sqrt(eigw_b)), eigv_b))

    return noise_scale_sqrt, scale_sqrt

  return init, update, get
