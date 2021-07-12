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

from typing import Any, Callable, Tuple

from collections import namedtuple

from jax import tree_util, flatten_util, lax
import jax.numpy as jnp

from jax_sgmc.util import Array

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

manifold = namedtuple(
  "manifold",
  ["ndim",
   "g_inv",
   "sqrt_g_inv",
   "gamma"]
)
"""Adapted manifold.

Attributes:
  ndim: Diagonal or full adaption.
  g_inv: Adaption matrix for gradient
  sqrt_g_inv: Adaption matrix for noise
  gamma: Diffusion due to positional dependence of manifold
"""

covariance = namedtuple(
  "covariance",
  ["cov_sqrt",
   "inv_cov",
   "ndim"]
)

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
  return tree_util.Partial(unravel_fn)

def adaption(adaption_fn: Adaption):
  """Decorator to make adaption strategies operate on 1D arrays."""

  @functools.wraps(adaption_fn)
  def pytree_adaption(*args, **kwargs) -> AdaptionStrategy:
    init, update, get = adaption_fn(*args, **kwargs)
    @functools.wraps(init)
    def new_init(x0: PyTree,
                 *init_args,
                 **init_kwargs) -> adaption_state:
      # Calculate the flattened state and the ravel and unravel fun
      ravel_fn = tree_util.Partial(
        lambda tree: flatten_util.ravel_pytree(tree)[0]
      )
      unravel_fn = get_unravel_fn(x0)
      x0_flat = ravel_fn(x0)
      state = init(x0_flat, *init_args, **init_kwargs)

      # Wrap the potential in a flatten function if potential is provided as
      # kwarg
      minibatch_potential = kwargs.get("minibatch_potential")
      if minibatch_potential is not None:
        @tree_util.Partial
        def flat_potential(sample, mini_batch):
          sample_tree = unravel_fn(sample)
          return minibatch_potential(sample_tree, mini_batch)
      else:
        flat_potential = None

      return adaption_state(state=state,
                            ravel_fn=ravel_fn,
                            unravel_fn=unravel_fn,
                            flat_potential=flat_potential)

    @functools.wraps(update)
    def new_update(state: adaption_state,
                   x: PyTree,
                   grad_x: PyTree,
                   mini_batch: PyTree,
                   *update_args,
                   **update_kwargs):
      # Flat the params and the gradient
      x_flat = state.ravel_fn(x)
      grad_flat = state.ravel_fn(grad_x)

      # Update with flattened arguments
      if state.flat_potential is None:
        new_state = update(
          state.state, x_flat, grad_flat,
          *update_args, **update_kwargs
        )
      else:
        new_state = update(
          state.state, x_flat, grad_flat,
          mini_batch, state.flat_potential,
          *update_args, **update_kwargs)

      return adaption_state(state=new_state,
                            ravel_fn=state.ravel_fn,
                            unravel_fn=state.unravel_fn,
                            flat_potential=state.flat_potential)

    @functools.wraps(get)
    def new_get(state: adaption_state,
                x: PyTree,
                grad_x: PyTree,
                mini_batch: PyTree,
                *get_args,
                **get_kwargs):
      # Flat the params and the gradient
      x_flat = state.ravel_fn(x)
      grad_flat = state.ravel_fn(grad_x)

      # Get with flattened arguments
      if state.flat_potential is None:
        g_inv, sqrt_g_inv, gamma = get(state.state,
                                       x_flat,
                                       grad_flat,
                                       *get_args,
                                       **get_kwargs)
      else:
        g_inv, sqrt_g_inv, gamma = get(state.state,
                                       x_flat,
                                       grad_flat,
                                       mini_batch,
                                       state.flat_potential,
                                       *get_args,
                                       **get_kwargs)

      if g_inv.ndim == 1 and sqrt_g_inv.ndim == 1:
        unraveled_manifold = manifold(
          ndim=1,
          g_inv=state.unravel_fn(g_inv),
          sqrt_g_inv=state.unravel_fn(sqrt_g_inv),
          gamma=state.unravel_fn(gamma)
        )
      else:
        unraveled_manifold = manifold(
          ndim=2,
          g_inv=g_inv,
          sqrt_g_inv=sqrt_g_inv,
          gamma=state.unravel_fn(gamma)
        )

      # Unravel the results
      return unraveled_manifold
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

@adaption
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
             unused_sample: Array,
             sample_grad: Array,
             *unused_args: Any,
             **unused_kwargs: Any):
    """Updates the RMSprop adaption.

    Args:
      state: Adaption state
      sample_grad: Stochastic gradient

    Returns:
      Returns adapted RMSprop state.
    """

    v, alpha, lmbd = state
    new_v = alpha * v + (1 - alpha) * jnp.square(sample_grad)
    return new_v, alpha, lmbd

  def get(state: Tuple[Array, Array, Array],
          unused_sample: Array,
          unused_sample_grad: Array,
          *unused_args: Any,
          **unused_kwargs: Any):
    """Calculates the current manifold of the RMSprop adaption.

    Args:
      state: Current RMSprop adaption state

    Returns:
      Returns a manifold tuple with ``ndim == 1``.
    """
    v, _, lmbd = state
    g = jnp.power(lmbd + jnp.sqrt(v), -1.0)
    return g, jnp.sqrt(g), jnp.zeros_like(g)

  return init, update, get
