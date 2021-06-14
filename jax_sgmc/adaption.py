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

# Todo: Müssen gradient und sample noch ein weiteres mal übergeben werden?
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

from typing import Any, Optional, Callable, Tuple

from collections import namedtuple

from jax import tree_util, flatten_util
import jax.numpy as jnp

from jax_sgmc.util import Array

# Todo: Correctly specify the return type

PartialFn = tree_util.Partial
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
   "unravel_fn"])
"""State of the manifold adaption.

This tuple stores functions to ravel and unravel the parameter and gradient
pytree in addition to the adaption state.

Attributes:
  state: State of the adaption strategy
  ravel_fn: Jax-partial function to transform pytree to 1D array
  unravel_fn: Jax-partial function to undo ravelling of pytree
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
      return adaption_state(state=state,
                            ravel_fn=ravel_fn,
                            unravel_fn=unravel_fn)

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
      new_state = update(
        state.state, x_flat, grad_flat, mini_batch,
        *update_args, *update_kwargs
      )
      return adaption_state(state=new_state,
                            ravel_fn=state.ravel_fn,
                            unravel_fn=state.unravel_fn)

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
      g_inv, sqrt_g_inv, gamma = get(state.state,
                                     x_flat,
                                     grad_flat,
                                     mini_batch,
                                     *get_args,
                                     **get_kwargs)

      if g_inv.ndim == 1 and sqrt_g_inv.ndim == 1:
        unraveled_manifold = manifold(
          ndim=1,
          g_inv=state.unravel_fn(g_inv),
          sqrt_g_inv=sqrt_g_inv,
          gamma=gamma
        )

      # Unravel the results
      return manifold
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

def rms_prop():
  """RMSProp adaption."""

  def init(sample, alpha=0.9, lmbd=1e-5):
    v = jnp.ones_like(sample)
    g = jnp.ones_like(sample)
    return (v, g, alpha, lmbd)

  def update(state, sample, sample_grad, *unused_args, **unused_kwargs):
    v, g, alpha, lmbd = state
    new_v = alpha * v + (1 - alpha) * jnp.square(sample_grad)
    new_g = jnp.power(lmbd + jnp.sqrt(new_v), -1.0)
    return (new_v, new_g, alpha, lmbd)

  def get(state, sample, sample_grad, *unused_args, **unused_kwargs):
    v, g, alpha, lmbd = state
    return g, jnp.sqrt(g), jnp.zeros_like(g)

  return init, update, get
