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

The adaption gets three arguments passed:

1. parameters
2. stochastic gradient
3. reference data

Mostly, only the second argument is important, otherwise, argument 1. and 3.
allow to derive further quantities, such as the stochastic hessian of the
potential.

"""

import functools

from typing import Any, Optional, Callable, Tuple

from collections import namedtuple

from jax import tree_util, flatten_util

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
                             Tuple[PyTree, PyTree, PyTree]]]

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
      manifold, sqrt_manifold, gamma = get(state.state,
                                           x_flat,
                                           grad_flat,
                                           mini_batch,
                                           *get_args,
                                           **get_kwargs)

      # Unravel the results
      # Todo: Distinguish whether matrices or vectors are returned
      return state.unravel_fn(manifold), state.unravel_fn(sqrt_manifold), state.unravel_fn(gamma)
    return new_init, new_update, new_get
  return pytree_adaption

static_conditioning_state = namedtuple(
  "static_conditioning_state",
  ["matrix"]
)

def static_conditioning(sample, matrix
                        ) -> Tuple[Callable[[Any, Any], Tuple[Any, Any, Any]], Any]:
  """Condition the model by a static matrix.

  This is the default adaption. The simplest form is to choose the matrix to be
  the identity.

  Arguments:
    sample: Defines the shape of the samples.
    Gamma: The preconditining matrix. Must allow multiplication of the potential
      gradient with itself.

  Returns:
    Returns a function which adapts its internal states and returns the
    preconditioning matrix and the drift vector consisting of Gamme term and
    possible additional drifts.

  """

  # Get a zero pytree
