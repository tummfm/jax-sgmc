"""Adapt conditioning matrix """

from collections import namedtuple

from typing import Callable, Tuple, Any

from jax.tree_util import tree_map

# Todo: Correctly specify the return type.

# Todo: Initializing the adaption shold return an init_function, update_function
#       and get_function

static_conditioning = namedtuple(
  "static_conditioning",
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

  gamma = tree_map(lambda x: 0 * x, sample)

  def adapt(state: static_conditioning, *args, **kwargs
            ) -> Tuple[static_conditioning, Any, Any]:
    return state, matrix, gamma

  return adapt, None