"""Defines types special to jax or this library. """

from typing import Any
from functools import partial

from jax import tree_map
import jax.numpy as jnp

Array = jnp.ndarray
PyTree = Any

# Probably not important

@partial(partial, tree_map)
def tree_multiply(a: PyTree, b: PyTree) -> PyTree:
  """Maps elementwise product over two vectors.

  Args:
    a: First pytree
    b: Second pytree, must have the same shape as a

  Returns:
    Returns a PyTree obtained by an element-wise product of all PyTree leaves.

  """
  return jnp.multiply(a, b)


def tree_scale(alpha: Array, a: PyTree) -> PyTree:
  """Scalar-Pytree product via tree_map.

  Args:
    alpha: Scalar
    a: Arbitrary PyTree

  Returns:
    Returns a PyTree with all leaves scaled by alpha.

  """
  return tree_map(lambda x: alpha * x, a)

@partial(partial, tree_map)
def tree_add(a: PyTree, b: PyTree) -> PyTree:
  """Maps elementwise sum over PyTrees.

  Arguments:
    a: First PyTree
    b: Second PyTree with the same shape as a

  Returns:
    Returns a PyTree obtained by leave-wise summation.
  """
  return jnp.add(a, b)
