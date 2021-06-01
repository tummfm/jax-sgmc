"""Defines types special to jax or this library. """

from typing import Any
from functools import partial

from jax import tree_map
import jax.numpy as jnp

Array = jnp.ndarray
PyTree = Any

# Probably not important

def tree_multiply(a: PyTree, b: PyTree) -> PyTree:
  """Maps elementwise product over two vectors.

  Args:
    a: First pytree
    b: Second pytree, must have the same shape as a

  Returns:
    Returns a PyTree obtained by an element-wise product of all PyTree leaves.

  """
  return tree_map(jnp.multiply, a, b)


def tree_scale(alpha: Array, a: PyTree) -> PyTree:
  """Scalar-Pytree product via tree_map.

  Args:
    alpha: Scalar
    a: Arbitrary PyTree

  Returns:
    Returns a PyTree with all leaves scaled by alpha.

  """
  @partial(partial, tree_map)
  def tree_scale_imp(x: PyTree):
    return alpha * x
  return tree_scale_imp(a)


def tree_add(a: PyTree, b: PyTree) -> PyTree:
  """Maps elementwise sum over PyTrees.

  Arguments:
    a: First PyTree
    b: Second PyTree with the same shape as a

  Returns:
    Returns a PyTree obtained by leave-wise summation.
  """
  @partial(partial, tree_map)
  def tree_add_imp(leaf_a, leaf_b):
    return leaf_a + leaf_b
  return tree_add_imp(a, b)
