"""Defines types special to jax.edited.bak or this library. """

from typing import Any
from functools import partial

from jax import tree_util
from jax import flatten_util
import jax.numpy as jnp

Array = jnp.ndarray
PyTree = Any

# Probably not important

def tree_multiply(tree_a: PyTree, tree_b: PyTree) -> PyTree:
  """Maps elementwise product over two vectors.

  Args:
    a: First pytree
    b: Second pytree, must have the same shape as a

  Returns:
    Returns a PyTree obtained by an element-wise product of all PyTree leaves.

  """
  return tree_util.tree_map(jnp.multiply, tree_a, tree_b)


def tree_scale(alpha: Array, tree: PyTree) -> PyTree:
  """Scalar-Pytree product via tree_map.

  Args:
    alpha: Scalar
    a: Arbitrary PyTree

  Returns:
    Returns a PyTree with all leaves scaled by alpha.

  """
  @partial(partial, tree_util.tree_map)
  def tree_scale_imp(x: PyTree):
    return alpha * x
  return tree_scale_imp(tree)


def tree_add(tree_a: PyTree, tree_b: PyTree) -> PyTree:
  """Maps elementwise sum over PyTrees.

  Arguments:
    a: First PyTree
    b: Second PyTree with the same shape as a

  Returns:
    Returns a PyTree obtained by leave-wise summation.
  """
  @partial(partial, tree_util.tree_map)
  def tree_add_imp(leaf_a, leaf_b):
    return leaf_a + leaf_b
  return tree_add_imp(tree_a, tree_b)


def tree_matmul(tree_mat: Array, tree_vec: PyTree):
  """Matrix tree product for LD on manifold.

  Arguments:
    tree_mat: Matrix to be multiplied with flattened tree
    tree_vec: Tree representing vector

  Returns:
    Returns the un-flattened product of the matrix and the flattened tree.
  """
  # Todo: Redefine without need for flatten util
  vec_flat, unravel_fn = flatten_util.ravel_pytree(tree_vec)
  return unravel_fn(jnp.matmul(tree_mat, vec_flat))


def pytree_list_transform(pytrees):

  treedef_list = tree_util.tree_structure([0 for _ in pytrees])
  treedef_single = tree_util.tree_structure(pytrees[0])

  def pytree_list_to_leaves(pytrees):
    """Transform a list of pytrees to allow pmap/vmap."""

    # Transpose the pytress, i. e. make a list (array) of leaves from a list of
    # pytrees. Only then vmap can be used to vectorize an operation over pytrees
    @partial(partial, tree_util.tree_map)
    def concatenate_leaves(*leaves):
      return jnp.concatenate([jnp.array([leaf]) for leaf in leaves], axis=0)

    return concatenate_leaves(*pytrees)

  def pytree_leaves_to_list(pytree):
    """Backtransform a pytree form pytree_list_to_leaves."""

    # We need to undo the concatenation along the first dimension
    @partial(partial, tree_util.tree_map)
    def split_leaves(leaf):
      return jnp.split(leaf, leaf.shape[0], axis=0)
    pytree_with_lists = split_leaves(pytree)

    # The pytree must be transposed
    pytrees = tree_util.tree_transpose(treedef_single,
                                       treedef_list,
                                       pytree_with_lists)

    # Each leaf is wrapped in an uneccessary list of length 1
    @partial(partial, tree_util.tree_map)
    def unlist(leaf):
      return leaf[0]

    return unlist(pytrees)

  return pytree_list_to_leaves, pytree_leaves_to_list
